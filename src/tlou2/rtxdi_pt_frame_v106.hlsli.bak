// TLOU RT - ReSTIR PT's frame (v83; included by rtxdi_bridge.hlsli when TLOU_PT is defined: the passes of rtxdi_pt.hlsl only,
// compiled with 16-bit types for NVIDIA's PT headers): its constants, its reservoirs and the primary surfaces the paths start
// from, this frame's and last frame's. v84: + the spatial neighbours CGNS picked.

#include "Rtxdi/PT/ReSTIRPTParameters.h"

// The frame's ReSTIR PT constants: tlou_rt.cpp prepare_rtxdi_pt (rtxdi::ReSTIRPTContext's parameters), in the frame's draw
// table after the Ray Reconstruction block, a root CBV. NVIDIA's parameter structs keep the layout in C++ and HLSL alike.
// v84: the parameters named restirPT, as NVIDIA's sample names them (its spatial resampling reads g_Const.restirPT)
struct RtxdiPTFrame
{
	RTXDI_PTParameters restirPT;
	RTXDI_RuntimeParameters runtime;
	uint flags;             // PT_FLAG_*
	uint surfaceAt;         // byte offsets in Data: the primary surfaces (two planes by frame parity),
	uint neighborAt;        //   v84: the neighbours CGNS picked for the spatial resampling (4 per pixel; v83: first vertices),
	uint dupAt;             //   the duplication map (per pixel the sample's id, its count, its stagnancy)
	uint smoothAt;          //   and the smoothed stagnancy (two planes by frame parity)
	uint pad0;
	uint pad1;
	uint pad2;
};
ConstantBuffer<RtxdiPTFrame> g_pt : register(b2, space100);
#define PT_FLAG_HISTORY 1u      // last frame ran the passes at this size: its reservoirs are the temporal input
#define PT_FLAG_DUPMAP 2u       // ... and its duplication map counts are there (RAB_GetDuplicationMapCount)

// The path reservoirs (NVIDIA's packed format, three arrays: two for the resampling - by frame in temporal mode; v84 with the
// spatial pass the temporal pass's and the final ones - one keeping the initial samples for the decorrelation), a root UAV into
// the data buffer
RWStructuredBuffer<RTXDI_PackedPTReservoir> RtxdiPTReservoirs : register(u4, space100);
#define RTXDI_PT_RESERVOIR_BUFFER RtxdiPTReservoirs

// Last frame's camera relative to this frame's: what moves last frame's camera-relative positions into this frame's
float3 pt_camera_step()
{
	return asfloat(Data.Load3(FRAME_CAM_PREV)) - asfloat(Data.Load3(FRAME_CAM_CUR));
}

// ---- the primary surfaces ----
// The paths' starting points as cs_re9_path makes them - the camera ray re-traced onto the scene's triangle, the shading
// normal lifted onto its side, the lobes' albedos and the diffuse lobe's chance - 32 bytes per pixel: the position relative to
// that frame's camera and the view depth (0: none), the shading and the triangle's normals, Ad (gamma bytes) and the
// roughness, F0 (gamma bytes) and the diffuse lobe's chance. Every pass reads them back the same way (quantized): the paths
// are sampled and shifted on the same surfaces
#define PT_SURFACE_BYTES 32u
struct pt_record
{
	float3 p;
	float view_depth;
	float3 n;
	float3 ng;
	float3 ad;
	float roughness;
	float3 f0;
	float p_diffuse;
};
uint pt_surface_at(uint parity, uint2 px)
{
	const uint2 size = full_size();
	return g_pt.surfaceAt + parity * align_up(size.x * size.y * PT_SURFACE_BYTES, 256) + (px.y * size.x + px.x) * PT_SURFACE_BYTES;
}
void pt_record_pack(pt_record r, out uint4 a, out uint4 b)
{
	const float3 ad = pow(saturate(r.ad), 1.0 / 2.2), f0 = pow(saturate(r.f0), 1.0 / 2.2);
	a = uint4(asuint(r.p), asuint(r.view_depth));
	b = uint4(pack_normal(r.n), pack_normal(r.ng), unorm8(ad.x) | (unorm8(ad.y) << 8) | (unorm8(ad.z) << 16) | (unorm8(r.roughness) << 24),
		unorm8(f0.x) | (unorm8(f0.y) << 8) | (unorm8(f0.z) << 16) | (unorm8(r.p_diffuse) << 24));
}
pt_record pt_record_unpack(uint4 a, uint4 b)
{
	pt_record r;
	r.p = asfloat(a.xyz);
	r.view_depth = asfloat(a.w);
	r.n = unpack_normal(b.x);
	r.ng = unpack_normal(b.y);
	r.ad = pow(float3(b.z & 0xFFu, (b.z >> 8) & 0xFFu, (b.z >> 16) & 0xFFu) / 255.0, 2.2);
	r.roughness = (b.z >> 24) / 255.0;
	r.f0 = pow(float3(b.w & 0xFFu, (b.w >> 8) & 0xFFu, (b.w >> 16) & 0xFFu) / 255.0, 2.2);
	r.p_diffuse = (b.w >> 24) / 255.0;
	if (!all(isfinite(r.p)) || !(r.view_depth > 0.0 && r.view_depth < 1e6))
		r.view_depth = 0.0;                                 // the sky, never written, or broken
	return r;
}
void pt_record_store(uint parity, uint2 px, pt_record r)
{
	uint4 a, b;
	pt_record_pack(r, a, b);
	const uint at = pt_surface_at(parity, px);
	Data.Store4(at, a);
	Data.Store4(at + 16, b);
}
pt_record pt_record_load(uint parity, uint2 px)
{
	const uint at = pt_surface_at(parity, px);
	return pt_record_unpack(Data.Load4(at), Data.Load4(at + 16));
}
// A record as RTXDI's surface: last frame's moved into this frame's camera-relative space (it looks toward last frame's camera)
RAB_Surface pt_record_surface(pt_record r, bool previousFrame)
{
	RAB_Surface surface = RAB_EmptySurface();
	if (!(r.view_depth > 0.0))
		return surface;
	const float3 eye = previousFrame ? pt_camera_step() : 0.0;
	surface.worldPos = r.p + eye;
	surface.viewDir = normalize(eye - surface.worldPos);
	surface.viewDepth = r.view_depth;
	surface.normal = r.n;
	surface.geoNormal = r.ng;
	surface.material.diffuseAlbedo = r.ad;
	surface.material.specularF0 = r.f0;
	surface.material.roughness = r.roughness;
	surface.diffuseProbability = r.p_diffuse;
	return surface;
}
// RAB_GetGBufferSurface in the ReSTIR PT passes: the primary surfaces as the paths started from them
RAB_Surface pt_gbuffer_surface(int2 pixelPosition, bool previousFrame)
{
	const uint2 size = full_size();
	if (any(pixelPosition < 0) || pixelPosition.x >= int(size.x) || pixelPosition.y >= int(size.y))
		return RAB_EmptySurface();
	return pt_record_surface(pt_record_load((g_frame & 1u) ^ (previousFrame ? 1u : 0u), uint2(pixelPosition)), previousFrame);
}
