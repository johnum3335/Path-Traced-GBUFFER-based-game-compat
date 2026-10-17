// TLOU RT - NVIDIA's ReSTIR PT for the RE9 renderer (v83; user: "improve the path tracer, with ... 'ReSTIR PT foundation plus
// temporal reuse'"; v84, user: "restir pt enhanced should REPLACE current path reuse and add reuse over time, the duplication
// cap" and "spatial resampling cgns"). NVIDIA's RTXDI library (Include/Rtxdi/PT: its path tracer interface, initial sampling,
// temporal and spatial resampling with the hybrid shift, compatibility-guided neighbour selection, the duplication map, the
// decorrelation for DLSS Ray Reconstruction) is compiled in unmodified - NVIDIA RTX SDKs license, object code only: "This
// software contains source code provided by NVIDIA Corporation." - on the add-on's data through its bridge (rtxdi_bridge.hlsli,
// rtxdi_pt_frame.hlsli and the RAB_ functions below).
// The RE9 renderer's path is one bounce: the primary surface's BSDF sample (its two lobes, picked by their albedos), the ray,
// and the light leaving the hit toward the surface - the game's own light where the camera sees the hit, SHaRC's memory of it,
// else DD2's estimate there (re9.hlsl re9_dd2_bounce) - or the sky. To ReSTIR PT that light is the hit's emission (the path's
// light vertex at its first bounce) and the sky its environment; the path reconnects at the hit when the footprint rule finds
// the primary rough enough and the hit far enough (rcVertexLength 2), else it is replayed from the stored random numbers.
// The passes (tlou_rt.cpp post_lighting, before cs_re9_path, which then lights the frame directly and leaves the bounce here;
// v84: they replace v68's path reuse, reuse.hlsl, on this renderer - v83 ran it after them):
//   cs_pt_initial   - the primary surfaces (rtxdi_pt_frame.hlsli: the camera ray re-traced as cs_re9_path does), RTXDI's
//                     initial sampling (GenerateInitialSamples over RAB_PathTrace), the reservoir and its copy for the
//                     decorrelation; and last frame's reservoirs moved into this frame's camera-relative space (or emptied
//                     without a last frame)
//   cs_pt_neighbors - v84: CGNS, compatibility-guided neighbour selection (RTXDI_PTSpatialNeighborSelection, Junkins et al.):
//                     per pixel up to four neighbours drawn from 32 candidates in a 50-pixel disk by how alike their surfaces
//                     are (normals, positions against the pixel's footprint)
//   cs_pt_temporal  - RTXDI_PTTemporalResampling: last frame's path at the reprojected pixel shifted here (reconnection or
//                     replay), pairwise MIS with ray-traced bias correction; the duplication-based history reduction (ReSTIR
//                     PT Enhanced), then the boiling filter (no RR); without the spatial pass also the firefly flags for the
//                     decorrelation (RR)
//   cs_pt_spatial   - v84: RTXDI_PTSpatialResampling: the CGNS neighbours' paths (after their temporal resampling) shifted
//                     here and combined with pairwise MIS, more neighbours where the pixel has no history (the disocclusion
//                     boost); then the firefly flags
//   cs_pt_dupmap    - per pixel how many of its 17 x 17 neighbours carry the same sample (as RTXDI_PTComputeDuplicationMap)
//   cs_pt_smooth    - the samples' stagnancy smoothed over space and time (as RTXDI_PTComputeSmoothedDuplicationMap)
//   cs_pt_shade     - RTXDI_PTApplyDecorrelation (DLSS RR: stale or flagged pixels take their fresh initial sample), the
//                     reservoir's light split into the RE9 renderer's demodulated diffuse and specular samples and RR's hit
//                     distance (v84: from the reservoir; v83 kept the paths' first vertices for v68's path reuse)
#define TLOU_PT 1
#define RTXDI_ENABLE_BOILING_FILTER                         // the group's boiling filter / firefly flags (cs_pt_temporal)
#define RTXDI_BOILING_FILTER_GROUP_SIZE 8
#include "re9.hlsl"

#include "Rtxdi/Utils/BrdfRaySample.hlsli"
#include "Rtxdi/Utils/SampledLightData.hlsli"
#include "Rtxdi/Utils/Color.hlsli"
#include "Rtxdi/Utils/RandomSamplerPerPassSeeds.hlsli"

#define RAB_DISTANT_LIGHT_DISTANCE 10000.0  // metres: an escaped path's sky as RTXDI's environment vertex, this far along its ray
#define PT_SKY_DISTANCE 5000.0              // a reservoir vertex further than this from its surface is that sky vertex

// ---- the path tracer's bridge ----
// The trace result: a scene triangle (the draw's instance, the primitive) or nothing
struct RAB_RayPayload
{
	float committedRayT;
	uint hit;
	uint draw;
	uint primitive;
};
float RAB_RayPayloadGetCommittedHitT(RAB_RayPayload payload)
{
	return payload.committedRayT;
}

// What RAB_PathTrace needs of the pixel whose light the path brings (re9_dd2_bounce's terms: its baked ambient, its sky
// visibility, the lighting's factor, its roughness; v83 also kept the path's first vertex for v68's path reuse)
struct RAB_PathTracerUserData
{
	uint pathType;          // RTXDI_PTPathTraceInvocationType_*
	uint2 px;
	float3 a4;
	float3 a5;
	float3 axis;
	float open_k;
	float3 post;
	uint emit_on;
	float rough0;
};
void RAB_PathTracerUserDataSetPathType(inout RAB_PathTracerUserData ptud, uint type)
{
	ptud.pathType = type;
}
RAB_PathTracerUserData pt_user_data(uint2 px, float roughness)
{
	RAB_PathTracerUserData u = (RAB_PathTracerUserData)0;
	u.px = px;
	u.a4 = InT4[px].rgb;
	u.a5 = InT5[px].rgb;
	u.axis = decode_ambient_direction(GBuffer1[px]);
	const float2 uv = (float2(px) + 0.5) * c[8].zw;
	u.open_k = 1.0;                                         // v97: no game sky visibility (the Occlusion slider is gone)
	u.post = max(lighting_factor(px, Depth[px]) * g_brightness, 1e-4);
	u.emit_on = g_emit > 0.0 && Data.Load(emit_list_offset() + 8) != 0u ? 1u : 0u;
	u.rough0 = roughness;
	return u;
}

// The paths' BSDF at a primary surface, the RE9 renderer's lobes: the full BRDF x cos toward l - diffuse Ad cos / pi,
// specular F D G1(v) G1(l) / (4 N.V) (re9.hlsl re9_spec_integrand times As) - and its specular part apart
float3 pt_specular_times_nol(RAB_Surface s, float3 l)
{
	const float cosl = dot(s.normal, l);
	if (!(cosl > 0.0))
		return 0.0;
	const float ndv = max(dot(s.normal, s.viewDir), 1e-4);
	const float alpha = max(s.material.roughness * s.material.roughness, 0.002);
	const float3 h = normalize(s.viewDir + l);
	const float3 F = s.material.specularF0 + (1.0 - s.material.specularF0) * pow(1.0 - saturate(dot(s.viewDir, h)), 5.0);
	return F * (rtxdi_ggx_d(saturate(dot(s.normal, h)), alpha) * rtxdi_smith_g1(ndv, alpha) * rtxdi_smith_g1(cosl, alpha) / (4.0 * ndv));
}
float3 pt_bsdf_times_nol(RAB_Surface s, float3 l)
{
	const float cosl = dot(s.normal, l);
	if (!(cosl > 0.0))
		return 0.0;
	return s.material.diffuseAlbedo * (cosl * 0.31830989) + pt_specular_times_nol(s, l);
}
// The BSDF sample, as cs_re9_path's v44-v82: the lobe picked with the surface's chance (three numbers of the stream: the lobe,
// the direction), the cosine or GGX's visible normals; its pdf the mixture's (RAB_SurfaceEvaluateBrdfPdf)
RTXDI_BrdfRaySample pt_brdf_ray(RAB_Surface s, inout RTXDI_RandomSamplerState rng)
{
	const float u0 = RTXDI_GetNextRandom(rng);
	const float2 u = float2(RTXDI_GetNextRandom(rng), RTXDI_GetNextRandom(rng));
	const bool spec = u0 >= s.diffuseProbability;
	const float alpha = max(s.material.roughness * s.material.roughness, 0.002);
	RTXDI_BrdfRaySample brs = RTXDI_EmptyBrdfRaySample();
	brs.outDirection = spec ? reflect(-s.viewDir, sample_ggx_vndf(s.viewDir, s.normal, alpha, u)) : cosine_direction(s.normal, u);
	if (spec)
		brs.properties.SetSpecular();
	brs.outPdf = RAB_SurfaceEvaluateBrdfPdf(s, brs.outDirection);
	brs.brdfTimesNoL = pt_bsdf_times_nol(s, brs.outDirection);
	return brs;
}
float3 pt_finite(float3 v)
{
	return all(isfinite(v)) ? clamp(v, 0.0, 60000.0) : 0.0;
}

#include "Rtxdi/PT/Reservoir.hlsli"
#include "Rtxdi/PT/PathTracerContext.hlsli"

// NVIDIA's path tracer callback, the RE9 renderer's path in RTXDI's terms (RAB_PathTracer.hlsli of RTXDI's full sample: the
// same order of the context's calls). From the context's surface (the primary, bounce depth 2 at the hit): the BSDF sample
// from the replay stream, the ray off the triangle (cs_re9_path's origin), the cut-outs solid; a hit is recorded and its
// leaving light streamed as its emission (the hybrid shift asks for it at the replay's last vertex only), a miss streams the
// sky as the environment. The light's own random numbers (DD2's estimate: the sun's cone, the light picked, the lamps) come
// from the replay stream too: a shift replaying the path gets the same light the path had
template<typename PTContextType>
void RAB_PathTrace(inout RTXDI_PathTracerContext<PTContextType> ctx, inout RTXDI_PathTracerRandomContext ptRandContext, inout RAB_PathTracerUserData ptud)
{
	[loop]
	while (ctx.GetBounceDepth() <= ctx.GetMaxPathBounce())
	{
		ctx.BeginPathState();
		const RAB_Surface here = ctx.GetIntersectionSurface();
		ctx.SetBrdfRaySample(pt_brdf_ray(here, ptRandContext.replayRandomSamplerState));
		if (!ctx.ValidContinuationRayBrdfOverPdf())
			break;
		ctx.MultiplyPathThroughput(ctx.GetContinuationRayBrdfOverPdf());
		const float3 dir = ctx.GetBrdfRaySample().outDirection;
		RayDesc ray;
		ray.Origin = here.worldPos + here.geoNormal * (0.005 + length(here.worldPos) * 0.001);
		ray.Direction = dir;
		ray.TMin = 0.0;
		ray.TMax = DD2_TMAX;
		ctx.SetContinuationRay(ray);
		if (!ctx.AnalyzePathReconnectibilityBeforeTrace())
			break;
		uint rng = RTXDI_murmur3(ptRandContext.replayRandomSamplerState);
		// v87: the lobe that launched the ray, for NVIDIA SHaRC's query gate (sharc.hlsli sharc_query: 1 the diffuse lobe)
		RTXDI_BrdfRaySample launched = ctx.GetBrdfRaySample();
		const float lobe_rough = launched.properties.IsSpecular() ? here.material.roughness : 1.0;
		const scene_hit h = trace_cutouts(ray.Origin, dir, DD2_TMAX);
		RAB_RayPayload payload = (RAB_RayPayload)0;
		payload.committedRayT = h.hit ? h.t : 0.0;
		payload.hit = h.hit ? 1u : 0u;
		payload.draw = h.draw;
		payload.primitive = h.primitive;
		ctx.SetTraceResult(payload);
		float source;
		if (!h.hit)
		{
			const float3 sky = pt_finite(re9_dd2_bounce(ptud.px, ray.Origin, dir, h, ptud.a4, ptud.a5, ptud.axis, ptud.open_k, ptud.post,
				ptud.emit_on != 0u, ptud.rough0, lobe_rough, rng, source));
			ctx.RecordPathRadianceMiss(ptRandContext.initialRandomSamplerState);
			ctx.RecordEnvironmentMapLightSample(sky, here, ptRandContext.initialRandomSamplerState);
			break;
		}
		RAB_Surface hs = RAB_EmptySurface();
		hs.worldPos = ray.Origin + dir * h.t;
		float3 nh = hit_normal(h.draw, h.primitive);
		if (dot(nh, dir) > 0.0)
			nh = -nh;                                       // toward the ray
		hs.normal = nh;
		hs.geoNormal = nh;
		hs.viewDir = -dir;
		hs.viewDepth = 1.0;
		const float4 m = hit_material(h.draw, h.primitive);
		hs.material.diffuseAlbedo = m.rgb;
		hs.material.roughness = m.w >= 0.0 ? m.w : 0.5;
		hs.diffuseProbability = 1.0;
		ctx.RecordPathIntersection(hs);
		if (ctx.IsPathTerminated())
			break;
		if (ctx.ShouldSampleEmissiveSurfaces())
		{
			const float3 light = pt_finite(re9_dd2_bounce(ptud.px, ray.Origin, dir, h, ptud.a4, ptud.a5, ptud.axis, ptud.open_k, ptud.post,
				ptud.emit_on != 0u, ptud.rough0, lobe_rough, rng, source));
			ctx.RecordEmissiveLightSample(light, here, ptRandContext.initialRandomSamplerState);
		}
		if (ctx.IsPathTerminated())
			break;
		ctx.IncreaseBounceDepth();
	}
}
// The denoiser's callbacks (RTXDI's sample keeps its primary surface replacement there): nothing here
void RAB_ReconnectionDenoiserCallback(const RTXDI_PTReservoir neighborSample, const RAB_Surface rcPrevSurface, inout RAB_PathTracerUserData ptud)
{
}
void RAB_LastBounceDenoiserCallback(const float3 lightPos, const RAB_Surface preLightSurface, inout RAB_PathTracerUserData ptud)
{
}
// The paths take no NEE (their light is the hit's own): a light sample is never their last vertex
float RAB_GetMISWeightForNEE(uint lightIndex, const RAB_LightSample lightSample, float3 lightDirection, float lightSolidAnglePdf, float scatterPdf)
{
	return 1.0;
}
// The light from a vertex at samplePosition as the surface reflects it (the BSDF x cos): the reconnection's target function
float3 RAB_GetReflectedBsdfRadianceForSurface(float3 incomingRadianceLocation, float3 incomingRadiance, RAB_Surface surface)
{
	return incomingRadiance * pt_bsdf_times_nol(surface, normalize(incomingRadianceLocation - surface.worldPos));
}
float3 RAB_GetPTSampleTargetPdfForSurface(float3 samplePosition, float3 sampleRadiance, RAB_Surface surface)
{
	return max(RAB_GetReflectedBsdfRadianceForSurface(samplePosition, sampleRadiance, surface), 0.0);
}

// ---- the passes' planes (rtxdi_pt_frame.hlsli offsets) ----
// v84: CGNS's picks (RTXDI_SPATIAL_NEIGHBOR_SELECTION_BUFFER, NVIDIA's RWByteAddressBuffer: SPATIAL_HEURISTIC_MAX_NEIGHBORS
// packed pixels per pixel) - a view of the data buffer at its plane
struct pt_neighbor_view
{
	uint Load(uint at)
	{
		return Data.Load(g_pt.neighborAt + at);
	}
	void Store(uint at, uint value)
	{
		Data.Store(g_pt.neighborAt + at, value);
	}
};
static pt_neighbor_view g_pt_neighbor_view;
#define RTXDI_SPATIAL_NEIGHBOR_SELECTION_BUFFER g_pt_neighbor_view
// CGNS's surface at a pixel (the primary surfaces the paths start from): its normal, its position, its view depth (false: none)
bool RAB_GetNeighborSelectionSurface(int2 pixel, out float3 worldNormal, out float3 worldPos, out float viewDepth)
{
	const RAB_Surface s = pt_gbuffer_surface(pixel, false);
	worldNormal = s.normal;
	worldPos = s.worldPos;
	viewDepth = s.viewDepth;
	return RAB_IsSurfaceValid(s);
}
// The duplication map, per pixel 16 bytes: the final sample's id (its replay seed; 0 none), its duplication count / 255
// (cs_pt_dupmap: last frame's while cs_pt_temporal reads it, as RTXDI's), its stagnancy (age / 40); the smoothed stagnancy
// in two planes by frame parity (cs_pt_smooth)
uint pt_dup_at(uint2 px)
{
	return g_pt.dupAt + (px.y * full_size().x + px.x) * 16;
}
float2 pt_dup_load(uint2 px)
{
	return asfloat(Data.Load2(pt_dup_at(px) + 4));
}
uint pt_smooth_at(uint parity, uint2 px)
{
	const uint2 size = full_size();
	return g_pt.smoothAt + parity * align_up(size.x * size.y * 4, 256) + (px.y * size.x + px.x) * 4;
}
// As RTXDI_PTStoreDuplicationInputs (PT/DuplicationMap.hlsli, which writes textures the add-on does not have)
void pt_store_duplication_inputs(uint2 px, RTXDI_PTReservoir reservoir)
{
	const uint at = pt_dup_at(px);
	Data.Store(at, reservoir.weightSum == 0.0 ? 0u : reservoir.randomSeed);
	Data.Store(at + 8, asuint(min(1.0, float(reservoir.age) / 40.0)));
}
// RTXDI's duplication-based history reduction: how many of the 17 x 17 pixels around last frame's position carried the same
// sample last frame (0 without last frame's map)
uint RAB_GetDuplicationMapCount(int2 prevPixelPos)
{
	const uint2 size = full_size();
	if ((g_pt.flags & PT_FLAG_DUPMAP) == 0u || any(prevPixelPos < 0) || prevPixelPos.x >= int(size.x) || prevPixelPos.y >= int(size.y))
		return 0u;
	return uint(round(saturate(pt_dup_load(uint2(prevPixelPos)).x) * 255.0));
}

#include "Rtxdi/PT/InitialSampling.hlsli"
#include "Rtxdi/PT/TemporalResampling.hlsli"
#include "Rtxdi/PT/BoilingFilter.hlsli"
// v84: the spatial resampling and CGNS; NVIDIA's spatial resampling reads CGNS's neighbour count from its sample's constant
// buffer by name (g_Const.restirPT): here the frame's ReSTIR PT constants
#define g_Const g_pt
#include "Rtxdi/PT/SpatialNeighborSelection.hlsli"
#include "Rtxdi/PT/SpatialResampling.hlsli"

// The duplication map as NVIDIA's decorrelation reads it (RTXDI_PT_DUPLICATION_MAP .y the stagnancy,
// RTXDI_PT_SMOOTHED_DUPLICATION_MAP this frame's smoothed stagnancy): views of the data buffer
struct pt_dup_view
{
	float2 operator[](uint2 px)
	{
		return pt_dup_load(px);
	}
};
struct pt_smooth_view
{
	float operator[](uint2 px)
	{
		return asfloat(Data.Load(pt_smooth_at(g_frame & 1u, px)));
	}
};
static const pt_dup_view g_pt_dup_view;
static const pt_smooth_view g_pt_smooth_view;
#define RTXDI_PT_DUPLICATION_MAP g_pt_dup_view
#define RTXDI_PT_SMOOTHED_DUPLICATION_MAP g_pt_smooth_view
#include "Rtxdi/PT/Decorrelation.hlsli"

// ---- the passes ----
// The frame's ReSTIR PT counters for the status line (rt.hlsl FRAME_PT; cs_setup zeroes them): one atomic per wave
void pt_count(uint offset, uint4 v)
{
	v = WaveActiveSum(v);
	if (WaveIsFirstLane() && any(v != 0u))
	{
		Data.InterlockedAdd(FRAME_PT + offset, v.x);
		Data.InterlockedAdd(FRAME_PT + offset + 4, v.y);
		Data.InterlockedAdd(FRAME_PT + offset + 8, v.z);
		Data.InterlockedAdd(FRAME_PT + offset + 12, v.w);
	}
}
// The camera positions for RTXDI's footprints, in the add-on's camera-relative space: this frame's camera at the origin,
// last frame's at the camera's step (and the one before taken as the same, so RTXDI's translation term vanishes)
void pt_cameras(out float3 cur, out float3 prev, out float3 prev_prev)
{
	cur = 0.0;
	prev = pt_camera_step();
	prev_prev = prev;
}
// Where the pixel's surface was last frame, in pixels, and the change of its view depth (as rtxdi_di.hlsl rtxdi_motion)
float3 pt_motion(uint2 px, RAB_Surface surface)
{
	const float3 clip = previous_clip(px, surface.worldPos);
	if (!(clip.z > 0.0))
		return float3(1e5, 1e5, 0.0);                   // not on screen last frame: no temporal reuse
	const float2 fpos = float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float2(full_size()) - 0.5;
	return float3(fpos - float2(px), clip.z - surface.viewDepth);
}

// The primary surface the paths start from, as cs_re9_path makes it (v44-v82): hair facing the viewer with its own normal,
// else the camera ray's end re-traced onto the scene's triangle (RE9 RecastPrimaryRays) and its normal; the lobes' albedos;
// the shading normal lifted onto the triangle's side; the lobe chance from the albedos (a smooth surface's specular at least
// half the time, v70)
pt_record pt_primary(uint2 px, float d)
{
	pt_record r = (pt_record)0;
	if (d >= 1.0)
		return r;
	const bool hair = (pixel_flags(px) & PIXEL_HAIR) != 0;
	surface s = load_surface(px, d);
	if (hair)
		face_viewer(s);
	float3 ng = hair ? s.n : geometric_normal(px, s.p, s.n);
	if (!hair)
	{
		const float dist = length(s.p);
		const float3 dir = s.p / max(dist, 1e-6);
		const float window = RE9_RECAST + dist * 0.01;
		const float start = max(dist - max(window, RE9_RECAST_FRONT + dist * 0.05), 0.0);
		const scene_hit h = trace_scene(dir * start, dir, dist + window - start);
		if (h.hit)
		{
			s.p = dir * (start + h.t);
			const float3 nt = hit_normal(h.draw, h.primitive);
			ng = dot(nt, dir) > 0.0 ? -nt : nt;
		}
	}
	float3 ad, as, f0;
	re9_albedos(s, ad, as, f0);
	const float lad = restir_lum(ad), las = restir_lum(as);
	const float spec_floor = lerp(0.5, 0.1, saturate((s.roughness - 0.2) * 2.5));
	const float p_spec = lad + las > 0.0 ? clamp(las / (lad + las), spec_floor, 0.9) : 0.5;
	r.p = s.p;
	r.view_depth = s.view_depth;
	r.n = visible_normal(s.n, s.v, ng);
	r.ng = ng;
	r.ad = ad;
	r.roughness = s.roughness;
	r.f0 = f0;
	r.p_diffuse = 1.0 - p_spec;
	return r;
}

// Pass 1: this frame's primary surfaces and initial samples; last frame's final reservoirs moved into this frame's space
[numthreads(8, 8, 1)]
void cs_pt_initial(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint2 px = id.xy;
	const RTXDI_PTParameters P = g_pt.restirPT;
	// last frame's reservoirs (this frame's temporal input): its camera-relative positions into this frame's - every pixel its
	// own before the temporal pass reads any of them; without a last frame (the passes off, another size) they are emptied.
	// A path reconnecting at its hit takes the light leaving the hit as this frame's lighting has it - the game's light where
	// the camera sees the hit, else SHaRC (re9_dd2_bounce's first two answers; a hit off screen and not in SHaRC keeps last
	// frame's): RTXDI's temporal reuse keeps a path's light as it was found, and the bounce light of a light that moved (the
	// flashlight) would stay behind for as long as the path lives. The path's weight does not depend on its light: the reuse
	// stays unbiased with the light of now
	{
		const uint pointer = RTXDI_ReservoirPositionToPointer(P.reservoirBuffer, px, P.bufferIndices.temporalResamplingInputBufferIndex);
		RTXDI_PackedPTReservoir last = RTXDI_PT_RESERVOIR_BUFFER[pointer];
		if ((g_pt.flags & PT_FLAG_HISTORY) != 0u)
		{
			const float3 x = asfloat(last.Data0.xyz) + pt_camera_step();
			last.Data0.xyz = asuint(x);
			const uint rc = (last.Data2.w >> 8) & 0xFFu, path = (last.Data2.w >> 16) & 0xFFu;
			if (rc == 2u && path == 2u && length(x) < PT_SKY_DISTANCE * 0.5 && !isinf(asfloat(last.Data2.x)))
			{
				const float3 n = normalize(DecodeSnorm3x16(last.Data1.xy));
				float3 now;
				if (dd2_seen_light(x, n, now) || (sharc_nv_on() ? sharc_lookup(x, n, now) : sharc_get(x + c[9].xyz, n, now)))   // v87
					last.Data2.xyz = asuint(pt_finite(now));
			}
		}
		else
			last = (RTXDI_PackedPTReservoir)0;
		RTXDI_PT_RESERVOIR_BUFFER[pointer] = last;
	}
	const float d = Depth[px];
	pt_record rec = pt_primary(px, d);
	uint4 pa, pb;
	pt_record_pack(rec, pa, pb);
	const uint at = pt_surface_at(g_frame & 1u, px);
	Data.Store4(at, pa);
	Data.Store4(at + 16, pb);
	rec = pt_record_unpack(pa, pb);                         // the paths start from the surface as the passes read it back
	RTXDI_PTReservoir reservoir = RTXDI_EmptyPTReservoir();
	if (rec.view_depth > 0.0)
	{
		const RAB_Surface primary = pt_record_surface(rec, false);
		RAB_PathTracerUserData ptud = pt_user_data(px, rec.roughness);
		RTXDI_PathTracerRandomContext random = RTXDI_InitializePathTracerRandomContext(px, g_pt.runtime.frameIndex,
			RTXDI_PT_GENERATE_INITIAL_SAMPLES_RANDOM_SEED, RTXDI_PT_GENERATE_INITIAL_SAMPLES_REPLAY_RANDOM_SEED);
		RTXDI_PTInitialSamplingRuntimeParameters runtime = RTXDI_EmptyPTInitialSamplingRuntimeParameters();
		pt_cameras(runtime.cameraPos, runtime.prevCameraPos, runtime.prevPrevCameraPos);
		reservoir = GenerateInitialSamples(P.initialSampling, runtime, P.reconnection, random, primary, ptud);
	}
	RTXDI_StorePTReservoir(reservoir, P.reservoirBuffer, px, P.bufferIndices.initialPathTracerOutputBufferIndex);
	if (RTXDI_PTNeedsPreservedInitialSample(P.decorrelation))
		RTXDI_StorePTReservoir(reservoir, P.reservoirBuffer, px, P.bufferIndices.initialPathTracerPreservedBufferIndex);
}

// Pass 2 (v84): CGNS - NVIDIA's compatibility-guided neighbour selection (RTXDI_PTSpatialNeighborSelection): per pixel up to
// the spatial samples' count (at most SPATIAL_HEURISTIC_MAX_NEIGHBORS) neighbours by weighted reservoir sampling over 32
// candidates in a 50-pixel disk, each scored by how alike its surface is (the normals' dot product ^ 8 x its position's
// distance against the pixel's footprint), the search ending once enough good ones are found; the picks for cs_pt_spatial
[numthreads(8, 8, 1)]
void cs_pt_neighbors(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint count = g_pt.restirPT.spatialResampling.numSpatialSamples;
	RTXDI_PTSpatialNeighborSelection(id.xy, size, g_pt.runtime.frameIndex, count);
	uint found = 0u;                                        // the status line: the neighbours found
	for (uint k = 0u; k < min(count, SPATIAL_HEURISTIC_MAX_NEIGHBORS); ++k)
		found += g_pt_neighbor_view.Load(((id.y * size.x + id.x) * SPATIAL_HEURISTIC_MAX_NEIGHBORS + k) * 4u) != SPATIAL_HEURISTIC_INVALID_NEIGHBOR ? 1u : 0u;
	pt_count(32, uint4(0u, found, 0u, 0u));
}

// Pass 3: RTXDI's temporal resampling - the pixel's initial sample and last frame's path at the reprojected pixel (shifted
// here: reconnected at its vertex or replayed), the duplication map's history cap; then over the group the boiling filter
// (without RR). Without the spatial pass this is the last resampling: also the firefly flags (RR's decorrelation) and the
// duplication map's inputs - with it those are cs_pt_spatial's, and the reservoir's flag bit carries the disocclusion boost
// (no history here) to it instead
[numthreads(8, 8, 1)]
void cs_pt_temporal(uint3 id : SV_DispatchThreadID, uint3 gt : SV_GroupThreadID)
{
	const uint2 size = full_size();
	const bool inside = id.x < size.x && id.y < size.y;
	const uint2 px = min(id.xy, size - 1u);
	const RTXDI_PTParameters P = g_pt.restirPT;
	RTXDI_PTReservoir reservoir = RTXDI_EmptyPTReservoir();
	bool from_last = false;
	bool valid = false;
	if (inside)
	{
		const RAB_Surface primary = RAB_GetGBufferSurface(int2(px), false);
		if (RAB_IsSurfaceValid(primary))
		{
			valid = true;
			RTXDI_PTTemporalResamplingRuntimeParameters runtime = RTXDI_EmptyPTTemporalResamplingRuntimeParameters();
			runtime.pixelPosition = px;
			runtime.reservoirPosition = px;
			runtime.motionVector = pt_motion(px, primary);
			pt_cameras(runtime.cameraPos, runtime.prevCameraPos, runtime.prevPrevCameraPos);
			RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(px, g_pt.runtime.frameIndex, RTXDI_PT_TEMPORAL_RESAMPLING_RANDOM_SEED);
			RAB_PathTracerUserData ptud = pt_user_data(px, primary.material.roughness);
			reservoir = RTXDI_PTTemporalResampling(P.temporalResampling, runtime, P.hybridShift, P.reconnection, g_pt.runtime, P.reservoirBuffer, rng,
				P.bufferIndices, from_last, ptud);
		}
	}
	// over the group, every thread taking part (group-shared sums): the boiling filter empties a sample far brighter than the
	// group's (without RR), the decorrelation flags it for the shading's fresh sample (RR)
	if (P.boilingFilter.enableBoilingFilter != 0u)
		RTXDI_PTBoilingFilter(gt.xy, P.boilingFilter.boilingFilterStrength, reservoir);
	if (P.bufferIndices.temporalResamplingOutputBufferIndex == P.bufferIndices.finalShadingInputBufferIndex)
		RTXDI_PTDetectDecorrelationFireflies(gt.xy, reservoir, P.decorrelation, P.bufferIndices);
	if (!inside)
		return;
	if (RTXDI_PTNeedsDuplicationInputs(P.bufferIndices.temporalResamplingOutputBufferIndex, P))
		pt_store_duplication_inputs(px, reservoir);
	RTXDI_StorePTReservoir(reservoir, P.reservoirBuffer, px, P.bufferIndices.temporalResamplingOutputBufferIndex);
	const bool ok = valid && RTXDI_IsValidPTReservoir(reservoir) && reservoir.weightSum > 0.0;
	pt_count(0, uint4(valid ? 1u : 0u, from_last && ok ? 1u : 0u, 0u, 0u));
}

// Pass 4 (v84): RTXDI's spatial resampling - the pixel's path after the temporal pass and its CGNS neighbours' (more where it
// has no history: the disocclusion boost, the picks past CGNS's from RTXDI's random offsets), each shifted here (reconnected
// at its vertex, a shadow ray, or replayed) and combined with pairwise MIS (the pixel's path shifted to each neighbour); a path
// taken from a neighbour starts its age anew (NVIDIA's sample, with the duplication-based history reduction); then over the
// group the firefly flags (RR's decorrelation) and the duplication map's inputs
[numthreads(8, 8, 1)]
void cs_pt_spatial(uint3 id : SV_DispatchThreadID, uint3 gt : SV_GroupThreadID)
{
	const uint2 size = full_size();
	const bool inside = id.x < size.x && id.y < size.y;
	const uint2 px = min(id.xy, size - 1u);
	const RTXDI_PTParameters P = g_pt.restirPT;
	RTXDI_PTReservoir reservoir = RTXDI_EmptyPTReservoir();
	bool picked = false;
	if (inside)
	{
		const RAB_Surface primary = RAB_GetGBufferSurface(int2(px), false);
		if (RAB_IsSurfaceValid(primary))
		{
			RTXDI_PTSpatialResamplingRuntimeParameters runtime = RTXDI_EmptyPTSpatialResamplingRuntimeParameters();
			runtime.pixelPosition = px;
			runtime.reservoirPosition = px;
			pt_cameras(runtime.cameraPos, runtime.prevCameraPos, runtime.prevPrevCameraPos);
			runtime.viewportSize = size;
			RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(px, g_pt.runtime.frameIndex, RTXDI_PT_SPATIAL_RESAMPLING_RANDOM_SEED);
			RAB_PathTracerUserData ptud = pt_user_data(px, primary.material.roughness);
			reservoir = RTXDI_PTSpatialResampling(runtime, P.spatialResampling, P.hybridShift, P.reconnection, P.reservoirBuffer, P.bufferIndices,
				g_pt.runtime, rng, picked, ptud);
			if (picked && P.temporalResampling.enableAgeBasedRejection == 0u)
				reservoir.age = 0;
		}
	}
	RTXDI_PTDetectDecorrelationFireflies(gt.xy, reservoir, P.decorrelation, P.bufferIndices);
	if (!inside)
		return;
	if (RTXDI_PTNeedsDuplicationInputs(P.bufferIndices.spatialResamplingOutputBufferIndex, P))
		pt_store_duplication_inputs(px, reservoir);
	RTXDI_StorePTReservoir(reservoir, P.reservoirBuffer, px, P.bufferIndices.spatialResamplingOutputBufferIndex);
	pt_count(32, uint4(picked ? 1u : 0u, 0u, 0u, 0u));
}

// Pass 5: the duplication map (as RTXDI_PTComputeDuplicationMap) - groups of 16 x 16 load a 32 x 32 tile of the final samples'
// ids and count, per pixel, its 17 x 17 neighbours with the same sample
groupshared uint g_pt_ids[32][32];
[numthreads(16, 16, 1)]
void cs_pt_dupmap(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID)
{
	const int2 size = int2(full_size());
	const int2 base = int2(gid.xy) * 16 - 8;
	[unroll]
	for (int dy = 0; dy < 2; ++dy)
	{
		[unroll]
		for (int dx = 0; dx < 2; ++dx)
		{
			const int2 q = base + int2(int(gt.x) * 2 + dx, int(gt.y) * 2 + dy);
			g_pt_ids[gt.y * 2 + dy][gt.x * 2 + dx] = all(q >= 0) && all(q < size) ? Data.Load(pt_dup_at(uint2(q))) : 0u;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	const int2 px = base + int2(gt.xy) + 8;
	if (any(px >= size))
		return;
	const uint own = g_pt_ids[gt.y + 8][gt.x + 8];
	uint count = 0u;
	if (own != 0u)
	{
		[unroll]
		for (int ny = -8; ny <= 8; ++ny)
		{
			[unroll]
			for (int nx = -8; nx <= 8; ++nx)
				count += g_pt_ids[int(gt.y) + 8 + ny][int(gt.x) + 8 + nx] == own && (nx | ny) != 0 ? 1u : 0u;
		}
	}
	Data.Store(pt_dup_at(uint2(px)) + 4, asuint(saturate(float(count) / 255.0)));
}

// Pass 6: the stagnancy smoothed (as RTXDI_PTComputeSmoothedDuplicationMap): this frame's against last frame's smoothed
// value around the reprojected pixel (a 5 x 5 box, every second pixel), an exponential average
[numthreads(8, 8, 1)]
void cs_pt_smooth(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint2 px = id.xy;
	const float current = saturate(pt_dup_load(px).y);
	float smoothed = current;
	const RAB_Surface primary = RAB_GetGBufferSurface(int2(px), false);
	if (RAB_IsSurfaceValid(primary) && (g_pt.flags & PT_FLAG_DUPMAP) != 0u)
	{
		const int2 prev = int2(round(float2(px) + pt_motion(px, primary).xy));
		if (all(prev >= 0) && all(prev < int2(size)))
		{
			float sum = 0.0, n = 0.0;
			[unroll]
			for (int dy = -2; dy <= 2; ++dy)
			{
				[unroll]
				for (int dx = -2; dx <= 2; ++dx)
				{
					const int2 q = prev + int2(dx, dy) * 2;
					if (all(q >= 0) && all(q < int2(size)))
					{
						sum += asfloat(Data.Load(pt_smooth_at((g_frame & 1u) ^ 1u, uint2(q))));
						n += 1.0;
					}
				}
			}
			const float ema = saturate(g_pt.restirPT.decorrelation.decorrelationEmaFactor);
			smoothed = ema * current + (1.0 - ema) * (n > 0.0 ? sum / n : current);
		}
	}
	Data.Store(pt_smooth_at(g_frame & 1u, px), asuint(saturate(smoothed)));
}

// Pass 7: the final shading - the decorrelation (RR: a stale or flagged pixel takes its fresh initial sample), the
// reservoir's light (its target function x its weight: the light the path brings through the surface's BSDF) split by the
// lobes toward its direction into the RE9 renderer's demodulated diffuse and specular samples (re9.hlsl RE9_RAW_D / RE9_RAW_S,
// clamped as cs_re9_path's) and RR's specular hit distance (v84: from the reservoir - its reconnection vertex, the sky's
// DD2_TMAX, a replayed path's ray traced again; v83 read the first vertices it kept for v68's path reuse, which no longer
// runs after these passes)
[numthreads(8, 8, 1)]
void cs_pt_shade(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint2 px = id.xy;
	const RAB_Surface cur = RAB_GetGBufferSurface(int2(px), false);
	if (!RAB_IsSurfaceValid(cur))
		return;                                             // the sky: cs_re9_path writes its planes
	const RTXDI_PTParameters P = g_pt.restirPT;
	RTXDI_PTReservoir reservoir = RTXDI_LoadPTReservoir(P.reservoirBuffer, px, P.bufferIndices.finalShadingInputBufferIndex);
	{
		// the status line: the final sample before the decorrelation - its age, reconnected at its hit, replayed, the sky
		const bool ok = RTXDI_IsValidPTReservoir(reservoir) && reservoir.weightSum > 0.0;
		const bool rc = ok && reservoir.rcVertexLength <= reservoir.pathLength;
		const bool sky = rc && length(reservoir.translatedWorldPosition - cur.worldPos) > PT_SKY_DISTANCE;
		pt_count(16, uint4(ok ? reservoir.age : 0u, rc && !sky ? 1u : 0u, ok && !rc ? 1u : 0u, sky ? 1u : 0u));
	}
	const bool firefly = RTXDI_GetFireflyDetected(reservoir);
	const float factor = RTXDI_PTApplyDecorrelation(px, px, reservoir, P.decorrelation, P.bufferIndices, P.reservoirBuffer, g_pt.runtime.frameIndex);
	bool swapped = false;
	if (factor > 0.0)
	{
		RTXDI_RandomSamplerState draw = RTXDI_InitRandomSampler(px, g_pt.runtime.frameIndex, RTXDI_PT_FINAL_SHADING_RANDOM_SEED);
		swapped = RTXDI_GetNextRandom(draw) < factor;       // the same draw as RTXDI_PTApplyDecorrelation's
	}
	// the lobes' albedos as the composite modulates (re9_albedos on the G-buffer surface)
	const float d = Depth[px];
	surface s = load_surface(px, d);
	if ((pixel_flags(px) & PIXEL_HAIR) != 0)
		face_viewer(s);
	float3 ad, as, f0;
	re9_albedos(s, ad, as, f0);
	float3 sd = 0.0, ss = 0.0;
	float hit_t = 0.0;
	if (RTXDI_IsValidPTReservoir(reservoir) && reservoir.weightSum > 0.0)
	{
		// the sample's direction and RR's hit distance: toward its reconnection vertex (the hit, or the sky's far vertex), or the
		// replayed BSDF sample of its random numbers and that ray traced again
		float3 l;
		if (reservoir.rcVertexLength <= reservoir.pathLength)
		{
			const float3 v = reservoir.translatedWorldPosition - cur.worldPos;
			const float len = length(v);
			l = v / max(len, 1e-6);
			hit_t = len > PT_SKY_DISTANCE ? DD2_TMAX : len;
		}
		else
		{
			RTXDI_RandomSamplerState replay = RTXDI_GetRngForShading(reservoir);
			l = pt_brdf_ray(cur, replay).outDirection;
			const scene_hit h = trace_cutouts(cur.worldPos + cur.geoNormal * (0.005 + length(cur.worldPos) * 0.001), l, DD2_TMAX);
			hit_t = h.hit ? h.t : DD2_TMAX;
		}
		const float3 b = pt_bsdf_times_nol(cur, l);
		const float3 C = pt_finite(reservoir.targetFunction * reservoir.weightSum);
		const float cosl = dot(cur.normal, l);
		if (all(isfinite(l)) && cosl > 0.0 && any(b > 0.0))
		{
			const float3 LW = C / max(b, 1e-8);                 // the light times its weight, per channel
			sd = LW * (cosl * 0.31830989);
			ss = LW * pt_specular_times_nol(cur, l) / max(as, 1e-4);
		}
	}
	// cs_re9_path's clamp (v70): the light the sample adds to the pixel against RE9_MAX times the albedos' luminance
	const float lum = restir_lum(ad * sd + as * ss), lmax = re9_clamp_limit(ad, as);
	float kmax = 1.0;
	if (!(lum <= lmax))
		kmax = lum > lmax ? lmax / lum : 0.0;
	sd *= kmax;
	ss *= kmax;
	re9_store(RE9_RAW_D, px, float4(max(sd, 0.0), 1.0));
	re9_store(RE9_RAW_S, px, float4(max(ss, 0.0), hit_t));
	pt_count(0, uint4(0u, 0u, swapped ? 1u : 0u, firefly ? 1u : 0u));
}
