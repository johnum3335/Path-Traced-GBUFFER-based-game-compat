// TLOU RT - NVIDIA's RTXDI library in the add-on's shaders (v77; included by re9.hlsl after rt.hlsl, so the RE9 and DD2
// renderers, the path reuse and the RTXDI passes of rtxdi_di.hlsl all have it). NVIDIA's files (RTGI_builds\deps\
// rtxdi-library) are used unmodified: this file sets the library's switches, declares the resources it reads (root
// descriptors in space 100, bound by tlou_rt.cpp bind_own), includes the add-on's bridge (rtxdi_bridge.hlsli) and then the
// library's sampling code.
// v76: ReSTIR DI for the lamps (rtxdi_di.hlsl cs_rtxdi_di). v77 (user: "rtxdi not only fixes all meshes issues, it also
// improves game graphics by alot aswell, go ahead with next version"): presampling and ReGIR - the lamps' power as a
// mipmapped PDF (cs_rtxdi_pdf), RIS tiles drawn from it (cs_rtxdi_presample: power-proportional lamp picks) and ReGIR's onion
// of world-space cells around the camera, each holding lamps picked for that cell (cs_regir_build). ReSTIR DI draws its
// candidates from the pixel's cell, and the paths' bounce hits light themselves from their cell (rtxdi_lamps_at) in place of
// emitter_irradiance's uniform picks over the whole list.

#define RTXDI_ENABLE_PRESAMPLING 1
#define RTXDI_REGIR_MODE RTXDI_REGIR_ONION                  // tlou_rt.cpp's ReGIRContext: the onion (cells grow with the distance)
#include "Rtxdi/RtxdiTypes.h"
#include "Rtxdi/DI/ReSTIRDIParameters.h"
#include "Rtxdi/ReGIR/ReGIRParameters.h"
#include "Rtxdi/LightSampling/RISBufferSegmentParameters.h"
#include "Rtxdi/Utils/RandomSamplerState.hlsli"

// The frame's RTXDI constants: tlou_rt.cpp rtxdi_frame, in the frame's draw table, a root CBV. NVIDIA's parameter structs keep
// the constant-buffer layout in C++ and HLSL alike
struct RtxdiFrame
{
	RTXDI_RuntimeParameters runtime;
	RTXDI_ReservoirBufferParameters reservoirs;
	RTXDI_DIInitialSamplingParameters initial;
	RTXDI_DISpatioTemporalResamplingParameters spatiotemporal;
	RTXDI_RISBufferSegmentParameters localRis;              // the lamps' RIS tiles (power-proportional picks)
	ReGIR_Parameters regir;
	uint input;             // the DI reservoir array last frame wrote (the temporal input)
	uint output;            // this frame's
	uint reuse;             // 0: this frame's samples alone ("Reuse light samples" off)
	uint flags;             // RTXDI_FRAME_*: what was built this frame
	uint pdfAt;             // byte offset of the lamps' PDF mip chain in Data (cs_rtxdi_pdf)
	uint pdfSize;           // its mip 0 is pdfSize x pdfSize (EMIT_MAX lamps)
	uint offsetsAt;         // byte offset of the neighbour offsets in Table (rtxdi::FillNeighborOffsetBuffer: int8 pairs)
	uint hitSamples;        // lamp candidates per bounce hit (rtxdi_lamps_at)
	uint regirSlots;        // ReGIR's light slots (cells x lights per cell: cs_regir_build's threads)
	uint gameAt;            // v86: byte offset in Data of the game's lights for RTXDI (cs_rtxdi_game: two frames' tables)
	uint regirPlainSlots;    // v92: a cell's first slots with one power-proportional pick each (variety: rtxdi_di.hlsl regir_build_mixed)
	uint gameShare;          // v96: (float) the ReGIR cells' candidates drawn evenly among the game's lights (regir_source)
};
ConstantBuffer<RtxdiFrame> g_rtxdi : register(b1, space100);
#define RTXDI_FRAME_PRESAMPLED 1u                           // the lamps' PDF and RIS tiles are this frame's
#define RTXDI_FRAME_REGIR 2u                                // ReGIR's cells are this frame's
#define RTXDI_FRAME_GAME 4u                                 // v86: the game's lights are in RTXDI's list (after the lamps)
#define RTXDI_FRAME_PHYSICAL 8u                             // v99: the RTXPT renderer's Physical light falloff (the game's lights
                                                            //   as sphere lights everywhere RTXDI weighs them: rt.hlsl light_arrival_ex)
#define RTXDI_FRAME_EXACT_GAME 16u                          // v104: the RTXPT renderer lights the camera's surfaces with every game
                                                            //   light itself - RTXDI's ReSTIR DI there takes the lamps alone
#define RTXDI_FRAME_EXACT_LAMPS 32u                         // v111: this frame's exact lamps per camera pixel are stored (rtxpt.hlsl
                                                            //   cs_rtxpt_rank) - ReSTIR DI's target skips them (rtxdi_bridge.hlsli)
#define RTXDI_FRAME_EXACT_PREV 64u                          // v111: ... and last frame's (its surfaces in ReSTIR DI's temporal reuse)

// The lamps' PDF "texture" (RTXDI_TEX2D: only the presampling's mip descent reads one, through RTXDI_TEX2D_LOAD's t.Load):
// a mip chain of floats in the data buffer, mip 0 the lamps' power at their Z-curve texels, each mip the next one's 2 x 2 mean
struct rtxdi_pdf_texture
{
	uint unused;
	float4 Load(int3 p)
	{
		const uint size = g_rtxdi.pdfSize >> uint(p.z);
		if (p.x < 0 || p.y < 0 || uint(p.x) >= size || uint(p.y) >= size)
			return 0.0;
		uint at = g_rtxdi.pdfAt;
		for (uint m = 0; m < uint(p.z); ++m)
			at += (g_rtxdi.pdfSize >> m) * (g_rtxdi.pdfSize >> m) * 4u;
		return asfloat(Data.Load(at + (uint(p.y) * size + uint(p.x)) * 4u));
	}
};
#undef RTXDI_TEX2D
#define RTXDI_TEX2D rtxdi_pdf_texture

RWStructuredBuffer<RTXDI_PackedDIReservoir> RtxdiReservoirs : register(u2, space100);   // v76: ReSTIR DI's reservoirs
RWStructuredBuffer<uint2> RtxdiRis : register(u3, space100);                            // v77: the RIS buffer (lamp index, 1 / pdf)
#define RTXDI_LIGHT_RESERVOIR_BUFFER RtxdiReservoirs
#define RTXDI_RIS_BUFFER RtxdiRis
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER g_rtxdi_offsets
#include "rtxdi_bridge.hlsli"

#include "Rtxdi/DI/InitialSampling.hlsli"
#include "Rtxdi/DI/SpatioTemporalResampling.hlsli"
#include "Rtxdi/LightSampling/PresamplingFunctions.hlsli"

// v96 (user, v95's capture: the remembered lamps - the table lamp's shade x Lamps 95 - held 2/3 of the lights' power and the
// game's blue fill light that reaches the wall 0.35%; "reducing lamp and increasing candidates does reduce the blotch
// effect"): the ReGIR cells' candidates (the onion's regir_build_mixed, the surface cells' cs_regir_fill) from a mixture -
// with probability g_rtxdi.gameShare a pick among this frame's game lights alike, else RTXDI's power-proportional one (a tile
// shared by the cells of a group: the fill light in a few of them, absent from the rest) - each weighed by its pdf under the
// mixture: its share of the PDF's power x (1 - share), + share / the game's lights for one of them. A light dim in power
// that reaches the cell comes up in every cell, with weights no longer 1 / 0.35% - the cell-sized blotches that Ray
// Reconstruction and the temporal reuse turned into a pulse. Without game lights, or a share of 0: v95's picks
struct regir_source
{
	RTXDI_LocalLightSelectionContext ctx;
	bool power;             // the context picks by power (RTXDI's RIS tiles; else uniformly: its own 1 / pdf, no mixture)
	uint gameFirst, gameCount;
	float share, powerTotal;
};
float rtxdi_power_of(uint light)                            // a light's power in the PDF's mip 0 (cs_rtxdi_pdf)
{
	const uint2 t = RTXDI_LinearIndexToZCurve(light);
	return asfloat(Data.Load(g_rtxdi.pdfAt + (t.y * g_rtxdi.pdfSize + t.x) * 4u));
}
float rtxdi_power_total()                                   // the PDF's 1 x 1 mip (the mean) x its texels
{
	uint at = g_rtxdi.pdfAt;
	for (uint s = g_rtxdi.pdfSize; s > 1u; s >>= 1)
		at += s * s * 4u;
	return asfloat(Data.Load(at)) * float(g_rtxdi.pdfSize * g_rtxdi.pdfSize);
}
regir_source regir_source_init(RTXDI_LocalLightSelectionContext ctx, bool power)
{
	regir_source s;
	s.ctx = ctx;
	s.power = power;
	s.gameFirst = rtxdi_lamps_used(false);
	s.gameCount = rtxdi_game_count(false);
	s.powerTotal = power ? rtxdi_power_total() : 0.0;
	s.share = power && s.gameCount != 0u && s.powerTotal > 0.0 ? saturate(asfloat(g_rtxdi.gameShare)) : 0.0;
	return s;
}
// A light's pdf under the mixture (v99: apart, for RTXPT's NEE - its candidates weighed under both of its distributions)
float regir_source_pdf(regir_source s, uint light)
{
	const bool game = light >= s.gameFirst && light - s.gameFirst < s.gameCount;
	const float pdf = (s.powerTotal > 0.0 ? (1.0 - s.share) * max(rtxdi_power_of(light), 0.0) / s.powerTotal : 0.0) +
		(game ? s.share / float(s.gameCount) : 0.0);
	return pdf > 0.0 && isfinite(pdf) ? pdf : 0.0;
}
// A candidate: its light (index, info) and 1 / its pdf under the mixture (0: none)
void regir_source_next(inout regir_source s, inout RTXDI_RandomSamplerState rng, out uint light, out RAB_LightInfo info, out float invPdf)
{
	const float u = RTXDI_GetNextRandom(rng);
	float invSource = 0.0;
	if (u < s.share)
	{
		light = s.gameFirst + min(uint(u / s.share * float(s.gameCount)), s.gameCount - 1u);
		info = RAB_LoadLightInfo(light, false);
	}
	else
		RTXDI_SelectNextLocalLight(s.ctx, RTXDI_GetNextRandom(rng), info, light, invSource);
	if (!s.power)
	{
		invPdf = invSource;
		return;
	}
	const float pdf = regir_source_pdf(s, light);
	invPdf = pdf > 0.0 ? 1.0 / pdf : 0.0;
}

// v90: ReGIR's surface cells' pick for a bounce hit (regir_cells.hlsli, included after the SHARC SDK's types)
bool regir_cells_pick(RAB_Surface surface, uint seed, out RAB_LightSample sample, out float inv_pdf);

// The light a path's bounce hit takes (p relative to the camera, n its normal, toward v; ReGIR this frame): NVIDIA's
// world-space sampling - RTXDI_SampleLocalLights over the hit's ReGIR cell (the power-proportional tiles outside the onion),
// g_rtxdi.hitSamples candidates - and one shadow ray for the pick: true with the pick (the surface as RTXDI took it, the
// sample, its 1 / pdf) when it is seen. v82: the hit's own material (the material cache: albedo, metalness, F0, roughness)
// in RTXDI's target. v86: with the game's lights in RTXDI's list (RTXDI_FRAME_GAME) the cells hold them too - the pick is a
// lamp or a game's light, one shadow ray either way. post: the path's pixel's lighting factor x Brightness (RTXDI's target
// takes the game's lights x it, as the lamps' light). v98: apart from rtxdi_lamps_at's shading (the RTXPT renderer shades
// the pick with RTXPT's BSDF - rtxpt.hlsl pt_lights_at)
bool rtxdi_pick_at(float3 p, float3 n, float3 v, float3 albedo, float metal, float3 f0, float roughness, uint seed, float3 post,
	out RAB_Surface surface, out RAB_LightSample sample, out float inv_pdf)
{
	surface = RAB_EmptySurface();
	sample = RAB_EmptyLightSample();
	inv_pdf = 0.0;
	const uint count = rtxdi_light_count(false);
	if (count == 0u)                                        // (v77-v85: also with Lamps 0 - rtxdi_lamp_count is 0 then)
		return false;
	surface.worldPos = p;
	surface.viewDir = v;
	surface.viewDepth = 1.0;
	surface.normal = n;
	surface.geoNormal = n;
	surface.material.diffuseAlbedo = albedo * (1.0 - metal);
	surface.material.specularF0 = f0;
	surface.material.roughness = roughness;
	surface.px = uint2(RTXDI_HIT_PIXEL, RTXDI_HIT_PIXEL);  // v86: a bounce hit (the game's lights: the hit's rules and taps)
	surface.post = post;
	surface.base = albedo;
	surface.metal = metal;
	// v90: the hit's surface cell (regir_cells.hlsli) when one filled this frame holds it, else NVIDIA ReGIR's onion
	if (!regir_cells_pick(surface, seed, sample, inv_pdf))
	{
		RTXDI_RandomSamplerState rng = RTXDI_CreateRandomSamplerFromDirectSeed(pcg(seed + 0x7E61u), 1u);
		RTXDI_RandomSamplerState cell_rng = RTXDI_CreateRandomSamplerFromDirectSeed(pcg(seed + 0x5C3Bu), 1u);
		RTXDI_DIInitialSamplingParameters params = g_rtxdi.initial;
		params.numLocalLightSamples = g_rtxdi.hitSamples;
		params.numInfiniteLightSamples = 0u;
		params.numEnvironmentSamples = 0u;
		params.numBrdfSamples = 0u;
		RTXDI_LightBufferRegion lamps = (RTXDI_LightBufferRegion)0;
		lamps.numLights = count;
		const RTXDI_DIReservoir reservoir = RTXDI_SampleLocalLights(rng, cell_rng, surface, params, RTXDI_ComputeInitialSamplingMisData(params),
			ReSTIRDI_LocalLightSamplingMode_REGIR_RIS, lamps, g_rtxdi.localRis, g_rtxdi.regir, sample);
		if (RTXDI_IsValidDIReservoir(reservoir))
			inv_pdf = RTXDI_GetDIReservoirInvPdf(reservoir);
	}
	return inv_pdf > 0.0 && RAB_GetConservativeVisibility(surface, sample);
}

// The lamps' light a surface at a path's bounce hit reflects toward v (the caller divides by the lighting's factor): the pick
// of rtxdi_pick_at - its diffuse and, with rtxdi_reflected, its specular part (v77-v81 a white diffuse surface; the caller
// multiplied by the albedo) - clamped as emitter_irradiance's. Without ReGIR this frame: emitter_irradiance's uniform picks
// over the whole list (rt.hlsl), as v39-v76, diffuse. v86: the callers drop their own uniform pick of the game's lights
// while the cells hold them (rtxdi_game_in_regir)
float3 rtxdi_lamps_at(float3 p, float3 n, float3 v, float3 albedo, float metal, float3 f0, float roughness, uint seed, float3 post)
{
	if ((g_rtxdi.flags & RTXDI_FRAME_REGIR) == 0u)
		return albedo * (1.0 - metal) * (0.31830989 * g_emit) * emitter_irradiance(p, n, n, seed);
	RAB_Surface surface;
	RAB_LightSample sample;
	float inv_pdf;
	if (!rtxdi_pick_at(p, n, v, albedo, metal, f0, roughness, seed, post, surface, sample, inv_pdf))
		return 0.0;
	float3 light = rtxdi_reflected(sample, surface) * inv_pdf;
	const float le = restir_lum(light), cap = 0.31830989 * g_emit * EMIT_MAX_IRRADIANCE;
	if (!isfinite(le))
		return 0.0;
	if (sample.game == 0u && le > cap)                      // (v86: the lamps' clamp; the game's lights' uniform pick had none)
		light *= cap / le;
	return max(light, 0.0);
}
// v86: the bounce hits' game lights come from the ReGIR cells this frame (rtxdi_lamps_at), not from the hits' own uniform pick
bool rtxdi_game_in_regir()
{
	return (g_rtxdi.flags & (RTXDI_FRAME_REGIR | RTXDI_FRAME_GAME)) == (RTXDI_FRAME_REGIR | RTXDI_FRAME_GAME) && rtxdi_game_count(false) != 0u;
}
