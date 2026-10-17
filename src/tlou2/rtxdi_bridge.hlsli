// TLOU RT - the RTXDI application bridge (v76; v77: included by rtxdi.hlsli): the RAB_ structures and functions NVIDIA's
// RTXDI library calls (its Doc\RtxdiApplicationBridge.md), on the add-on's data - the surfaces from the game adapter
// (adapter.hlsl: this frame's and last frame's), the lamps from the add-on's emitter lists (rt.hlsl: this frame's and last
// frame's, a lamp's index translated through its slot in the lamps' memory), visibility from the scene's rays
// (trace_visible), the lamps' shading as the add-on's own ReSTIR DI (di_eval: diffuse; v82: plus GGX specular,
// rtxdi_reflected). v86: the game's own lights too, after the lamps in RTXDI's list (cs_rtxdi_game's tables; their terms the
// game's - light_arrival, its shadow maps, the lighting's rules - and v71's DD2 response). This file is the add-on's own
// code; the library is compiled in unmodified.
// Positions are relative to this frame's camera (the add-on's convention; the rays add c[9]); last frame's surfaces are
// moved by the camera's step (FRAME_CAM_PREV - FRAME_CAM_CUR).

// RTXDI_NEIGHBOR_OFFSETS_BUFFER: the library reads it as a Buffer<float2> (RG8_SNORM); the int8 pairs ride in the frame's
// draw table (v77: where the frame's constants say - rtxdi.hlsli g_rtxdi.offsetsAt)
struct rtxdi_offsets_view
{
	float2 operator[](uint i)
	{
		const uint w = Table.Load(g_rtxdi.offsetsAt + (i >> 1) * 4u) >> ((i & 1u) * 16u);
		return max(float2(int(w << 24) >> 24, int(w << 16) >> 24) / 127.0, -1.0);
	}
};
static const rtxdi_offsets_view g_rtxdi_offsets;

// ---- surfaces ----
struct RAB_Material
{
	float3 diffuseAlbedo;   // albedo x (1 - metalness)
	float3 specularF0;
	float roughness;
};
struct RAB_Surface
{
	float3 worldPos;        // relative to this frame's camera
	float3 viewDir;         // toward the camera that saw it
	float viewDepth;        // 0: none
	float3 normal;          // the shading normal lifted onto the triangle's side and toward the viewer (visible_normal)
	float3 geoNormal;
	RAB_Material material;
	float diffuseProbability;   // v83: the chance of the paths' BSDF sample taking the diffuse lobe (ReSTIR PT, rtxdi_pt.hlsl);
	                            //   1 on ReSTIR DI's surfaces (its BRDF rays: cosine-weighted)
	// v86: what the game's lights read of a surface (rtxdi_game_reflected: the lighting's rules, its shadow maps, DD2's
	// response) and the lighting's factor there - the game's lights come in the light buffer's units before it, the lamps
	// after it (the light buffer's own); ReSTIR DI's surfaces only (a bounce hit: px.x RTXDI_HIT_PIXEL)
	uint2 px;
	uint stencil;
	uint character;
	float cavity;
	float lightOcclusion;
	float3 post;                // the lighting's factor x Brightness (a bounce hit: its path's pixel's)
	float3 base;                // the albedo before the metalness
	float metal;
	uint4 exact;                // v111: a camera pixel's exact lamps (rtxdi_exact_at's ids: 8 x 16 bits, 0 none) - none elsewhere
};
#define RTXDI_HIT_PIXEL 0xFFFFu
#include "regir_vis.hlsli"                                  // v93: ReGIR's visibility cache (the lights' weights, the rays' records)
RAB_Material RAB_EmptyMaterial()
{
	return (RAB_Material)0;
}
RAB_Surface RAB_EmptySurface()
{
	return (RAB_Surface)0;
}
bool RAB_IsSurfaceValid(RAB_Surface surface)
{
	return surface.viewDepth > 0.0;
}
float3 RAB_GetSurfaceWorldPos(RAB_Surface surface)
{
	return surface.worldPos;
}
float3 RAB_GetSurfaceNormal(RAB_Surface surface)
{
	return surface.normal;
}
float RAB_GetSurfaceLinearDepth(RAB_Surface surface)
{
	return surface.viewDepth;
}
RAB_Material RAB_GetMaterial(RAB_Surface surface)
{
	return surface.material;
}
// v83: what NVIDIA's ReSTIR PT reads and moves of a surface (its reconnection rules, its sky vertex)
float3 RAB_GetSurfaceGeoNormal(RAB_Surface surface)
{
	return surface.geoNormal;
}
float3 RAB_GetSurfaceViewDir(RAB_Surface surface)
{
	return surface.viewDir;
}
float RAB_GetSurfaceRoughness(RAB_Surface surface)
{
	return surface.material.roughness;
}
void RAB_SetSurfaceNormal(inout RAB_Surface surface, float3 normal)
{
	surface.normal = normal;
}
void RAB_SetSurfaceWorldPos(inout RAB_Surface surface, float3 worldPos)
{
	surface.worldPos = worldPos;
}
// Reuse between similar materials only: roughness within half of itself, reflectance and albedo (luminance) within 0.25
bool RAB_AreMaterialsSimilar(RAB_Material a, RAB_Material b)
{
	return RTXDI_CompareRelativeDifference(a.roughness, b.roughness, 0.5) &&
		abs(restir_lum(a.specularF0) - restir_lum(b.specularF0)) <= 0.25 &&
		abs(restir_lum(a.diffuseAlbedo) - restir_lum(b.diffuseAlbedo)) <= 0.25;
}
#ifdef TLOU_PT
#include "rtxdi_pt_frame.hlsli"                             // v83: ReSTIR PT's constants, reservoirs and primary surfaces
#endif
// v111: the RTXPT renderer's exact lamps per camera pixel - the lamps cs_rtxpt_path lights each with its own ray (v106), chosen
// before ReSTIR DI runs (rtxpt.hlsl cs_rtxpt_rank) so that its target skips them. Two planes by frame parity at the frame
// block's TABLE_EXACT_LAMPS (tlou_rt.cpp data_layout exact_lamps), RTXDI_EXACT_BYTES a pixel: a uint4 of the lamps' ids - their
// slot in the lamps' memory + 1 (RAB_LightInfo.lamp, the same lamp across frames), two 16-bit halves a word, 0 none - then a
// uint4 of their index in that frame's list + 1, the same way
#define TABLE_EXACT_LAMPS (TABLE_REGIR + 28)                // tlou_rt.cpp regir_frame exact_at
#define RTXDI_EXACT_BYTES 32u
uint rtxdi_exact_at(uint parity, uint2 px)
{
	const uint2 size = full_size();
	return Table.Load(TABLE_EXACT_LAMPS) + parity * align_up(size.x * size.y * RTXDI_EXACT_BYTES, 256u) + (px.y * size.x + px.x) * RTXDI_EXACT_BYTES;
}
bool rtxdi_exact_has(uint4 exact, uint lamp)                // lamp: RAB_LightInfo.lamp (0: not a lamp - never)
{
	return lamp != 0u && (any((exact & 0xFFFFu) == lamp) || any((exact >> 16) == lamp));
}
// The adapter's surface at a pixel, this frame's or last frame's; hair (this frame's: cs_motion's flags) faces the viewer
// with its own normal as the triangle's, as the RE9 renderer shades it. v83: in ReSTIR PT's passes the surfaces its paths
// started from (rtxdi_pt_frame.hlsli: cs_pt_initial's, the camera ray re-traced as cs_re9_path does)
RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame)
{
#ifdef TLOU_PT
	return pt_gbuffer_surface(pixelPosition, previousFrame);
#endif
	RAB_Surface surface = RAB_EmptySurface();
	const uint2 size = full_size();
	if (any(pixelPosition < 0) || pixelPosition.x >= int(size.x) || pixelPosition.y >= int(size.y))
		return surface;
	const surface_record r = load_surface_record((g_frame & 1u) ^ (previousFrame ? 1u : 0u), uint2(pixelPosition));
	if (!(r.view_depth > 0.0))
		return surface;
	const float3 eye = previousFrame ? asfloat(Data.Load3(FRAME_CAM_PREV)) - asfloat(Data.Load3(FRAME_CAM_CUR)) : 0.0;
	surface.worldPos = r.p + eye;
	surface.viewDir = normalize(eye - surface.worldPos);
	surface.viewDepth = r.view_depth;
	float3 n = r.n, ng = r.ng;
	if (!previousFrame && (pixel_flags(uint2(pixelPosition)) & PIXEL_HAIR) != 0u)
	{
		n = dot(n, surface.viewDir) < 0.0 ? -n : n;
		ng = n;
	}
	surface.geoNormal = ng;
	surface.normal = visible_normal(n, surface.viewDir, ng);
	surface.material.diffuseAlbedo = r.albedo * (1.0 - r.metallic);
	surface.material.specularF0 = lerp(r.spec * (0.08 / 255.0), r.albedo, r.metallic);
	surface.material.roughness = r.roughness;
	surface.diffuseProbability = 1.0;
	// v86: for the game's lights (the adapter's words: the stencil, the lights' occlusion; last frame's surface takes this
	// frame's lighting factor at its pixel)
	surface.px = uint2(pixelPosition);
	surface.stencil = r.game1;
	surface.character = (r.flags & SURFACE_CHARACTER) != 0u ? 1u : 0u;
	surface.cavity = r.ao;
	surface.lightOcclusion = f16tof32(r.game0);
	surface.post = max(lighting_factor(uint2(pixelPosition), Depth[uint2(pixelPosition)]) * g_brightness, 1e-4);
	surface.base = r.albedo;
	surface.metal = r.metallic;
	// v111: the exact lamps of the pixel in its frame (only while that frame's were made: RTXDI_FRAME_EXACT_LAMPS / _PREV)
	if ((g_rtxdi.flags & (previousFrame ? RTXDI_FRAME_EXACT_PREV : RTXDI_FRAME_EXACT_LAMPS)) != 0u)
		surface.exact = Data.Load4(rtxdi_exact_at((g_frame & 1u) ^ (previousFrame ? 1u : 0u), uint2(pixelPosition)));
	return surface;
}
// BRDF rays (RTXDI's BRDF samples; off: numBrdfSamples 0) - cosine-weighted, as the lamps' shading is diffuse
bool RAB_SurfaceImportanceSampleBrdf(RAB_Surface surface, inout RTXDI_RandomSamplerState rng, out float3 dir)
{
	const float2 u = float2(RTXDI_GetNextRandom(rng), RTXDI_GetNextRandom(rng));
	dir = cosine_direction(surface.normal, u);
	return dot(surface.geoNormal, dir) > 0.0;
}
// v83: GGX at a surface, as the RE9 renderer's specular lobe (re9.hlsl re9_ggx_d, smith_g1): the distribution, Smith's G1
float rtxdi_ggx_d(float ndh, float alpha)
{
	const float a2 = alpha * alpha;
	const float k = ndh * ndh * (a2 - 1.0) + 1.0;
	return min(a2 / (3.14159265 * k * k), 1e9);
}
float rtxdi_smith_g1(float nx, float alpha)
{
	const float a2 = alpha * alpha;
	return 2.0 * nx / max(nx + sqrt(a2 + (1.0 - a2) * nx * nx), 1e-6);
}
// The pdf of the paths' BSDF sample toward dir (v83: rtxdi_pt.hlsl pt_brdf_ray, the RE9 renderer's lobe mixture) - the
// diffuse lobe's cosine with probability diffuseProbability, GGX visible normals G1(v) D / (4 N.V) otherwise; ReSTIR DI's
// surfaces (diffuseProbability 1): the cosine alone, as v76-v82
float RAB_SurfaceEvaluateBrdfPdf(RAB_Surface surface, float3 dir)
{
	const float cosl = dot(surface.normal, dir);
	if (!(cosl > 0.0))
		return 0.0;
	float pdf = surface.diffuseProbability * cosl * 0.31830989;
	if (surface.diffuseProbability < 1.0)
	{
		const float ndv = max(dot(surface.normal, surface.viewDir), 1e-4);
		const float alpha = max(surface.material.roughness * surface.material.roughness, 0.002);
		const float3 h = normalize(surface.viewDir + dir);
		pdf += (1.0 - surface.diffuseProbability) * rtxdi_smith_g1(ndv, alpha) * rtxdi_ggx_d(saturate(dot(surface.normal, h)), alpha) / (4.0 * ndv);
	}
	return pdf;
}

// ---- lights: the lamps (the emitter lists; v76 two, this frame's and last frame's) ----
struct RAB_LightInfo
{
	float3 centre;          // absolute world (the list's)
	uint normal;            // packed: the side it shines toward (a cardioid around it)
	float3 Le;              // radiance
	float area;             // its falloff (emit_falloff) and how far short of it its shadow ray stops (emit_skip)
	uint lamp;              // its slot in the lamps' memory + 1 (0: none)
	uint game;              // v86: one of the game's lights - its index in the game's light records + 1 (0: a lamp); centre,
	                        //   Le and area then its position, colour and radius
};
RAB_LightInfo RAB_EmptyLightInfo()
{
	return (RAB_LightInfo)0;
}
uint rtxdi_lamp_count(bool previousFrame)                   // the lamps listed (v76-v85 rtxdi_light_count)
{
	const uint4 h = Data.Load4(previousFrame ? emit_list_prev_offset() : emit_list_offset());
	return h.z != 0u && g_emit > 0.0 ? min(h.x, EMIT_MAX) : 0u;
}
// v86 (user: "go for rtxdi for sky and game lights"): the game's lights in RTXDI's list after the lamps - cs_rtxdi_game's
// tables, this frame's and last frame's by frame parity (RTXDI_GAME_BYTES each): a header (count, the frame it is from), per
// slot the light's index in the game's records, its flags, radius and cookie (its identity: the census lists the lights in
// a new order every frame, the slots do not carry over), and the map from a record's index to its slot + 1
#define RTXDI_GAME_BYTES 8192u
#define RTXDI_GAME_SLOTS 16u                                // the slots' offset in a table (after the header)
#define RTXDI_GAME_MAP (16u + 128u * 16u)                   // the map's (1024 records)
uint rtxdi_game_at(bool previousFrame)
{
	return g_rtxdi.gameAt + ((g_frame & 1u) ^ (previousFrame ? 1u : 0u)) * RTXDI_GAME_BYTES;
}
uint rtxdi_game_count(bool previousFrame)
{
	if ((g_rtxdi.flags & RTXDI_FRAME_GAME) == 0u)
		return 0u;
	const uint2 h = Data.Load2(rtxdi_game_at(previousFrame));
	return h.y == (previousFrame ? g_frame - 1u : g_frame) ? min(h.x, LIGHT_LIST_MAX) : 0u;
}
// The lamps in RTXDI's list: as many as fit before the game's lights in the lamps' power PDF (EMIT_MAX texels)
uint rtxdi_lamps_used(bool previousFrame)
{
	return min(rtxdi_lamp_count(previousFrame), EMIT_MAX - rtxdi_game_count(previousFrame));
}
uint rtxdi_light_count(bool previousFrame)                  // v86: the lamps and the game's lights
{
	return rtxdi_lamps_used(previousFrame) + rtxdi_game_count(previousFrame);
}
// The game's light in slot g of a frame's table: this frame's record (the game keeps no last frame's) if it is the same light
// (flags, radius, cookie); none otherwise
RAB_LightInfo rtxdi_game_light_info(uint g, bool previousFrame)
{
	RAB_LightInfo light = RAB_EmptyLightInfo();
	if (g >= rtxdi_game_count(previousFrame))
		return light;
	const uint4 e = Data.Load4(rtxdi_game_at(previousFrame) + RTXDI_GAME_SLOTS + g * 16u);
	const LightDesc L = Lights[e.x];
	if (L.flags != e.y || asuint(L.radius) != e.z || L.cookie != e.w || !all(isfinite(L.position)) || !all(isfinite(L.color)))
		return light;
	light.centre = L.position;
	light.Le = L.color;
	light.area = L.radius;
	light.game = e.x + 1u;
	return light;
}
RAB_LightInfo RAB_LoadLightInfo(uint index, bool previousFrame)
{
	RAB_LightInfo light = RAB_EmptyLightInfo();
	const uint lamps = rtxdi_lamps_used(previousFrame);
	if (index >= lamps)
		return rtxdi_game_light_info(index - lamps, previousFrame);   // v86
	const uint base = previousFrame ? emit_list_prev_offset() : emit_list_offset();
	if (index >= min(Data.Load(base), EMIT_MAX))
		return light;
	const uint at = base + 16u + index * EMIT_BYTES;
	const uint4 a = Data.Load4(at), b = Data.Load4(at + 16);
	light.centre = asfloat(a.xyz);
	light.normal = a.w;
	light.Le = float3(f16tof32(b.x), f16tof32(b.x >> 16), f16tof32(b.y));
	light.area = asfloat(b.z);
	light.lamp = b.w;
	if (!all(isfinite(light.centre)) || !all(isfinite(light.Le)) || !(light.area >= 0.0 && light.area < 1e6))
		light = RAB_EmptyLightInfo();
	return light;
}
RAB_LightInfo RAB_LoadCompactLightInfo(uint linearIndex)    // presampling's (off)
{
	return RAB_EmptyLightInfo();
}
bool RAB_StoreCompactLightInfo(uint linearIndex, RAB_LightInfo lightInfo)
{
	return false;
}
// v88: the smallest and largest cosine between an axis and the directions toward a sphere whose centre lies at cosine c from
// it and whose angular radius is asin(sr) (sr: the sphere's radius over its distance; 1 and over: the sphere holds the apex)
float2 rtxdi_cos_range(float c, float sr)
{
	if (sr >= 1.0)
		return float2(-1.0, 1.0);
	const float t = acos(clamp(c, -1.0, 1.0)), a = asin(max(sr, 0.0));
	return float2(cos(min(t + a, 3.14159265)), cos(max(t - a, 0.0)));
}
// v77: ReGIR's weight of a lamp for a cell (its centre and radius relative to this frame's camera): the lamp's brightness x its
// area's falloff at the cell's near side x its cardioid toward the cell, widened by the cell's angular size (no lamp that
// lights part of the cell gets 0). v88 (user: the ReGIR article, "go 88v"): its light-orientation terms - each the largest the
// light gives anywhere in the cell, so a light is 0 only where it lights no point of the cell (no bias): the lamp's cardioid
// toward the cell's most favourable point (v77-v87 kept at least half of it: a lamp facing away from the whole cell weighed
// 0.5), a spot light's cone toward the cell (v86-v87: none - a spot weighed as much behind it as in its beam)
float RAB_GetLightTargetPdfForVolume(RAB_LightInfo light, float3 volumeCenter, float volumeRadius)
{
	if (light.game != 0u)
	{
		// v86: a game's light - its colour x its falloff at the cell's near side (light_arrival's; its cookies and frustum left
		// out: never 0 where part of the cell is lit) x Brightness (the lamps' light comes after the lighting's factor, the
		// game's lights' before it). v88: a spot without a cookie x its cone(s) at the cell's most favourable direction
		// v99: with Physical light falloff (the RTXPT renderer) the sphere light's falloff there (rt.hlsl light_arrival_ex)
		const LightDesc L = Lights[light.game - 1u];
		const float3 to_light = (light.centre - c[9].xyz) - volumeCenter;
		const float dist = length(to_light), len = max(dist - volumeRadius, 0.0);
		const float strength = (g_rtxdi.flags & RTXDI_FRAME_PHYSICAL) != 0u && (L.flags & 0x4Fu) != 0x41u ?
			light_intensity(L) * sphere_light_falloff(len, max(L.radius, g_light_radius)) : light_curve(L, len, len * len);
		float shape = 1.0;
		if ((L.flags & 15u) == 2u && ((L.flags >> 9) & 7u) == 0u && dist > 1e-4)
		{
			const float2 cl = rtxdi_cos_range(dot(to_light / dist, L.axis), volumeRadius / dist);   // light_arrival's dot(l, axis)
			shape = saturate(max(cl.x * L.cone.y, cl.y * L.cone.y) + L.cone.x);
			if ((L.flags & 0x20000000u) != 0u)
				shape *= saturate(max(cl.x * L.cone2.y, cl.y * L.cone2.y) + L.cone2.x);
		}
		const float w = restir_lum(light.Le) * strength * shape * g_brightness;
		return isfinite(w) ? max(w, 0.0) : 0.0;
	}
	if (!(light.area > 0.0))
		return 0.0;
	const float3 d = volumeCenter - (light.centre - c[9].xyz);
	const float len = length(d);
	const float facing = len > 1e-4 ? rtxdi_cos_range(dot(unpack_normal(light.normal), d / len), volumeRadius / len).y : 1.0;
	// (v86: x Lamps - with the game's lights in the cells the two must weigh against each other as they light)
	const float w = restir_lum(light.Le) * g_emit * (0.5 + 0.5 * facing) * emit_falloff(light.area, max(len - volumeRadius, 0.0));
	return isfinite(w) ? max(w, 0.0) : 0.0;
}
// A lamp's index in the other frame's list: through its slot in the lamps' memory and the two frames' maps (cs_lamp_merge);
// -1 when it is not listed there or its slot took another lamp this frame
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
	const uint lamps = rtxdi_lamps_used(!currentToPrevious);
	if (lightIndex >= lamps)
	{
		// v86: a game's light - its record's index in the other frame's map, the slot there holding the same light (index,
		// flags, radius, cookie), after that frame's lamps
		const uint g = lightIndex - lamps;
		if (g >= rtxdi_game_count(!currentToPrevious) || rtxdi_game_count(currentToPrevious) == 0u)
			return -1;
		const uint4 e = Data.Load4(rtxdi_game_at(!currentToPrevious) + RTXDI_GAME_SLOTS + g * 16u);
		const uint other = rtxdi_game_at(currentToPrevious);
		const uint slot = Data.Load(other + RTXDI_GAME_MAP + min(e.x, 1023u) * 4u);
		if (slot == 0u || slot > rtxdi_game_count(currentToPrevious) || any(Data.Load4(other + RTXDI_GAME_SLOTS + (slot - 1u) * 16u) != e))
			return -1;
		return int(rtxdi_lamps_used(currentToPrevious) + slot - 1u);
	}
	const RAB_LightInfo light = RAB_LoadLightInfo(lightIndex, !currentToPrevious);
	if (light.lamp == 0u)
		return -1;
	const uint now = Data.Load(lamp_map_offset(g_frame & 1u) + (light.lamp - 1u) * 4u);
	if (now == 0u || (now & LAMP_MAP_NEW) != 0u)
		return -1;
	uint mapped;
	if (!currentToPrevious)
		mapped = now - 1u;
	else
	{
		const uint before = Data.Load(lamp_map_offset((g_frame & 1u) ^ 1u) + (light.lamp - 1u) * 4u);
		if (before == 0u)
			return -1;
		mapped = (before & ~LAMP_MAP_NEW) - 1u;
	}
	return mapped < rtxdi_lamps_used(currentToPrevious) ? int(mapped) : -1;   // v86: a lamp past the list's lamps is not in it
}
// The pdf of drawing a lamp uniformly from this frame's list (the BRDF rays' MIS: off); no environment light
float RAB_EvaluateLocalLightSourcePdf(uint lightIndex)
{
	const uint count = rtxdi_light_count(false);
	return count != 0u ? 1.0 / float(count) : 0.0;
}
float RAB_EvaluateEnvironmentMapSamplingPdf(float3 L)
{
	return 0.0;
}
float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir)
{
	return 0.0;
}

// ---- light samples: a lamp is a point at its centre with its area's falloff - analytic for RTXDI ----
struct RAB_LightSample
{
	float3 position;        // the lamp's centre, relative to this frame's camera
	float3 normal;
	float3 radiance;        // its radiance x the Lamps slider
	float area;
	float solidAnglePdf;    // 1 (analytic: no BRDF ray finds a lamp)
	uint game;              // v86: a game's light (RAB_LightInfo.game; its terms are the game's own: rtxdi_game_reflected)
	float2 uv;              //   and the point of its sphere its shadow ray aims at (light_visibility's xi)
	uint lamp;              // v111: a lamp's slot in the lamps' memory + 1 (RAB_LightInfo.lamp; 0: not a lamp)
};
RAB_LightSample RAB_EmptyLightSample()
{
	return (RAB_LightSample)0;
}
bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
	return true;
}
float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
	return lightSample.solidAnglePdf;
}
float3 RAB_LightSamplePosition(RAB_LightSample lightSample)  // v83 (ReSTIR PT's NEE reconnection: unused, its paths take no NEE)
{
	return lightSample.position;
}
float3 RAB_LightSampleRadiance(RAB_LightSample lightSample)
{
	return lightSample.radiance;
}
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
	RAB_LightSample sample = RAB_EmptyLightSample();
	if (lightInfo.game != 0u)
	{
		// v86: a game's light - its centre (the shading's direction, as the lighting's L), the sample's numbers for the point of
		// its sphere the shadow ray aims at
		sample.position = lightInfo.centre - c[9].xyz;
		sample.radiance = lightInfo.Le;
		sample.area = lightInfo.area;
		sample.solidAnglePdf = 1.0;
		sample.game = lightInfo.game;
		sample.uv = uv;
		return sample;
	}
	if (!(lightInfo.area > 0.0))
		return sample;
	sample.position = lightInfo.centre - c[9].xyz;
	sample.normal = unpack_normal(lightInfo.normal);
	sample.radiance = lightInfo.Le * g_emit;
	sample.area = lightInfo.area;
	sample.solidAnglePdf = 1.0;
	sample.lamp = lightInfo.lamp;                           // v111
	return sample;
}
// The lamp's light the surface reflects, unshadowed, as di_eval: diffuse albedo / pi x radiance x the cardioid around the
// lamp's normal x N.L x its area's falloff (the light buffer's units after the lighting's factor). v82 (user: "most light
// is not being interacted with PBR material"): plus the specular part - GGX with the surface's roughness and F0 (the game
// adapter's material, the auto-PBR layer's where a game has none), Smith's height-correlated visibility, Schlick's Fresnel,
// the lobe widened by the lamp's angular radius and renormalised (a sphere light, Karis 2013): lamps light highlights on wet,
// glossy and metal surfaces (v76-v81: diffuse only), and RTXDI's target picks lamps by both parts. v86: the two parts apart
// (cs_rtxdi_di keeps the specular apart: the skin's blur and RR take the diffuse light alone)
void rtxdi_lamp_reflected(RAB_LightSample sample, RAB_Surface surface, out float3 diffuse, out float3 specular)
{
	diffuse = 0.0;
	specular = 0.0;
	const float3 d = sample.position - surface.worldPos;
	const float len = length(d);
	if (!(len > 0.05) || !(sample.area > 0.0))
		return;
	const float3 l = d / len;
	const float cs = dot(surface.normal, l);
	if (cs <= 0.0)
		return;
	const float ce = 0.5 + 0.5 * dot(sample.normal, -l);
	const float3 incoming = (ce * cs * emit_falloff(sample.area, len)) * sample.radiance;   // the lamp's light on the surface (x N.L)
	diffuse = surface.material.diffuseAlbedo * 0.31830989 * incoming;
	const float3 v = surface.viewDir;
	const float cv = dot(surface.normal, v);
	if (cv > 0.0 && any(surface.material.specularF0 > 0.0))
	{
		const float3 h = normalize(l + v);
		const float ch = saturate(dot(surface.normal, h)), vh = saturate(dot(v, h));
		const float a = max(surface.material.roughness * surface.material.roughness, 0.002);
		const float aw = saturate(a + sqrt(sample.area * 0.31830989) / (2.0 * len));   // + the lamp's angular radius / 2
		const float a2 = aw * aw;
		const float dd = ch * ch * (a2 - 1.0) + 1.0;
		const float D = a2 / (3.14159265 * dd * dd) * (a * a) / a2;   // GGX, widened and renormalised
		const float ar = a * a;
		const float V = 0.5 / max(cs * sqrt(cv * cv * (1.0 - ar) + ar) + cv * sqrt(cs * cs * (1.0 - ar) + ar), 1e-6);   // G / (4 N.L N.V)
		const float3 F = surface.material.specularF0 + (1.0 - surface.material.specularF0) * pow(1.0 - vh, 5.0);
		specular = min(D * V, 1000.0) * F * incoming;
	}
}

// v86: the game's lights through RTXDI - re9.hlsl's DD2 light rules and response (defined there, after this bridge)
bool dd2_light_takes(LightDesc L, surface s);
float3 dd2_light_dir(LightDesc L, float3 l);
void dd2_response(float3 n, float3 v, float3 l, float3 base, float metal, float3 f0, float a, float ecomp, out float3 diffuse, out float3 specular);
float dd2_ecomp(float roughness, float nov);
// A surface as the game's light functions read it (rt.hlsl's: the shadow maps' terms); a bounce hit as hit_surface makes it
surface rtxdi_game_surface(RAB_Surface s)
{
	surface g = hit_surface(s.worldPos, s.normal);
	g.ng = s.geoNormal;
	if (s.px.x == RTXDI_HIT_PIXEL)
		return g;
	g.v = s.viewDir;
	g.albedo = s.base;
	g.roughness = s.material.roughness;
	g.metallic = s.metal;
	g.stencil = s.stencil;
	g.character = s.character != 0u;
	g.px = s.px;
	g.view_depth = s.viewDepth;
	g.cavity = s.cavity;
	g.light_occlusion = s.lightOcclusion;
	return g;
}
// One of the game's lights at a surface as v71-v85's per-light loops took it (re9.hlsl cs_re9_path, dd2.hlsl cs_dd2_path):
// the lighting's rules for the pixel (stencil, characters; at a bounce hit the lights every surface takes, as its uniform
// pick took them), the light arriving there (light_arrival: falloff, cones, cookies, frustum), DD2's response with the
// light's roughness floor, its diffuse and specular switches and its shadow map (the lighting's filter at a pixel, one tap at
// a bounce hit) - not the rays. In the lamps' units: x the lighting's factor at the surface, which the passes divide by again
void rtxdi_game_reflected(RAB_LightSample sample, RAB_Surface s, out float3 diffuse, out float3 specular)
{
	diffuse = 0.0;
	specular = 0.0;
	const LightDesc L = Lights[sample.game - 1u];
	const bool hit = s.px.x == RTXDI_HIT_PIXEL;
	const surface gs = rtxdi_game_surface(s);
	if (hit ? ((L.flags & 0x80u) == 0u || (L.flags & 0x404E020u) != 0u) : !dd2_light_takes(L, gs))
		return;
	float3 color, l;
	float dist;
	if (!light_arrival_ex(L, s.worldPos, (g_rtxdi.flags & RTXDI_FRAME_PHYSICAL) != 0u, color, l, dist))   // (v99: physical)
		return;
	const float alpha = max(s.material.roughness * s.material.roughness, 1e-4);
	float3 dl, sl;
	dd2_response(s.normal, s.viewDir, dd2_light_dir(L, l), s.base, s.metal, s.material.specularF0, max(alpha, L.min_roughness * L.min_roughness),
		dd2_ecomp(s.material.roughness, dot(s.normal, s.viewDir)), dl, sl);
	if ((L.flags & 128u) == 0u)
		dl = 0.0;
	if ((L.flags & 256u) == 0u)
		sl = 0.0;
	if (!any(dl + sl > 0.0))
		return;
	const float sh = hit ? (L.shadow_map != -1 ? game_shadow_map(L, gs, false) : 1.0) : game_light_shadow(L, gs);
	if (!(sh > 0.0))
		return;
	diffuse = color * (sh * dl) * s.post;
	specular = color * (sh * sl) * s.post;
}
void rtxdi_reflected_split(RAB_LightSample sample, RAB_Surface surface, out float3 diffuse, out float3 specular)
{
	if (sample.game != 0u)
		rtxdi_game_reflected(sample, surface, diffuse, specular);
	else
		rtxdi_lamp_reflected(sample, surface, diffuse, specular);
}
float3 rtxdi_reflected(RAB_LightSample sample, RAB_Surface surface)
{
	float3 diffuse, specular;
	rtxdi_reflected_split(sample, surface, diffuse, specular);
	return diffuse + specular;
}
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
	// v104: a game's light at a camera's surface while the RTXPT renderer lights those with every game light itself
	// (RTXDI_FRAME_EXACT_GAME): none - ReSTIR DI's picks there are the lamps' (the estimate stays the lamps' own light)
	if (lightSample.game != 0u && (g_rtxdi.flags & RTXDI_FRAME_EXACT_GAME) != 0u && surface.px.x != RTXDI_HIT_PIXEL)
		return 0.0;
	// v111 (user: v106's "shadows are right ... the ONLY issue is small noise"; its lamp view rt-recon\inputs-20261015-133736:
	// the walls in two levels - ReSTIR DI's pick was one of the lamps cs_rtxpt_path lights exactly about 30 % of the time, its
	// light then counted for nothing, else about the lamps' whole light): one of the camera pixel's exact lamps - none, so
	// ReSTIR DI picks among the rest of the lamps alone (their estimate, the exact ones' own rays: the same light, v106's way)
	if ((g_rtxdi.flags & RTXDI_FRAME_EXACT_LAMPS) != 0u && rtxdi_exact_has(surface.exact, lightSample.lamp))
		return 0.0;
	float p = restir_lum(rtxdi_reflected(lightSample, surface));
	if (p > 0.0)
		p *= regir_vis_chance(surface.worldPos, surface.geoNormal, lightSample.position);   // v93: the chance it is seen (>= floor)
	return isfinite(p) ? max(p, 0.0) : 0.0;
}
void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample, out float3 o_lightDir, out float o_lightDistance)
{
	const float3 d = lightSample.position - surface.worldPos;
	o_lightDistance = length(d);
	o_lightDir = d / max(o_lightDistance, 1e-6);
}
// The lamps are not the scene's triangles: a BRDF ray finds none (BRDF samples are off)
bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax, out uint o_lightIndex, out float2 o_randXY)
{
	o_lightIndex = RTXDI_InvalidLightIndex;
	o_randXY = 0.0;
	return false;
}

// ---- visibility: the scene's rays, from off the surface as di_shade's ----
float3 rtxdi_origin(RAB_Surface surface)
{
	return surface.worldPos + surface.geoNormal * (0.01 + length(surface.worldPos) * 0.002);
}
bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
	if (lightSample.game != 0u)
	{
		// v86: a game's light - light_visibility's ray toward a point of its sphere (at least the Light size slider's radius),
		// short of the fixture and of its shadow map's near plane; none with Light shadows 0 (a value between counts as 1:
		// RTXDI's visibility is all or nothing) nor on characters for a light with a shadow map (they keep the game's, v71's rule)
		const LightDesc L = Lights[lightSample.game - 1u];
		if (g_light_shadows <= 0.0 || (L.shadow_map != -1 && surface.character != 0u))
			return true;
		const bool seen = light_visibility(rtxdi_game_surface(surface), rtxdi_origin(surface), L, lightSample.uv) > 0.0;
		regir_vis_record(surface.worldPos, surface.geoNormal, lightSample.position, seen);   // v93: the visibility cache
		return seen;
	}
	const float3 o = rtxdi_origin(surface);
	const float3 d = lightSample.position - o;
	const float len = length(d);
	const float tmax = len - emit_skip(lightSample.area, len);
	if (!(tmax > 0.0))
		return true;
	const bool seen = trace_visible(o, d / len, tmax) > 0.0;
	regir_vis_record(surface.worldPos, surface.geoNormal, lightSample.position, seen);       // v93
	return seen;
}
// Last frame's surface: its ray through this frame's scene (the add-on keeps no previous BVH)
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, RAB_LightSample lightSample)
{
	return RAB_GetConservativeVisibility(previousSurface, lightSample);
}
bool RAB_GetConservativeVisibility(RAB_Surface surface, float3 samplePosition)
{
	const float3 o = rtxdi_origin(surface);
	const float3 d = samplePosition - o;
	const float len = length(d);
	if (!(len > 0.01))
		return true;
	return trace_visible(o, d / len, len * 0.999) > 0.0;
}
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, float3 samplePosition)
{
	return RAB_GetConservativeVisibility(previousSurface, samplePosition);
}

// ---- screen ----
int2 RAB_ClampSamplePositionIntoView(int2 pixelPosition, bool previousFrame)
{
	return clamp(pixelPosition, int2(0, 0), int2(full_size()) - 1);
}
