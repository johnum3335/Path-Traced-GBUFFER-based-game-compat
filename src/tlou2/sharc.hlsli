// TLOU RT - NVIDIA's SHaRC, the spatially hashed radiance cache (v87; user: "check this ... sharc from NVIDIA, i want an
// official integration with its files, replacing current one if necessary, check AGENTS.md, NVIDIA has laid out a step by
// step"). NVIDIA's SHARC SDK 1.8.3 (RTGI_builds\deps\sharc\include - SharcCommon.h, HashGridCommon.h, SharcTypes.h,
// HashGridTypes.h, unmodified; its AGENTS.md and docs\Integration.md beside them) is compiled in - NVIDIA's license, object
// code only: "This software contains source code provided by NVIDIA Corporation." Its three passes each frame, before the
// renderers (tlou_rt.cpp post_lighting):
//   SHaRC Update  - sharc.hlsl cs_sharc_update (SHARC_UPDATE 1, shader model 6.6: 64-bit atomics): sparse paths from the
//                   camera's surfaces (one pixel of each 5 x 5 tile, another every frame), their light submitted to the
//                   cache at every hit (SharcUpdateHit / SharcUpdateMiss / SharcSetThroughput)
//   SHaRC Resolve - sharc.hlsl cs_sharc_resolve_nv: SharcResolveEntry for every entry (the frames' accumulation, stale
//                   entries evicted)
//   SHaRC Query   - in the renderers' bounces (re9.hlsl re9_dd2_bounce, dd2.hlsl cs_dd2_path, rtxdi_pt.hlsl's paths): the
//                   light leaving a bounce hit the camera does not see from the cache (SharcGetCachedRadiance, AGENTS.md's
//                   segment length and footprint gates), the path ending there
// This file: the cache's buffers (root UAVs u5-u7 in space 100, regions of the data buffer), its frame constants (the draw
// table, TABLE_SHARC: tlou_rt.cpp prepare_sharc) and the shared helpers. Included by re9.hlsl: every renderer pass queries
// (SHARC_UPDATE 0 unless the including file set it). v23-v86's own SHaRC (rt.hlsl sharc_*) stays as the fallback ("NVIDIA
// SHaRC" off, or its passes missing).

#ifndef SHARC_UPDATE
#define SHARC_UPDATE 0
#endif
#ifndef SHARC_QUERY
#define SHARC_QUERY (SHARC_UPDATE == 0 ? 1 : 0)
#endif
#include "SharcCommon.h"

// The cache: hash entries (64-bit keys), the frame's accumulation and the resolved light - the same number of entries each,
// zero-initialized (the data buffer is created zeroed; cs_sharc_clear on a reset)
RWStructuredBuffer<HashGridKey> SharcHashEntries : register(u5, space100);
RWStructuredBuffer<SharcAccumulationData> SharcAccumulation : register(u6, space100);
RWStructuredBuffer<SharcPackedData> SharcResolved : register(u7, space100);

// The frame's SHaRC constants in the draw table (tlou_rt.cpp sharc_frame at k_table_sharc)
#define TABLE_SHARC 805376
#define SHARC_FLAG_ON 1u            // "NVIDIA SHaRC": its passes ran this frame - the queries take its cache
#define SHARC_FLAG_COUNT 2u         // the status line's counters
struct sharc_frame
{
	float sceneScale;               // HashGridParameters::sceneScale (the voxels' size)
	float radianceScale;            // SharcParameters::radianceScale (the accumulation's fixed point)
	uint capacity;                  // entries
	uint accumulationFrames;        // SharcResolveParameters::accumulationFrameNum
	uint staleFrames;               // ... staleFrameNumMax
	uint flags;                     // SHARC_FLAG_*
	uint downscale;                 // the Update pass's tiles (one path per downscale x downscale pixels)
#if SHARC_UPDATE
	float roughnessMin;             // the Update pass's roughness floor (AGENTS.md 14.4: no local highlights in the cache) -
#endif                              //   in the Update permutation only (AGENTS.md 21: the queries never see it)
	uint bounces;                   // the Update pass's path length
	uint pad0;
	uint pad1;
	uint pad2;
};
sharc_frame sharc_load_frame()
{
	const uint4 a = Table.Load4(TABLE_SHARC), b = Table.Load4(TABLE_SHARC + 16), e = Table.Load4(TABLE_SHARC + 32);
	sharc_frame f;
	f.sceneScale = asfloat(a.x);
	f.radianceScale = asfloat(a.y);
	f.capacity = a.z;
	f.accumulationFrames = a.w;
	f.staleFrames = b.x;
	f.flags = b.y;
	f.downscale = max(b.z, 1u);
#if SHARC_UPDATE
	f.roughnessMin = asfloat(b.w);
#endif
	f.bounces = e.x;
	f.pad0 = e.y;
	f.pad1 = e.z;
	f.pad2 = e.w;
	return f;
}
bool sharc_nv_on()
{
	return (Table.Load(TABLE_SHARC + 20) & SHARC_FLAG_ON) != 0u;
}

// SharcParameters as NVIDIA's sample builds them (AGENTS.md 12), the field names of the local SDK (1.8.3): the camera in
// absolute world space (the grid's levels grow with the distance to it), the hash grid's base and bias, the buffers
SharcParameters sharc_parameters()
{
	const sharc_frame f = sharc_load_frame();
	SharcParameters p;
	p.hashGridParameters.cameraPosition = c[9].xyz;
	p.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
	p.hashGridParameters.sceneScale = f.sceneScale;
	p.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
	p.hashGridData.capacity = f.capacity;
	p.hashGridData.hashEntriesBuffer = SharcHashEntries;
	p.radianceScale = f.radianceScale;
	p.accumulationBuffer = SharcAccumulation;
	p.resolvedBuffer = SharcResolved;
	return p;
}
// A hit as SHaRC takes it: its absolute world position and its geometric normal (toward the ray that found it)
SharcHitData sharc_hit(float3 p_rel, float3 n)
{
	SharcHitData h;
	h.positionWorld = p_rel + c[9].xyz;
	h.normalWorld = n;
	return h;
}

// The status line's counters (rt.hlsl STATS: lookups, found)
void sharc_count(uint stat, bool add)
{
	const uint n = WaveActiveCountBits(add);
	if (WaveIsFirstLane() && n != 0u && (Table.Load(TABLE_SHARC + 20) & SHARC_FLAG_COUNT) != 0u)
		Data.InterlockedAdd(STATS + stat * 4, n);
}

// SHaRC Query (AGENTS.md 14.8): the light leaving a bounce hit (p_rel camera-relative, n its normal toward the ray) from the
// cache - only where the segment just traced is longer than the hit's voxel (its diagonal) and the lobe that launched it
// spreads wider than the voxel there (lobe_roughness: the specular lobe's roughness, 1 for the diffuse lobe)
bool sharc_query(float3 p_rel, float3 n, float segment, float lobe_roughness, out float3 radiance)
{
	radiance = 0.0;
	if (!sharc_nv_on())
		return false;
	const SharcParameters p = sharc_parameters();
	const SharcHitData h = sharc_hit(p_rel, n);
	const uint level = HashGridGetLevel(h.positionWorld, p.hashGridParameters);
	const float voxel = HashGridGetVoxelSize(level, p.hashGridParameters);
	const float r = min(lobe_roughness, 0.99);
	const float alpha2 = r * r * r * r;
	const float footprint = segment * sqrt(0.5 * alpha2 / max(1.0 - alpha2, 1e-6));
	const bool eligible = segment > voxel * 1.7320508 && footprint > voxel;
	const bool found = eligible && SharcGetCachedRadiance(p, h, radiance, false);
	sharc_count(STAT_SHARC_QUERIES, eligible);
	sharc_count(STAT_SHARC_FOUND, found);
	if (!found || !all(isfinite(radiance)))
	{
		radiance = 0.0;
		return false;
	}
	radiance = max(radiance, 0.0);
	return true;
}
// The cached light at a point without the gates (ReSTIR PT's refresh of a kept path's light at its vertex)
bool sharc_lookup(float3 p_rel, float3 n, out float3 radiance)
{
	radiance = 0.0;
	if (!sharc_nv_on())
		return false;
	if (!SharcGetCachedRadiance(sharc_parameters(), sharc_hit(p_rel, n), radiance, false) || !all(isfinite(radiance)))
	{
		radiance = 0.0;
		return false;
	}
	radiance = max(radiance, 0.0);
	return true;
}

// AGENTS.md 17: the debug views (Debug view 8, 9; both composites) - the cache's light at the camera's surfaces (looked up
// at the depth buffer's surface: the light leaving it as the cache holds it; v23-v86's cache with NVIDIA SHaRC off) and
// the hash grid's voxels (HashGridDebugColoredHash)
float3 sharc_debug_light(uint2 px, float d)
{
	const float3 p = camera_relative_position(float2(px), d);
	const float3 ng = geometric_normal(px, p, decode_normal(GBuffer0[px]));
	float3 light;
	if (sharc_nv_on() ? sharc_lookup(p, ng, light) : sharc_get(p + c[9].xyz, ng, light))
		return light;
	return 0.0;
}
float3 sharc_debug_voxels(uint2 px, float d)
{
	const float3 p = camera_relative_position(float2(px), d);
	const float3 ng = geometric_normal(px, p, decode_normal(GBuffer0[px]));
	return HashGridDebugColoredHash(p + c[9].xyz, ng, sharc_parameters().hashGridParameters);
}
