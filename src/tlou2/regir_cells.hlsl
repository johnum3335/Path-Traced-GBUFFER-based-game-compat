// TLOU RT - ReGIR's surface cells' passes (v90; regir_cells.hlsli is the bridge, NVIDIA's SHARC SDK hash grid compiled in
// unmodified - "This software contains source code provided by NVIDIA Corporation."), each frame after RTXDI's presampling
// (the fill draws from its power-proportional tiles), before the renderers (tlou_rt.cpp post_lighting): cs_regir_cells
// (shader model 6.6: the hash grid's 64-bit atomic insertion), a UAV barrier, cs_regir_fill, a UAV barrier
#include "re9.hlsl"

// A cell for a surface point (camera-relative p, its normal n toward the ray that found it): inserted (or found), its cone
// set (the first normal) or widened to hold n, marked used this frame
void regir_insert(regir_frame f, float3 p, float3 n)
{
	HashGridKey key;
	HashGridIndex index;
	const bool inserted = RegirGridInsertEntry(regir_grid_data(f), p + c[9].xyz, n, regir_grid_parameters(f), key, index) && index < f.capacity;
	regir_count(FRAME_REGIR_FULL, !inserted);
	if (!inserted)
		return;
	const uint at = f.cellsAt + index * REGIR_CELL_BYTES;
	uint prev;
	Data.InterlockedCompareExchange(at, 0u, max(pack_normal(n), 1u), prev);
	const float3 axis = prev == 0u ? n : unpack_normal(prev);
	Data.InterlockedMax(at + 4u, regir_cone_width(dot(n, axis)));
	Data.Store(at + 8u, g_frame);
}

// The cells where bounce hits land: one pixel of each 4 x 4 tile (another every frame), a cosine ray from its surface, the
// hit's cell (the RE9 renderer's bounce takes its light from there)
[numthreads(8, 8, 1)]
void cs_regir_cells(uint3 id : SV_DispatchThreadID)
{
	const regir_frame f = regir_load_frame();
	const uint2 size = full_size();
	const uint2 tiles = (size + 3u) / 4u;
	if (id.x >= tiles.x || id.y >= tiles.y)
		return;
	uint rng = pcg(id.x + pcg(id.y + pcg(g_frame * 0x6C8E9CF5u + 0x7F4Au)));
	const uint2 px = min(id.xy * 4u + uint2(rng & 3u, (rng >> 2) & 3u), size - 1u);
	rng = pcg(rng);
	const float d = Depth[px];
	if (d >= 1.0 || (pixel_flags(px) & PIXEL_HAIR) != 0u)
		return;
	const surface s = load_surface(px, d);
	const float3 ng = geometric_normal(px, s.p, s.n);
	const float3 n = visible_normal(s.n, s.v, ng);
	const float3 l = cosine_direction(n, float2(dd2_next(rng), dd2_next(rng)));
	if (!(dot(l, ng) > 0.0))
		return;
	const float3 o = s.p + ng * (0.005 + length(s.p) * 0.001);
	const scene_hit hh = trace_cutouts(o, l, DD2_TMAX);
	if (!hh.hit)
		return;
	float3 nh = hit_normal(hh.draw, hh.primitive);
	if (dot(nh, l) > 0.0)
		nh = -nh;                                           // toward the ray (as the bounce hits' normals)
	regir_insert(f, o + l * hh.t, nh);
}

// A cell's slot: RIS over `candidates` of RTXDI's power-proportional picks with the cell's weight (v88's light terms over the
// voxel x the receiver's cosine over the cell's cone, each the largest anywhere in the cell), stored as RTXDI's slots (the
// light, the RIS weight sum over the pick's weight: 1 / pdf). A cell unused for `stale` frames is removed
groupshared uint g_regir_live;
[numthreads(256, 1, 1)]
void cs_regir_fill(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
	const regir_frame f = regir_load_frame();
	if (gi == 0u)
		g_regir_live = 0u;
	GroupMemoryBarrierWithGroupSync();
	const uint cell = id.x / REGIR_CELL_SLOTS, slot = id.x % REGIR_CELL_SLOTS;
	if (cell < f.capacity)
	{
		const HashGridKey key = RegirKeys[cell];
		if (key != HASH_GRID_INVALID_HASH_KEY)
		{
			const uint at = f.cellsAt + cell * REGIR_CELL_BYTES;
			const uint4 h = Data.Load4(at);
			if (g_frame - h.z > f.stale || h.x == 0u)
			{
				if (slot == 0u)
				{
					RegirKeys[cell] = HASH_GRID_INVALID_HASH_KEY;
					Data.Store4(at, uint4(0u, 0u, 0u, 0u));
				}
			}
			else
			{
				const HashGridParameters p = regir_grid_parameters(f);
				const float3 centre = RegirGridGetPositionFromKey(key, p) - c[9].xyz;
				const uint level = uint((key >> REGIR_GRID_LEVEL_BIT_OFFSET) & REGIR_GRID_LEVEL_BIT_MASK);
				const float radius = RegirGridGetVoxelSize(level, p) * 0.8660254;
				const float3 axis = unpack_normal(h.x);
				const float cone_cos = 1.0 - float(h.y) / 65535.0;
				RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(uint2(id.x & 0xFFFu, id.x >> 12), g_rtxdi.runtime.frameIndex, 2u);
				RTXDI_RandomSamplerState cell_rng = RTXDI_InitRandomSampler(uint2(cell & 0xFFFu, cell >> 12), g_rtxdi.runtime.frameIndex, 3u);
				RTXDI_LocalLightSelectionContext ctx = RTXDI_InitializeLocalLightSelectionContextRIS(cell_rng, g_rtxdi.localRis);
				regir_source source = regir_source_init(ctx, true);   // v96: + the game's lights alike (rtxdi.hlsli)
				const uint m = slot < f.plain ? 1u : f.candidates;     // v92: the plain slots one power-proportional pick (variety)
				uint selected = 0u;
				float selected_target = 0.0, sum = 0.0;
				[loop]
				for (uint i = 0u; i < m; ++i)
				{
					uint light;
					RAB_LightInfo info = RAB_EmptyLightInfo();
					float inv_source_pdf;
					regir_source_next(source, rng, light, info, inv_source_pdf);
					float target = RAB_GetLightTargetPdfForVolume(info, centre, radius);
					if (target > 0.0)
						target *= max(regir_receiver_bound(axis, cone_cos, centre, radius, info.centre - c[9].xyz), 0.0) *
							regir_vis_chance(centre, axis, info.centre - c[9].xyz);   // v93: the visibility cache (>= its floor)
					const float w = target * inv_source_pdf / float(m);
					sum += w;
					if (RTXDI_GetNextRandom(rng) * sum < w)
					{
						selected = light;
						selected_target = target;
					}
				}
				const float weight = selected_target > 0.0 ? sum / selected_target : 0.0;
				Data.Store2(at + 16u + slot * 8u, uint2(selected, asuint(isfinite(weight) ? weight : 0.0)));
				if (slot == 0u)
				{
					Data.Store(at + 12u, g_frame);                  // filled this frame
					InterlockedAdd(g_regir_live, 1u);
				}
			}
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if (gi == 0u && g_regir_live != 0u && (f.flags & REGIR_FLAG_COUNT) != 0u)
		Data.InterlockedAdd(FRAME_REGIR_LIVE, g_regir_live);
}

// v93: the visibility cache's upkeep (regir_vis.hlsli), each frame before the surface cells' fill: an entry's counts halved
// once past 64 rays (the cache follows doors, moving lights), an entry recorded no ray for 60 frames dropped
groupshared uint g_regir_vis_live;
[numthreads(256, 1, 1)]
void cs_regir_vis_decay(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
	if (gi == 0u)
		g_regir_vis_live = 0u;
	GroupMemoryBarrierWithGroupSync();
	const uint4 vp = regir_vis_params();
	if ((vp.z & REGIR_VIS_ON) != 0u && id.x <= vp.y)
	{
		const uint at = vp.x + id.x * REGIR_VIS_BYTES;
		const uint3 e = Data.Load3(at);
		if (e.x != 0u)
		{
			if (g_frame - e.z > 60u)
				Data.Store3(at, uint3(0u, 0u, 0u));
			else
			{
				if ((e.y & 0xFFFFu) > 64u)
					Data.Store(at + 4u, (e.y >> 1) & 0x7FFF7FFFu);
				InterlockedAdd(g_regir_vis_live, 1u);
			}
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if (gi == 0u && g_regir_vis_live != 0u && (Table.Load(TABLE_REGIR + 8) & REGIR_FLAG_COUNT) != 0u)
		Data.InterlockedAdd(FRAME_REGIR_VIS_LIVE, g_regir_vis_live);
}
