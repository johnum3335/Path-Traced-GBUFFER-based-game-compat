// TLOU RT - ReGIR's surface cells (v90; user: "continue with the regir integration" - the ReGIR article's grid: Tom Clabault,
// "ReGIR - An advanced implementation for many-lights offline rendering", 2025). Cells on the scene's surfaces where the
// paths' bounce hits land, keyed by NVIDIA's hash grid (the SHARC SDK's HashGridCommon.h, unmodified - a second instance,
// RegirGrid: its voxels grow with the distance to the camera, the normal's octant in the key), each with REGIR_CELL_SLOTS
// light slots in RTXDI's format (light index, 1 / pdf). regir_cells.hlsl: cs_regir_cells inserts the cells that sparse cosine
// rays from the camera's surfaces reach (and widens each cell's normal cone), cs_regir_fill fills the slots by RIS over
// RTXDI's power-proportional tiles with the cell's weight - v88's light terms over the voxel x the receiver's cosine over
// the cell's normal cone, each the largest anywhere in the cell (no light that reaches a surface of the cell is dropped: no
// bias). rtxdi_lamps_at (rtxdi.hlsli) picks a bounce hit's light from its cell when the cell was filled this frame and holds
// the hit's normal, else from NVIDIA ReGIR's onion as before. Included by re9.hlsl after sharc.hlsli (the SHARC SDK's types).

#define HASH_GRID_KEY_TYPE HashGridKey                     // (as SharcCommon.h sets up its own instance)
#define HASH_GRID_PREFIX RegirGrid
#define HASH_GRID_CONST_PREFIX REGIR_GRID
#include "HashGridCommon.h"
#undef HASH_GRID_KEY_TYPE
#undef HASH_GRID_PREFIX
#undef HASH_GRID_CONST_PREFIX

RWStructuredBuffer<HashGridKey> RegirKeys : register(u8, space100);   // the cells' keys (a root UAV into the data buffer)

// The frame's block in the draw table (tlou_rt.cpp regir_frame at k_table_regir) and a cell's bytes in Data: a header (its
// normal cone's axis - the first inserted normal -, the cone's width as (1 - cos) x 65535 widened with InterlockedMax, the
// frame it was last used, the frame its slots were filled), then the slots
// (TABLE_REGIR: regir_vis.hlsli, included before this by rtxdi_bridge.hlsli)
#define REGIR_FLAG_ON 1u
#define REGIR_FLAG_COUNT 2u
#define REGIR_CELL_SLOTS 32u
#define REGIR_CELL_BYTES (16u + REGIR_CELL_SLOTS * 8u)
#define FRAME_REGIR_LIVE (FRAME_REGIR + 0)       // the status line: cells in use, bounce hits served by a cell / by the onion
#define FRAME_REGIR_SERVED (FRAME_REGIR + 4)
#define FRAME_REGIR_ONION (FRAME_REGIR + 8)
#define FRAME_REGIR_FULL (FRAME_REGIR + 12)
struct regir_frame
{
	float sceneScale;       // the hash grid's scene scale (the voxels: 2^level / sceneScale, level ~ log2 of the distance)
	uint capacity;          // cells
	uint flags;             // REGIR_FLAG_*
	uint cellsAt;           // byte offset of the cells in Data
	uint candidates;        // RIS candidates per slot (cs_regir_fill)
	uint stale;             // frames a cell is kept unused
	uint plain;             // v92: the first slots with one power-proportional pick each (variety)
};
regir_frame regir_load_frame()
{
	const uint4 a = Table.Load4(TABLE_REGIR), b = Table.Load4(TABLE_REGIR + 16);
	regir_frame f;
	f.sceneScale = asfloat(a.x);
	f.capacity = a.y;
	f.flags = a.z;
	f.cellsAt = a.w;
	f.candidates = max(b.x, 1u);
	f.stale = b.y;
	f.plain = b.z;
	return f;
}
bool regir_cells_on()
{
	return (Table.Load(TABLE_REGIR + 8) & REGIR_FLAG_ON) != 0u;
}
HashGridParameters regir_grid_parameters(regir_frame f)
{
	HashGridParameters p;
	p.cameraPosition = c[9].xyz;
	p.logarithmBase = 2.0;
	p.sceneScale = f.sceneScale;
	p.levelBias = 0.0;
	return p;
}
RegirGridData regir_grid_data(regir_frame f)
{
	RegirGridData d;
	d.capacity = f.capacity;
	d.hashEntriesBuffer = RegirKeys;
	return d;
}
// The cone's width stored for a normal at cosine cs from the axis: rounded up (the stored cone holds it)
uint regir_cone_width(float cs)
{
	return uint(ceil(saturate(1.0 - cs) * 65535.0));
}
// The largest cosine any normal within the cell's cone (its axis, cosine cone_cos at its edge) makes with any direction from
// the cell's voxel (centre, radius) toward the light's position: <= 0 - the light is behind every surface of the cell
float regir_receiver_bound(float3 axis, float cone_cos, float3 centre, float radius, float3 light_pos)
{
	const float3 d = light_pos - centre;
	const float len = length(d);
	if (!(len > radius))
		return 1.0;
	const float t = acos(clamp(dot(axis, d / len), -1.0, 1.0));
	return cos(min(max(t - acos(clamp(cone_cos, -1.0, 1.0)) - asin(saturate(radius / len)), 0.0), 3.14159265));
}
void regir_count(uint at, bool add)
{
	const uint n = WaveActiveCountBits(add);
	if (WaveIsFirstLane() && n != 0u && (Table.Load(TABLE_REGIR + 8) & REGIR_FLAG_COUNT) != 0u)
		Data.InterlockedAdd(at, n);
}

// The cell of a surface point (p relative to the camera, its normal n): the cell's byte offset in Data when a cell filled
// this frame holds it, else 0 (the onion's turn). A point whose normal lies outside its cell's cone widens the cone for the
// next fill (v98-: regir_cells_pick's; v99 apart - the RTXPT renderer's NEE takes the cell's slots as its local light list)
uint regir_cells_find(float3 p, float3 n)
{
	if (!regir_cells_on())
		return 0u;
	const regir_frame f = regir_load_frame();
	HashGridKey key;
	const HashGridIndex index = RegirGridFindEntry(regir_grid_data(f), p + c[9].xyz, n, regir_grid_parameters(f), key);
	bool ok = index != HASH_GRID_INVALID_CACHE_INDEX && index < f.capacity;
	uint at = 0u;
	if (ok)
	{
		at = f.cellsAt + index * REGIR_CELL_BYTES;
		const uint4 h = Data.Load4(at);
		const float cs = dot(n, unpack_normal(h.x));
		if (h.x == 0u || cs < 1.0 - float(h.y) / 65535.0)
		{
			Data.InterlockedMax(at + 4u, regir_cone_width(cs));
			ok = false;
		}
		else
			ok = h.w == g_frame;
		Data.Store(at + 8u, g_frame);                       // used: the cell stays
	}
	regir_count(FRAME_REGIR_SERVED, ok);
	regir_count(FRAME_REGIR_ONION, !ok);
	return ok ? at : 0u;
}

// A bounce hit's light from its cell (rtxdi_lamps_at): false - no cell filled this frame holds the hit (the onion's turn);
// true - the pick (sample, its 1 / pdf; 0: none) by RIS over g_rtxdi.hitSamples random slots with the surface's own target,
// RTXDI's local light sampling over the cell
bool regir_cells_pick(RAB_Surface surface, uint seed, out RAB_LightSample sample, out float inv_pdf)
{
	sample = RAB_EmptyLightSample();
	inv_pdf = 0.0;
	const uint at = regir_cells_find(surface.worldPos, surface.normal);
	if (at == 0u)
		return false;
	uint rng = pcg(seed + 0x2F9Bu);
	float sum = 0.0, chosen = 0.0;
	[loop]
	for (uint i = 0u; i < g_rtxdi.hitSamples; ++i)
	{
		rng = pcg(rng);
		const uint2 s = Data.Load2(at + 16u + (rng % REGIR_CELL_SLOTS) * 8u);
		const float w_slot = asfloat(s.y);
		if (!(w_slot > 0.0) || !isfinite(w_slot))
			continue;
		rng = pcg(rng);
		const RAB_LightSample candidate = RAB_SamplePolymorphicLight(RAB_LoadLightInfo(s.x, false), surface,
			float2(float(rng & 0xFFFFu), float(rng >> 16)) / 65536.0);
		const float w = RAB_GetLightSampleTargetPdfForSurface(candidate, surface) * w_slot / float(g_rtxdi.hitSamples);
		if (!(w > 0.0))
			continue;
		sum += w;
		rng = pcg(rng);
		if (float(rng >> 8) / 16777216.0 * sum < w)
		{
			sample = candidate;
			chosen = w;
		}
	}
	// (the chosen candidate's target = its w x hitSamples / its slot's weight; 1 / pdf = sum / target)
	if (chosen > 0.0)
	{
		const float target = RAB_GetLightSampleTargetPdfForSurface(sample, surface);
		inv_pdf = target > 0.0 ? sum / target : 0.0;
	}
	return true;
}

// Debug view 10 "ReGIR cells": the camera's surfaces coloured by their cell (NVIDIA's HashGridDebugColoredHash), dark grey
// without a cell filled this frame
float3 regir_debug_cells(uint2 px, float d)
{
	if (!regir_cells_on())
		return 0.0;
	const regir_frame f = regir_load_frame();
	const float3 p = camera_relative_position(float2(px), d);
	const float3 ng = geometric_normal(px, p, decode_normal(GBuffer0[px]));
	HashGridKey key;
	const HashGridIndex index = RegirGridFindEntry(regir_grid_data(f), p + c[9].xyz, ng, regir_grid_parameters(f), key);
	if (index == HASH_GRID_INVALID_CACHE_INDEX || index >= f.capacity || Data.Load(f.cellsAt + index * REGIR_CELL_BYTES + 12u) != g_frame)
		return 0.03;
	return RegirGridDebugColoredHash(p + c[9].xyz, ng, regir_grid_parameters(f));
}
