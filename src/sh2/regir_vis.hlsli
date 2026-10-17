// TLOU RT - ReGIR's visibility cache (v93; user: "visibility cache for regir is next up" - the ReGIR article's NEE++: Guo et
// al. 2020, how well a surface sees a light, learnt from the shadow rays the renderers trace anyway). A hash map in Data from
// (a receiver voxel - the world position at levels growing with the distance to the camera, 2^level / REGIR_VIS_SCALE across,
// and the normal's octant -, a light's voxel) to the shadow rays from there toward that light: how many saw it, of how many.
// RAB_GetConservativeVisibility records an eighth of the rays it traces (RTXDI's at the camera's surfaces, the bounce hits'
// picks); regir_cells.hlsl cs_regir_vis_decay halves the counts past 64 and drops entries unused for 60 frames. The lights'
// weights read it as the chance the light is seen from there - the camera's surfaces' and the bounce hits' RIS
// (RAB_GetLightSampleTargetPdfForSurface), the surface cells' fill - never below the floor: a light a ray has not seen yet
// keeps a fair chance (no light is ever dropped: no bias), a shadowed strong light no longer crowds out the others.
// Included by rtxdi_bridge.hlsli (rt.hlsl's Data, Table, c[9], pcg)

#define TABLE_REGIR 805632                  // tlou_rt.cpp k_table_regir: regir_frame (regir_cells.hlsli), then:
#define TABLE_REGIR_VIS (TABLE_REGIR + 32)  // uint4 - the entries' byte offset in Data, their count - 1, flags, the floor (float)
#define REGIR_VIS_ON 1u
#define REGIR_VIS_PROBES 8u
#define REGIR_VIS_BYTES 16u                 // an entry: key, (rays that saw the light << 16) | rays, the frame last recorded, -
#define REGIR_VIS_SCALE 8.0
#define REGIR_VIS_LIGHT_VOXEL 0.25          // metres
#define FRAME_REGIR_VIS_LIVE (FRAME_REGIR + 16)   // the status line: entries in use, rays recorded
#define FRAME_REGIR_VIS_RAYS (FRAME_REGIR + 20)

uint4 regir_vis_params()
{
	return Table.Load4(TABLE_REGIR_VIS);
}
bool regir_vis_on()
{
	return (Table.Load(TABLE_REGIR_VIS + 8) & REGIR_VIS_ON) != 0u;
}
// The pair's key: the receiver (p, n camera-relative) and the light (camera-relative position), never 0 (an empty entry)
uint regir_vis_key(float3 p_rel, float3 n, float3 light_rel)
{
	const float3 p = p_rel + c[9].xyz;
	const uint level = uint(clamp(floor(0.5 * log2(max(dot(p_rel, p_rel), 1e-6))), 1.0, 14.0));
	const int3 r = int3(floor(p * (REGIR_VIS_SCALE / exp2(float(level)))));
	const int3 l = int3(floor((light_rel + c[9].xyz) * (1.0 / REGIR_VIS_LIGHT_VOXEL)));
	const uint octant = (n.x >= 0.0 ? 0u : 1u) | (n.y >= 0.0 ? 0u : 2u) | (n.z >= 0.0 ? 0u : 4u);
	uint k = pcg(uint(r.x) + pcg(uint(r.y) + pcg(uint(r.z) + pcg(level | (octant << 8)))));
	k = pcg(k ^ (uint(l.x) + pcg(uint(l.y) + pcg(uint(l.z)))));
	return max(k, 1u);
}
// A shadow ray's outcome (from p, normal n, toward the light at light_rel): an eighth of them, by the pair and the pixel
void regir_vis_record(float3 p_rel, float3 n, float3 light_rel, bool visible)
{
	const uint4 vp = regir_vis_params();
	if ((vp.z & REGIR_VIS_ON) == 0u)
		return;
	const uint key = regir_vis_key(p_rel, n, light_rel);
	if ((pcg(key ^ (g_frame * 0x9E3779B9u) ^ asuint(p_rel.x) ^ (asuint(p_rel.y) << 7)) & 7u) != 0u)
		return;
	[loop]
	for (uint i = 0u; i < REGIR_VIS_PROBES; ++i)
	{
		const uint at = vp.x + ((key + i) & vp.y) * REGIR_VIS_BYTES;
		uint prev;
		Data.InterlockedCompareExchange(at, 0u, key, prev);
		if (prev == 0u || prev == key)
		{
			Data.InterlockedAdd(at + 4u, visible ? 0x10001u : 1u);
			Data.Store(at + 8u, g_frame);
			const uint n_rec = WaveActiveCountBits(true);
			if (WaveIsFirstLane() && (Table.Load(TABLE_REGIR + 8) & 2u) != 0u)   // (regir_cells.hlsli REGIR_FLAG_COUNT)
				Data.InterlockedAdd(FRAME_REGIR_VIS_RAYS, n_rec);
			return;
		}
	}
}
// The chance the light at light_rel is seen from p (normal n): the recorded rays' share (with a half ray of doubt), at least
// the floor; 1 where fewer than 4 rays were recorded (no guess)
float regir_vis_chance(float3 p_rel, float3 n, float3 light_rel)
{
	const uint4 vp = regir_vis_params();
	if ((vp.z & REGIR_VIS_ON) == 0u)
		return 1.0;
	const uint key = regir_vis_key(p_rel, n, light_rel);
	[loop]
	for (uint i = 0u; i < REGIR_VIS_PROBES; ++i)
	{
		const uint2 e = Data.Load2(vp.x + ((key + i) & vp.y) * REGIR_VIS_BYTES);
		if (e.x == key)
		{
			const uint rays = e.y & 0xFFFFu;
			if (rays < 4u)
				return 1.0;
			return max((float(e.y >> 16) + 0.5) / (float(rays) + 1.0), asfloat(vp.w));
		}
	}
	return 1.0;
}
