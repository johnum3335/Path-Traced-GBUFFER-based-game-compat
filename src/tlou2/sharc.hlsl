// TLOU RT - NVIDIA SHaRC's own passes (v87; sharc.hlsli is the bridge, NVIDIA's SHARC SDK 1.8.3 compiled in unmodified -
// "This software contains source code provided by NVIDIA Corporation."): SHaRC Update and SHaRC Resolve as its integration
// guide lays them out (RTGI_builds\deps\sharc\AGENTS.md, docs\Integration.md), and the cache's reset. cs_sharc_update needs
// 64-bit atomics (HashGridCommon.h's key insertion: shader model 6.6, build_rt.cmd), the others shader model 6.5.
// The order each frame (tlou_rt.cpp post_lighting, after the lamps' RTXDI passes, before the renderers): Update, a UAV
// barrier (its writes seen by the Resolve), Resolve, a UAV barrier (its writes seen by the renderers' queries)
#define SHARC_UPDATE 1
#include "re9.hlsl"

// No value the cache's fixed point cannot hold (SharcAddVoxelData converts to unsigned: a negative or infinite light wraps)
float3 sharc_clamp(float3 radiance)
{
	return all(isfinite(radiance)) ? clamp(radiance, 0.0, 1000.0) : 0.0;
}

// SHaRC Update (AGENTS.md 14.3-14.10): one path per downscale x downscale tile of the render pixels, the tile's pixel another
// one every frame (the whole view covered over time), from the camera's surface there - a split-primary renderer (AGENTS.md
// 14.2, case 2): the RE9 renderer's primary from the G-buffer, re-traced onto the scene's triangle (cs_re9_path's recast),
// the path's root. Two submissions:
//  - the camera's view: the game's own light leaving that surface (dd2_seen_light) as a one-vertex path - what v23-v86's own
//    SHaRC kept of the camera's view (cs_re9_history): the cache remembers the game's lighting where the camera saw it
//  - the path: up to `bounces` BSDF samples (the lobes by their albedos, as the renderers'; the roughness at least
//    roughnessMin - AGENTS.md 14.4: no local highlights in the cache), at each hit its light - the game's own where the
//    camera sees the hit (the light leaving it whole: the path ends there), else DD2's estimate there (re9_hit_estimate: the
//    hit's direct light, as the renderers' bounces take it) and on; a miss: the sky as the RE9 renderer's bounce takes it
//    (the root pixel's terms). SharcUpdateHit carries each vertex's light back along the path (SharcSetThroughput: each
//    segment's f cos / pdf); it returns false where the cache's own light took the rest of the path (cache resampling)
[numthreads(8, 8, 1)]
void cs_sharc_update(uint3 id : SV_DispatchThreadID)
{
	const sharc_frame f = sharc_load_frame();
	const uint2 size = full_size();
	const uint tile = f.downscale;
	const uint2 tiles = (size + tile - 1u) / tile;
	if (id.x >= tiles.x || id.y >= tiles.y)
		return;
	uint rng = pcg(id.x + pcg(id.y + pcg(g_frame * 0x2C1B3C6Du + 0x5A4Cu)));
	const uint2 px = min(id.xy * tile + uint2(rng % tile, (rng / tile) % tile), size - 1u);
	rng = pcg(rng);
	const float d = Depth[px];
	if (d >= 1.0 || (pixel_flags(px) & PIXEL_HAIR) != 0u)
		return;                                             // the sky; hair (the scene's rays leave it out)
	// the root: cs_re9_path's primary - the camera ray's end re-traced onto the scene's triangle, its normal
	surface s = load_surface(px, d);
	float3 ng = geometric_normal(px, s.p, s.n);
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
	s.n = visible_normal(s.n, s.v, ng);
	const SharcParameters sp = sharc_parameters();
	uint camera = 0u, vertices = 0u;
	// the camera's view
	{
		float3 seen;
		if (dd2_seen_light(s.p, ng, seen))
		{
			SharcState view;
			SharcInit(view);
			SharcUpdateHit(sp, view, sharc_hit(s.p, ng), sharc_clamp(seen), dd2_next(rng));
			camera = 1u;
		}
	}
	// the path
	const float3 a4 = InT4[px].rgb, a5 = InT5[px].rgb;
	const float3 axis = decode_ambient_direction(GBuffer1[px]);
	const float2 uv = (float2(px) + 0.5) * c[8].zw;
	const float open_k = 1.0;                               // v97: no game sky visibility (the Occlusion slider is gone)
	const float3 post = max(lighting_factor(px, d) * g_brightness, 1e-4);
	const bool emit_on = g_emit > 0.0 && Data.Load(emit_list_offset() + 8) != 0u;
	SharcState state;
	SharcInit(state);
	float3 o = s.p + ng * (0.005 + length(s.p) * 0.001);
	float3 n = s.n, v = s.v;
	float rough = max(s.roughness, f.roughnessMin);
	[loop]
	for (uint b = 0u; b < f.bounces; ++b)
	{
		// the BSDF sample: the specular lobe with the chance of its albedo (kept between 0.1 and 0.9), GGX's visible normals,
		// else the cosine; the segment's throughput f cos / the mixture's pdf (clamped: the cache's fixed point)
		const float lad = restir_lum(ad), las = restir_lum(as);
		const float p_spec = lad + las > 0.0 ? clamp(las / (lad + las), 0.1, 0.9) : 0.5;
		const float alpha = max(rough * rough, 0.002);
		const float2 xi = float2(dd2_next(rng), dd2_next(rng));
		const bool spec = dd2_next(rng) < p_spec;
		const float3 l = spec ? reflect(-v, sample_ggx_vndf(v, n, alpha, xi)) : cosine_direction(n, xi);
		const float cosl = dot(n, l), ndv = max(dot(n, v), 1e-4);
		if (!(cosl > 0.0))
			break;
		const float3 h = normalize(v + l);
		const float dh = rtxdi_ggx_d(saturate(dot(n, h)), alpha), g1v = rtxdi_smith_g1(ndv, alpha);
		const float3 F = f0 + (1.0 - f0) * pow(1.0 - saturate(dot(v, h)), 5.0);
		const float pdf = (1.0 - p_spec) * cosl * 0.31830989 + p_spec * g1v * dh / (4.0 * ndv);
		const float3 fcos = ad * (cosl * 0.31830989) + F * (dh * g1v * rtxdi_smith_g1(cosl, alpha) / (4.0 * ndv));
		if (!(pdf > 0.0))
			break;
		const float3 throughput = min(fcos / pdf, 2.0);
		if (!all(isfinite(throughput)) || !any(throughput > 0.0))
			break;
		SharcSetThroughput(state, throughput);
		const scene_hit hh = trace_cutouts(o, l, DD2_TMAX);
		if (!hh.hit)
		{
			SharcUpdateMiss(sp, state, sharc_clamp(dd2_escaped(px, l, a4, a5, axis, open_k, 1.0) * g_sky));
			break;
		}
		const float3 hit = o + l * hh.t;
		float3 nh = hit_normal(hh.draw, hh.primitive);
		if (dot(nh, l) > 0.0)
			nh = -nh;                                       // toward the ray
		const SharcHitData hd = sharc_hit(hit, nh);
		float3 seen;
		if (dd2_seen_light(hit, nh, seen))
		{
			SharcUpdateHit(sp, state, hd, sharc_clamp(seen), dd2_next(rng));
			++vertices;
			break;                                          // the game's light leaving the hit is whole: nothing beyond it
		}
		const float3 direct = re9_hit_estimate(px, hit, nh, l, hh, a4, a5, axis, post, emit_on, f.roughnessMin, rng);
		++vertices;
		if (!SharcUpdateHit(sp, state, hd, sharc_clamp(direct), dd2_next(rng)))
			break;
		// the next vertex: the hit's material (the material cache), the roughness at least roughnessMin
		const float4 m = hit_material(hh.draw, hh.primitive);
		const float2 mp = hit_material_pbr(hh.draw, hh.primitive);
		ad = m.rgb * (1.0 - mp.x);
		f0 = lerp(mp.y, m.rgb, mp.x);
		as = f0;
		rough = max(f.roughnessMin, m.w >= 0.0 ? m.w : 0.5);
		n = nh;
		v = -l;
		o = hit + nh * (DD2_EPS + length(hit) * 0.0005);
	}
	// the status line: the camera's view and the paths' vertices submitted
	const uint2 counts = WaveActiveSum(uint2(camera, vertices));
	if (WaveIsFirstLane() && (f.flags & SHARC_FLAG_COUNT) != 0u)
	{
		Data.InterlockedAdd(STATS + STAT_SHARC_INJECTED * 4, counts.x);
		Data.InterlockedAdd(STATS + STAT_SHARC_WRITES * 4, counts.y);
	}
}

// SHaRC Resolve (AGENTS.md 13): SharcResolveEntry for every entry - this frame's accumulation into the resolved light (over
// up to accumulationFrames frames), the adjacent levels' blend on camera movement (last frame's camera: FRAME_CAM_PREV, the
// absolute position cs_setup kept), an entry without samples for staleFrames frames evicted, the accumulation cleared.
// AGENTS.md 20's reset on a scene reload: the add-on sees no level load, but a load, a respawn or a cut to another place
// moves the camera further than SHARC_JUMP metres in a frame (walking, running, riding: well under a metre) - every entry
// is emptied in place of the resolve then (this frame's samples dropped; the next frames fill it again)
#define SHARC_JUMP 25.0
groupshared uint g_sharc_live;
[numthreads(256, 1, 1)]
void cs_sharc_resolve_nv(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
	const sharc_frame f = sharc_load_frame();
	if (gi == 0u)
		g_sharc_live = 0u;
	GroupMemoryBarrierWithGroupSync();
	SharcResolveParameters r;
	r.cameraPositionPrev = asfloat(Data.Load3(FRAME_CAM_PREV));
	r.accumulationFrameNum = f.accumulationFrames;
	r.responsiveFrameNum = f.accumulationFrames;        // (responsive lighting off: unused)
	r.staleFrameNumMax = f.staleFrames;
	r.frameIndex = g_frame;
	const float3 jump = c[9].xyz - r.cameraPositionPrev;
	if (all(isfinite(jump)) && dot(jump, jump) <= SHARC_JUMP * SHARC_JUMP)
		SharcResolveEntry(id.x, sharc_parameters(), r);
	else if (id.x < f.capacity)
	{
		SharcHashEntries[id.x] = HASH_GRID_INVALID_HASH_KEY;      // the camera jumped: the cache emptied (as cs_sharc_clear)
		SharcAccumulation[id.x] = SharcZeroAccumulationData();
		SharcResolved[id.x] = SharcZeroPackedData();
	}
	// the status line: the entries in use after the resolve (the group's count, one atomic per group: 4M entries)
	bool live = false;
	if (id.x < f.capacity)
		live = SharcHashEntries[id.x] != HASH_GRID_INVALID_HASH_KEY;
	const uint n = WaveActiveCountBits(live);
	if (WaveIsFirstLane() && n != 0u)
		InterlockedAdd(g_sharc_live, n);
	GroupMemoryBarrierWithGroupSync();
	if (gi == 0u && g_sharc_live != 0u && (f.flags & SHARC_FLAG_COUNT) != 0u)
		Data.InterlockedAdd(STATS + STAT_SHARC_LIVE * 4, g_sharc_live);
}

// The cache's reset (first use, the "Reset SHaRC" button): every entry zero, as NVIDIA's guide asks for a new cache
[numthreads(256, 1, 1)]
void cs_sharc_clear(uint3 id : SV_DispatchThreadID)
{
	const sharc_frame f = sharc_load_frame();
	if (id.x >= f.capacity)
		return;
	SharcHashEntries[id.x] = HASH_GRID_INVALID_HASH_KEY;
	SharcAccumulation[id.x] = SharcZeroAccumulationData();
	SharcResolved[id.x] = SharcZeroPackedData();
}
