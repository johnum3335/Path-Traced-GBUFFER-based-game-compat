// TLOU RT - the RE9 renderer (v44): Resident Evil Requiem's path tracer ported onto TLOU RT's scene (the stream-out BVH,
// the game's lights, the add-on's sky). Reconstructed from RE9's own shaders (RTGI_builds\re9-pt-analysis\pt_algorithms.txt):
// full render resolution, one path per pixel; the camera ray's end re-traced against the BVH (RecastPrimaryRays); direct
// light from one reservoir over every light with one shadow ray (here ReSTIR DI, rt.hlsl di_shade); one BSDF sample whose
// lobe is picked by the lobes' albedos (luma(Ad) : luma(As)), GGX visible-normal sampling for the specular lobe; split
// diffuse / specular throughput, demodulated by each lobe's albedo (what DLSS Ray Reconstruction takes); the light at the
// path's hit from the camera's view of it, SHaRC or its own bounce (hit_light), the sky for escaped paths. Until Ray
// Reconstruction (v45) a reprojected temporal accumulation stands in for the denoiser; the composite then writes the
// light buffer as cs_full did. Everything else (scene, lights, sky, caches) is rt.hlsl's.
#include "rt.hlsl"
#include "rtxdi.hlsli"                                      // v77: NVIDIA's RTXDI (its bridge, ReGIR for the bounce hits' lamps)
#include "sharc.hlsli"                                      // v87: NVIDIA's SHaRC (its cache's buffers, the bounces' queries)
#include "regir_cells.hlsli"                                // v90: ReGIR's surface cells (the bounce hits' lamp picks)

#define RE9_RAW_D 0             // full-resolution float4 planes after ReSTIR PT's: this frame's diffuse sample (demodulated)
#define RE9_RAW_S 1             //   ... specular sample (demodulated; w: hit distance - RR's specular guide)
#define RE9_ACC_D 2             //   2, 3: accumulated diffuse (rgb, frames), ping-pong by frame parity
#define RE9_ACC_S 4             //   4, 5: accumulated specular (rgb, frames)
#define RE9_GEO 6               //   6, 7: view depth and normal the accumulation was made at
#define RE9_DIRECT 8            //   this frame's direct light (ReSTIR DI; the light buffer's units before the lighting's factor),
                                //   w: v45 the contact ray's visibility
#define RE9_BOOST 9             //   the share of the composite the Bounce slider added: kept out of the history (cs_history)
#define RE9_DIRECT_D 10         //   v47: the direct light's diffuse part (RE9_DIRECT keeps the rest); w: the subsurface amount
#define RE9_SSS_E 11            //   v47: the skin's diffuse light for the blur (direct and indirect, demodulated by Ad; w: amount)
#define RE9_SSS_H 12            //   v47: ... after the blur's horizontal pass (cs_re9_sss)
#define RE9_RTXDI 13            //   v76: NVIDIA RTXDI's lamp light (rtxdi_di.hlsl cs_rtxdi_di; in di_shade's units, diffuse;
                                //   v86: rgb its diffuse part, w the specular - pack_r11g11b10's bits - and the game's lights
                                //   in it with "NVIDIA RTXDI game lights")
#define RE9_SETTLED 14          //   v81: the composite's accumulated light (the light buffer's, as without RR in the game) while the
                                //   light buffer holds this frame's noisy light for in-game RR: the history keeps it (cs_re9_history)
#define RE9_PLANES 15
#define RE9_RECAST 0.05         // metres (plus 1% of the distance) around the depth buffer's surface the camera ray is re-traced
#define RE9_RECAST_FRONT 0.3    // v54: ... and (plus 5% of the distance) in front of it: a parallax surface (custom depth: snow
                                //   ruts, decals) lies under its triangle by its relief, and its rays started under it
#define RELIGHT_SKY 0xFFFFFFFFu // v54: a sky pixel's P in the relight data - the relight leaves it as the game drew it
#define RE9_MAX 64.0            // the largest luminance one path sample may bring (fireflies)
#define RE9_HISTORY_D 32.0
#define RE9_HISTORY_S 32.0

uint re9_offset(uint plane)
{
	const uint2 size = full_size();
	return surface_offset(2) + plane * align_up(size.x * size.y * 16, 256);   // v76: after the adapter's surfaces (adapter.hlsl)
}
float4 re9_load(uint plane, uint2 px)
{
	return asfloat(Data.Load4(re9_offset(plane) + (px.y * full_size().x + px.x) * 16));
}
void re9_store(uint plane, uint2 px, float4 v)
{
	Data.Store4(re9_offset(plane) + (px.y * full_size().x + px.x) * 16, asuint(v));
}

float smith_g1(float nx, float alpha)
{
	const float a2 = alpha * alpha;
	return 2.0 * nx / max(nx + sqrt(a2 + (1.0 - a2) * nx * nx), 1e-6);
}

// v70: the RE9 renderer's firefly limit for one sample's light on the pixel, luma(Ad x diffuse + As x specular): RE9_MAX
// times the albedos' luminance - the same as v44-v69 for diffuse light (RE9_MAX on the demodulated value), the specular no
// longer cut where its albedo is small
float re9_clamp_limit(float3 ad, float3 as)
{
	return RE9_MAX * max(restir_lum(ad + as), 0.02);
}

// The lobes' albedos at surface s: diffuse (1 - metallic) x albedo, specular from the game's own split-sum LUT (t19) - RE9
// picks the lobe by their luminances and demodulates each lobe's light by its albedo
void re9_albedos(surface s, out float3 ad, out float3 as, out float3 f0)
{
	ad = s.albedo * (1.0 - s.metallic);
	f0 = lerp(s.spec * (0.08 / 255.0), s.albedo, s.metallic);
	const float2 lut = InT19.SampleLevel(SamplerS3, float2(saturate(s.roughness), saturate(dot(s.n, s.v))) * 0.96875 + 0.015625, 0.0).xy;
	as = f0 * lut.x + lut.y;
}

// v47: the skin's subsurface scattering, CS_SsssSepHor / CS_SsssSepVert's separable kernel (ssssSepKernel: 13 one-sided
// taps, rgb weight and offset - read as millimetres) over this renderer's own diffuse light instead of CS_SkinDiffuse's
// (the game's lighting). Taps on subsurface pixels within RE9_SSS_DEPTH of the pixel's depth, normalized by the weights
// taken (as the game's). The game jitters 7 of the 25 taps per pixel; all 25 here.
static const float4 RE9_SSS_KERNEL[13] = {
	float4(0.05807, 0.11255, 0.18867, 0.0),
	float4(0.07174, 0.12400, 0.17900, 0.0719),
	float4(0.07652, 0.10598, 0.11244, 0.2876),
	float4(0.07196, 0.07982, 0.06205, 0.6470),
	float4(0.06390, 0.05614, 0.03137, 1.1503),
	float4(0.05400, 0.03649, 0.01372, 1.7973),
	float4(0.04304, 0.02138, 0.00504, 2.5881),
	float4(0.03229, 0.01120, 0.00155, 3.5227),
	float4(0.02281, 0.00524, 0.00039, 4.6010),
	float4(0.01532, 0.00220, 0.00008, 5.8232),
	float4(0.01002, 0.00086, 0.00001, 7.1891),
	float4(0.00674, 0.00033, 0.0, 8.6988),
	float4(0.00263, 0.00010, 0.0, 10.3523) };
#define RE9_SSS_MM 0.001        // metres per kernel offset unit
#define RE9_SSS_DEPTH 0.02      // metres: a tap further in depth is another surface (an ear over the cheek, a hand)

float4 re9_sss_blur(uint plane, uint2 px, float d, int2 axis)
{
	const uint2 size = full_size();
	const float4 centre = re9_load(plane, px);
	const float3 p = camera_relative_position(float2(px), d);
	const float step = length(camera_relative_position(float2(px) + float2(axis), d) - p);   // metres per pixel along the axis
	const float vz = view_z(d);
	float3 sum = centre.rgb * RE9_SSS_KERNEL[0].rgb, wsum = RE9_SSS_KERNEL[0].rgb;
	[unroll]
	for (int k = 1; k < 13; ++k)
	{
		const float offset = RE9_SSS_KERNEL[k].w * RE9_SSS_MM / max(step, 1e-6);
		[unroll]
		for (int side = -1; side <= 1; side += 2)
		{
			const int2 q = int2(px) + axis * int(round(offset * side));
			if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y))
				continue;
			const float dq = Depth[uint2(q)];
			if (dq >= 1.0 || abs(view_z(dq) - vz) > RE9_SSS_DEPTH)
				continue;
			const float4 v = re9_load(plane, uint2(q));
			if (!(v.w > 0.0) || !all(isfinite(v.rgb)))
				continue;
			sum += max(v.rgb, 0.0) * RE9_SSS_KERNEL[k].rgb;
			wsum += RE9_SSS_KERNEL[k].rgb;
		}
	}
	return float4(sum / wsum, centre.w);
}

// ---- v68: a pixel's path as a sample for the path reuse (reuse.hlsl: ReSTIR PT Enhanced, both renderers) ----
// The path's reconnection vertex - its first hit (world position, the hit's normal toward the primary) or, for a path that
// left the scene, its direction - the light the path brings back to the primary (rgb, clamped as the raw sample) and
// W = 1 / the direction's pdf (one candidate), 32 bytes per pixel where v48-v67 kept RE9's ReSTIR GI reservoirs (two lobes
// x two frames of 16 bytes - the same room, which reuse.hlsl replaces: v48's temporal and spatial passes are gone)
#define REUSE_NONE 0u
#define REUSE_HIT 1u
#define REUSE_SKY 2u

uint re9_gi_offset(uint lobe, uint parity)                  // after the planes, four planes of 16 bytes (v48: lobe x frame parity)
{
	const uint2 size = full_size();
	return re9_offset(RE9_PLANES) + (lobe * 2u + parity) * align_up(size.x * size.y * 16, 256);
}
struct reuse_sample
{
	uint kind;                  // REUSE_*
	float3 x;                   // hit: world position; sky: the direction
	float3 n;                   // hit: its normal, toward the primary
	float3 L;                   // the light the path brings to the primary
	float W;                    // 1 / the direction's pdf
};
void reuse_store(uint2 px, reuse_sample r)
{
	const uint at = re9_gi_offset(0, 0) + (px.y * full_size().x + px.x) * 32;
	const float3 L = clamp(r.L, 0.0, 60000.0);
	Data.Store4(at, uint4(asuint(r.x), r.kind == REUSE_HIT ? pack_normal(r.n) : 0u));
	Data.Store4(at + 16, uint4(f32tof16(L.x) | (f32tof16(L.y) << 16), f32tof16(L.z) | (r.kind << 16), asuint(r.W), 0u));
}
reuse_sample reuse_load(uint2 px)
{
	const uint at = re9_gi_offset(0, 0) + (px.y * full_size().x + px.x) * 32;
	const uint4 a = Data.Load4(at), b = Data.Load4(at + 16);
	reuse_sample r;
	r.x = asfloat(a.xyz);
	r.n = unpack_normal(a.w);
	r.L = float3(f16tof32(b.x), f16tof32(b.x >> 16), f16tof32(b.y));
	r.kind = (b.y >> 16) & 3u;
	r.W = asfloat(b.z);
	if (!all(isfinite(r.x)) || !all(isfinite(r.L)) || !(r.W > 0.0 && r.W < 1e8))
		r.kind = REUSE_NONE;                                // never written, none this frame, or broken
	return r;
}

// v50: the forward relight's data after the reservoirs (tlou_rt.cpp data_layout relight_*): a copy of the light buffer
// (row pitch copy_pitch(); src\relight.hlsl works on it before the game's fog), per pixel the game's lighting result G and
// this renderer's P (R11G11B10 each), then a 16-byte header (x: the frame token) and per 16 x 16 tile the luminance sums
#define RELIGHT_TILE 16
#define RELIGHT_BLEND 0.1       // v62: the share of this frame's tile sums (cs_re9_ratio), the rest last frame's
uint relight_image_offset()
{
	return align_up(re9_gi_offset(2, 0), 512);
}
uint relight_gp_offset()
{
	return align_up(relight_image_offset() + copy_pitch() * full_size().y, 256);
}
uint relight_tile_offset()
{
	const uint2 size = full_size();
	return align_up(relight_gp_offset() + size.x * size.y * 8, 256);
}

// v68: the path reuse's planes after the relight's tiles (tlou_rt.cpp layout_for): per pixel the path's ray origin
// (camera-relative, after the recast) and the triangle's normal there, 16 bytes; then reuse.hlsl's shifts and pairing
// textures
uint reuse_origin_offset()
{
	const uint2 size = full_size();
	return align_up(relight_tile_offset() + 16 + ((size.x + RELIGHT_TILE - 1) / RELIGHT_TILE) * ((size.y + RELIGHT_TILE - 1) / RELIGHT_TILE) * 8, 256);
}
void reuse_origin_store(uint2 px, float3 origin, float3 ng)
{
	Data.Store4(reuse_origin_offset() + (px.y * full_size().x + px.x) * 16, uint4(asuint(origin), pack_normal(ng)));
}

// The lobes' integrands toward l, demodulated (what the path's sample weights are times its pdf): diffuse cos / pi (/ Ad);
// specular F D G1(v) G1(l) / (4 N.V) (/ As, the split-sum albedo), with pdf_s the GGX visible-normal pdf of l, G1(v) D / (4 N.V)
float re9_ggx_d(float ndh, float alpha)
{
	const float a2 = alpha * alpha;
	const float k = ndh * ndh * (a2 - 1.0) + 1.0;
	return min(a2 / (3.14159265 * k * k), 1e9);
}
float3 re9_spec_integrand(surface s, float3 l, float3 f0, float3 as, out float pdf_s)
{
	pdf_s = 0.0;
	const float ndl = dot(s.n, l), ndv = max(dot(s.n, s.v), 1e-4);
	if (ndl <= 0.0)
		return 0.0;
	const float alpha = max(s.roughness * s.roughness, 0.002);
	const float3 h = normalize(s.v + l);
	pdf_s = smith_g1(ndv, alpha) * re9_ggx_d(saturate(dot(s.n, h)), alpha) / (4.0 * ndv);
	const float3 F = f0 + (1.0 - f0) * pow(1.0 - saturate(dot(s.v, h)), 5.0);
	return F * (pdf_s * smith_g1(ndl, alpha)) / max(as, 1e-4);
}
// [3a] DD2's light response for unit light colour (x pi: TLOU's light units), diffuse and specular apart (dd2-pt-analysis
// pt_algorithm.txt; v70: here, for both renderers' sun - v52-v69 in dd2.hlsl)
void dd2_response(float3 n, float3 v, float3 l, float3 base, float metal, float3 f0, float a, float ecomp, out float3 diffuse, out float3 specular)
{
	diffuse = 0.0;
	specular = 0.0;
	const float nol = dot(n, l), nov = saturate(dot(n, v));
	if (nol <= 0.0 || nov <= 0.0)
		return;
	const float lov = dot(l, v);
	const float k = rsqrt(max(2.0 + 2.0 * lov, 1e-6));
	const float noh = (nol + nov) * k;
	const float voh = 0.5 * (2.0 + 2.0 * lov) * k;
	const float fc = pow(saturate(1.0 - voh), 5.0);
	const float a2 = a * a;
	const float d = noh * noh * (a2 - 1.0) + 1.0;
	const float lv = lerp(2.0 * nov * nol, nol + nov, a);
	const float dvnol = min(nol * a2 / max(d * d * 2.0 * lv, 1e-8), 65000.0);
	const float facing = 0.5 + 0.5 * lov;
	const float rough = facing * (0.7 - 0.2 * lov) * (1.0 + 0.5 / max(noh, 1e-4));
	const float smooth = 1.05 * (1.0 - pow(saturate(1.0 - nol), 5.0)) * (1.0 - pow(saturate(1.0 - nov), 5.0));
	const float single = saturate(lerp(smooth, rough, a) * nol);
	const float multi = max(0.0, 1.14 * a * nol);
	const float3 diff_alb = base * (1.0 - metal) * 0.31830989;
	diffuse = saturate(single + multi * diff_alb) * diff_alb * 3.14159265;
	specular = (dvnol * f0 + dvnol * fc * (1.0 - f0)) * (1.0 + f0 * ecomp);
}

// The split-sum LUT's multiple-scattering compensation 1 / (A + B) - 1 [1e] (t19: the game's AmbientBRDF)
float dd2_ecomp(float roughness, float nov)
{
	const float2 ab = InT19.SampleLevel(SamplerS3, float2(saturate(roughness), saturate(nov)) * 0.96875 + 0.015625, 0.0).xy;
	return 1.0 / max(ab.x + ab.y, 1e-3) - 1.0;
}

// ---- DD2's light rules, escape, seen light and sun mask (v71: here, for both renderers - v52-v70 in dd2.hlsl) ----
#define DD2_TMAX 1000.0         // the bounce ray's length [5a]
#define DD2_EPS 0.002           // metres (+ 0.05 % of the distance) off a hit's surface for the next rays (DD2: tMin 1e-5)

float dd2_next(inout uint state)
{
	state = pcg(state);
	return (state >> 8) / 16777216.0;
}

// Which of the game's lights a G-buffer surface takes (eval_light's stencil / character rules)
bool dd2_light_takes(LightDesc L, surface s)
{
	const uint flags = L.flags, kind = (flags >> 9) & 7u;
	if ((flags & ((s.stencil & 32u) != 0 ? 16u : 32u)) != 0 || (flags & 0x4002000u) != 0)
		return false;
	if (((flags & 0x4000u) != 0 && !s.character) || ((flags & 0x10000000u) != 0 && s.character))
		return false;
	if ((flags & 0x8000u) != 0 && (s.character || (s.stencil & 32u) == 0))
		return false;
	if ((kind == 5 && (s.stencil & 192u) == 64u) || ((flags & 0x40000u) != 0 && (s.stencil & 64u) == 0))
		return false;
	return true;
}

// A light's direction for the response (planar box lights: against their axis)
float3 dd2_light_dir(LightDesc L, float3 l)
{
	return (L.flags & 0x4Fu) == 0x41u ? -L.axis : normalize(l);
}

// v56: a path that leaves the scene - the game's baked ambient at the primary (its own occlusion of the room already in it)
// toward the open sky (sky_env) as Baked light goes to 0; only the open sky takes the game's sky visibility open_k
// (v53-v55 put t10.y on the baked ambient too: rays out through windows and gaps brought next to nothing indoors).
// v62: baked_w 0 once the path took the baked light at a hit (it enters a path once): the open sky's share alone
float3 dd2_escaped(uint2 px, float3 dir, float3 a4, float3 a5, float3 axis, float open_k, float baked_w)
{
	const float3 baked = sky_radiance(dir, a4, a5, axis) * baked_w;
	float3 sky;
	if (g_baked >= 1.0 || !sky_env(px, dir, sky))
		return baked;
	return lerp(sky * open_k, baked, g_baked);
}

// v62 (user: "some areas simply get pitch black ... increasing bounce intensity and brightness does not help", "anything
// specular or transparent/glossy does not get path traced", lamps unaffected): most of TLOU's light - its static lamps,
// windows, the light between walls - exists only as the game's baked lighting and emission, which DD2's scene has as real
// lights and surfaces. A path's hit the camera sees this frame (the depth buffer's surface within 3% there, facing the same
// way, not hair) takes the game's own light leaving that surface - its lighting result in the light buffer, over the
// lighting's factor there: baked lamps and windows, emission, its lights under their shadow maps, the sun - and ends: the
// surface's outgoing light holds all of its lighting, the rays between give the visibility (the shadows). v52-v61 lit a hit
// only by the sun, the game's lights and the emitters: indoors next to nothing, reflections black
bool dd2_seen_light(float3 hit, float3 n, out float3 radiance)
{
	radiance = 0.0;
	const float4 clip = mul(float4(hit, 1.0), load_matrix(FRAME_VP_CUR));
	if (clip.w <= 0.0)
		return false;
	const float2 ndc = clip.xy / clip.w;
	if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0)
		return false;
	const uint2 size = full_size();
	const uint2 px = min(uint2(float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * float2(size)), size - 1);
	const float d = Depth[px];
	if (d >= 1.0 || abs(view_z(d) - clip.w) >= max(0.1, clip.w * 0.03) || dot(decode_normal(GBuffer0[px]), n) < 0.5 ||
		(pixel_flags(px) & PIXEL_HAIR) != 0)
		return false;
	// (v64: without the pixel's own glow - the emitters light the paths directly, rt.hlsl emission_at)
	radiance = max(Light[px].rgb - emission_at(px), 0.0) / max(lighting_factor(px, d), 1e-4);
	return all(isfinite(radiance));
}

// v53: the game's sun shadow at a bounce hit the camera sees this frame (the depth buffer's surface within 3% there, facing
// the same way): its mask t10.x times the lighting's sun factor, the casters the scene lacks included; -1 = not on screen
float dd2_game_sun_mask(float3 hit, float3 n)
{
	const float4 clip = mul(float4(hit, 1.0), load_matrix(FRAME_VP_CUR));
	if (clip.w <= 0.0)
		return -1.0;
	const float2 ndc = clip.xy / clip.w;
	if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0)
		return -1.0;
	const uint2 size = full_size();
	const uint2 px = min(uint2(float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * float2(size)), size - 1);
	const float d = Depth[px];
	if (d >= 1.0 || abs(view_z(d) - clip.w) >= max(0.1, clip.w * 0.03) || dot(decode_normal(GBuffer0[px]), n) < 0.5)
		return -1.0;
	const float2 uv = (float2(px) + 0.5) * c[8].zw;
	return saturate(sun_factor(load_surface(px, d), GBuffer1[px], px, uv) * InT10.SampleLevel(SamplerS0, uv, 0.0).x);
}

// v71 (user: "port 3"): the light the RE9 renderer's one bounce brings back, as DD2's first bounce takes it (dd2.hlsl
// cs_dd2_path [5], thr 1): a miss - the sky (dd2_escaped); a hit the camera sees this frame - the game's own light leaving
// it (dd2_seen_light); else SHaRC's memory of the game's light there (cs_re9_history feeds it the game's light in both
// renderers since v71); else DD2's estimate at the hit, the path ending there: the baked light once (beyond Ray length the
// baked light from that direction, nearer the occluder lit by it), the sun (a ray; the game's shadow mask where the camera
// sees the hit), one of the game's lights picked uniformly (x the count; its shadow map and a ray), the lamps (emitters),
// through DD2's response with the hit's cached material (roughness never below the primary's). v44-v70: rt.hlsl hit_light -
// what the camera saw there LAST frame (this renderer's own path-traced result), SHaRC of that, the sun, one lamp and a
// continuation ray. v87: the cache is NVIDIA's SHaRC with "NVIDIA SHaRC" on (sharc.hlsli sharc_query - its gates take the
// segment just traced and the lobe that launched it: lobe_rough, 1 for the diffuse lobe), else v23-v86's own (sharc_get);
// DD2's estimate at the hit apart (re9_hit_estimate: SHaRC's Update pass takes it too, as the hit's direct light)
float3 re9_hit_estimate(uint2 px, float3 hit, float3 nh, float3 l, scene_hit hh, float3 a4, float3 a5, float3 axis, float3 post,
	bool emit_on, float rough0, inout uint rng);
float3 re9_dd2_bounce(uint2 px, float3 o, float3 l, scene_hit hh, float3 a4, float3 a5, float3 axis, float open_k,
	float3 post, bool emit_on, float rough0, float lobe_rough, inout uint rng, out float source)
{
	// v74: where the light came from, as cs_dd2_path's debug view codes it: 1 the game's light at a hit the camera sees,
	// 2 a hit off screen (the estimate below), 3 the sky, 4 SHaRC
	source = 3.0;
	if (!hh.hit)
		return dd2_escaped(px, l, a4, a5, axis, open_k, 1.0) * g_sky;
	const float3 hit = o + l * hh.t;
	float3 nh = hit_normal(hh.draw, hh.primitive);
	if (dot(nh, l) > 0.0)
		nh = -nh;                                           // toward the ray
	float3 seen;
	source = 1.0;
	if (dd2_seen_light(hit, nh, seen))
		return seen;
	float3 cached;
	source = 4.0;
	if (sharc_nv_on() ? sharc_query(hit, nh, hh.t, lobe_rough, cached) : sharc_get(hit + c[9].xyz, nh, cached))
		return cached;
	source = 2.0;
	return re9_hit_estimate(px, hit, nh, l, hh, a4, a5, axis, post, emit_on, rough0, rng);
}
// DD2's estimate of the light leaving a bounce hit toward the ray (l: the ray's direction, nh the hit's normal toward it):
// the baked light's share, the sun, the game's lights, the lamps - the hit's direct light (v87: SHaRC's Update submits it)
float3 re9_hit_estimate(uint2 px, float3 hit, float3 nh, float3 l, scene_hit hh, float3 a4, float3 a5, float3 axis, float3 post,
	bool emit_on, float rough0, inout uint rng)
{
	const float4 m = hit_material(hh.draw, hh.primitive);
	const float3 base = m.rgb;
	// v82: the triangle's metalness and dielectric F0 (the material cache's second slot, rt.hlsl hit_material_pbr) - v44-v81
	// shaded every hit as a dielectric (metalness 0, F0 0.04): metals reflected as grey plastic. The baked ambient takes the
	// diffuse albedo plus a metal's reflectance (a metal has no diffuse part, its light is specular)
	const float2 mp = hit_material_pbr(hh.draw, hh.primitive);
	const float metal = mp.x;
	const float3 fh = lerp(mp.y, base, metal);
	const float rough = max(rough0, max(0.01, m.w >= 0.0 ? m.w : 0.5));
	const float3 hv = -l;
	const float ah = max(rough * rough, 1e-4);
	const float ec = dd2_ecomp(rough, dot(nh, hv));
	const float3 from = hit + nh * (DD2_EPS + length(hit) * 0.0005);
	const float baked_share = g_baked < 1.0 && asfloat(Data.Load(SKY_A4 + 12)) >= 1.0 ? g_baked : 1.0;
	float3 lind = (hh.t > g_ray_length ? sky_radiance(l, a4, a5, axis) : (base * (1.0 - metal) + fh * metal) * (a4 + a5 * saturate(dot(nh, axis)))) *
		(g_sky * baked_share);
	const float3 sunh = pt_sun_colour(hit + c[9].xyz, false, 0u);
	const float2 xs = float2(dd2_next(rng), dd2_next(rng));
	if (any(sunh > 0.0) && dot(nh, c[6].xyz) > 0.0)
	{
		float3 dh, sh;
		dd2_response(nh, hv, c[6].xyz, base, metal, fh, ah, ec, dh, sh);   // v82: the hit's metalness and F0
		if (any(dh + sh > 0.0))
		{
			float vis = sun_visible(from, cone_direction(c[6].xyz, g_sun_cos, xs), SUN_RAY_LENGTH);
			if (vis > 0.0)                                  // (v97: the game's sun always - the Sun slider is gone)
			{
				const float mask = dd2_game_sun_mask(hit, nh);
				if (mask >= 0.0)
					vis *= mask;
			}
			lind += sunh * vis * (dh + sh);
		}
	}
	// (v86: with the game's lights in RTXDI's ReGIR cells they come with the lamps below, picked by their light at the hit -
	// not this uniform pick)
	const bool game_regir = rtxdi_game_in_regir();
	const uint count = g_lights_on != 0 && !game_regir ? min(Data.Load(LIGHT_LIST_COUNT), LIGHT_LIST_MAX) : 0u;
	const float ul = dd2_next(rng);
	const float2 xl = float2(dd2_next(rng), dd2_next(rng));
	if (count != 0)
	{
		const LightDesc L = Lights[Data.Load(LIGHT_LIST + min(uint(ul * count), count - 1u) * 4)];
		float3 color, ld;
		float dist;
		if ((L.flags & 0x80u) != 0 && (L.flags & 0x404E020u) == 0 && light_arrival(L, hit, color, ld, dist))
		{
			float3 dh, sh;
			dd2_response(nh, hv, dd2_light_dir(L, ld), base, metal, fh, max(ah, L.min_roughness * L.min_roughness), ec, dh, sh);   // v82
			if ((L.flags & 256u) == 0)
				sh = 0.0;
			if (any(dh + sh > 0.0))
			{
				const surface hs = hit_surface(hit, nh);
				const float sm = L.shadow_map != -1 ? game_shadow_map(L, hs, false) : 1.0;
				const float vis = lerp(1.0, light_visibility(hs, from, L, xl), g_light_shadows);
				lind += color * (float(count) * sm * vis) * (dh + sh);
			}
		}
	}
	if (emit_on || game_regir)
	{
		rng = pcg(rng);
		// v77: the hit's ReGIR cell (v39-v76 emitter_irradiance); v82: the hit's own material, specular included; v86: the
		// game's lights in the cell too
		lind += rtxdi_lamps_at(hit, nh, hv, base, metal, fh, rough, rng, post) / post;
	}
	return lind;
}

float3 re9_lobe_integrand(uint lobe, surface s, float3 f0, float3 as, float3 l)
{
	if (lobe == 0)
		return max(dot(s.n, l), 0.0) * 0.31830989;
	float pdf_s;
	return re9_spec_integrand(s, l, f0, as, pdf_s);
}

[numthreads(8, 8, 1)]
void cs_re9_path(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const float d = Depth[id.xy];
	if (d >= 1.0)
	{
		re9_store(RE9_RAW_D, id.xy, 0.0);                  // the sky
		re9_store(RE9_RAW_S, id.xy, 0.0);
		re9_store(RE9_DIRECT, id.xy, 0.0);
		re9_store(RE9_DIRECT_D, id.xy, 0.0);
		return;
	}
	// v46: hair is path traced too - v44-v45 left it the game's lighting, which follows the game's baked light and not this
	// renderer's: hair and eyelashes glowed in dark rooms and went black in the sun
	const bool hair = (pixel_flags(id.xy) & PIXEL_HAIR) != 0;
	surface s = load_surface(id.xy, d);
	if (hair)
		face_viewer(s);
	const uint4 g1 = GBuffer1[id.xy];
	const float2 uv = (float2(id.xy) + 0.5) * c[8].zw;
	const uint seed = pcg(id.x + pcg(id.y + pcg(g_frame + 0x9E9u)));
	// RE9 RecastPrimaryRays: the camera ray's end re-traced against the BVH - the path starts on the traced triangle, with its
	// normal (the depth buffer's surface and the stream-out triangles differ by rounding and by cut-outs)
	// (v46: not on hair - the scene's rays leave hair out, the re-traced ray would land on the scalp; a strand's depth
	// derivatives are noise: its own normal)
	float3 ng = hair ? s.n : geometric_normal(id.xy, s.p, s.n);
	if (!hair)
	{
		const float dist = length(s.p);
		const float3 dir = s.p / max(dist, 1e-6);
		const float window = RE9_RECAST + dist * 0.01;
		const float start = max(dist - max(window, RE9_RECAST_FRONT + dist * 0.05), 0.0);   // v54: further in front
		const scene_hit h = trace_scene(dir * start, dir, dist + window - start);
		if (h.hit)
		{
			s.p = dir * (start + h.t);
			const float3 nt = hit_normal(h.draw, h.primitive);
			ng = dot(nt, dir) > 0.0 ? -nt : nt;
		}
	}
	// v69 (user: "re9 also needs same mesh fixes from DD2"): the lobes' albedos from the G-buffer's normal (the composite
	// modulates by them), then DD2's v58 lift of the shading normal onto the triangle's side and toward the viewer - the normal
	// maps tilted it past the viewer or below the triangle, and the game's BRDF (the sun, ReSTIR DI's lights), the lobes and
	// the bounce hemisphere gave black patches in the textures' shapes on cloth, faces and decals (DD2's renderer: since v58)
	float3 ad, as, f0;
	re9_albedos(s, ad, as, f0);
	s.n = visible_normal(s.n, s.v, ng);
	const float3 origin = s.p + ng * (0.005 + length(s.p) * 0.001);
	// v70 (user: "enable dd2's path traced sun, i like its results, onto re9. NOT the one with the slider, currently extremely
	// broken and flickery"): DD2's sun [3b], as cs_dd2_path's - the sun's colour (Sun 0: the game's; above 0: the add-on's
	// own, pt_sun_colour), one ray toward a direction in the Sun softness cone, DD2's response (gated on its diffuse term),
	// with Sun 0 under the game's own shadow mask too (the casters the scene lacks). v45-v69: with Sun 0 the game's shadow
	// mask alone (no ray) and the game's BRDF; above 0 ReSTIR DI's sun candidate, which competed with the lights and lamps
	// for the pixel's one light sample - present one frame, gone the next (cs_di no longer draws it)
	const float alpha_dd2 = max(s.roughness * s.roughness, 1e-4);   // DD2's response terms here (the sun, v71 the game's lights)
	const float ecomp = dd2_ecomp(s.roughness, dot(s.n, s.v));
	uint rng = pcg(id.x + pcg(id.y + pcg(g_frame + 0x9E9Du)));   // v71: DD2's random stream (its lights, its bounce)
	float3 game_sun = 0.0, game_sun_d = 0.0;               // v47: + its diffuse part (the skin's blur)
	{
		const float3 sun = pt_sun_colour(s.p + c[9].xyz, s.character, s.stencil);
		if (any(sun > 0.0) && dot(ng, c[6].xyz) > 0.0)
		{
			float3 dsun, ssun;
			dd2_response(s.n, s.v, c[6].xyz, s.albedo, s.metallic, f0, alpha_dd2, ecomp, dsun, ssun);
			if (any(dsun > 0.0))
			{
				float vis = sun_visible(origin, cone_direction(c[6].xyz, g_sun_cos, random3(id.xy, g_frame + 0x5A1Fu).xy), SUN_RAY_LENGTH);
				vis *= saturate(sun_factor(s, g1, id.xy, uv) * InT10.SampleLevel(SamplerS0, uv, 0.0).x);   // (v97: always)
				const float3 sun_d = sun * (vis * dsun), sun_s = sun * (vis * ssun);
				if (all(isfinite(sun_d)) && all(isfinite(sun_s)))   // v73: a value that is not a finite number is left out
				{
					game_sun_d = max(sun_d, 0.0);
					game_sun = game_sun_d + max(sun_s, 0.0);
				}
			}
		}
	}
	// v71 (user: "port 4"): DD2's direct light from the game's lights [3c], as cs_dd2_path's - every light of the frame (the
	// lights the tiles name: cs_light_census' list), each with its own shadow ray toward a point on its sphere, times its
	// shadow map, through DD2's response; diffuse and specular apart (the skin's blur takes the diffuse). v42-v70: ReSTIR DI
	// drew one of the tile's lights (or a lamp) per pixel - the highlights came and went. ReSTIR DI keeps the lamps (emitters).
	// v86 (user: "go for rtxdi for sky and game lights"): with "NVIDIA RTXDI game lights" (g_pass bit 4) none here - they come
	// with the lamps from NVIDIA's ReSTIR DI (rtxdi_di.hlsl cs_rtxdi_di: RE9_RTXDI, its specular apart)
	float3 lights_d = 0.0, lights_s = 0.0;
	{
		const uint count = g_lights_on != 0 && (g_pass & 16u) == 0u ? min(Data.Load(LIGHT_LIST_COUNT), LIGHT_LIST_MAX) : 0u;
		[loop]
		for (uint i = 0; i < count; ++i)
		{
			const LightDesc L = Lights[Data.Load(LIGHT_LIST + i * 4)];
			const float2 xl = float2(dd2_next(rng), dd2_next(rng));
			if (!dd2_light_takes(L, s))
				continue;
			float3 color, l;
			float dist;
			if (!light_arrival(L, s.p, color, l, dist))
				continue;
			float3 dl, sl;
			dd2_response(s.n, s.v, dd2_light_dir(L, l), s.albedo, s.metallic, f0, max(alpha_dd2, L.min_roughness * L.min_roughness), ecomp, dl, sl);
			if ((L.flags & 128u) == 0)
				dl = 0.0;
			if ((L.flags & 256u) == 0)
				sl = 0.0;
			if (!any(dl > 0.0) && !any(sl > 0.0))
				continue;
			const float sh = game_light_shadow(L, s);
			if (!(sh > 0.0))
				continue;
			const float vis = (L.shadow_map != -1 && s.character) ? 1.0 : lerp(1.0, light_visibility(s, origin, L, xl), g_light_shadows);
			// v73: a light whose terms are not finite numbers is left out - v71-v72 summed it into the pixel's diffuse direct
			// light (RE9_DIRECT_D), which the composite adds unclamped: the whole pixel NaN, max() made it black (the single-
			// frame black splotches, 0.22 % of the surface pixels in the 2026-10-13 capture)
			const float3 add_d = color * (sh * vis) * dl, add_s = color * (sh * vis) * sl;
			if (all(isfinite(add_d)) && all(isfinite(add_s)))
			{
				lights_d += max(add_d, 0.0);
				lights_s += max(add_s, 0.0);
			}
		}
	}
	// the lamps (emitters): ReSTIR DI - f V W (v71: the lamps only; the sun v70 and the game's lights v71 are DD2's above)
	const float3 post = max(lighting_factor(id.xy, d) * g_brightness, 1e-4);
	// v45: the contact ray (rt.hlsl cs_full v40-v41): an occluder within CONTACT_RANGE darkens the indirect light - the scene
	// lacks what the game culled and the small detail, which RE9's complete scene has (it needs no occlusion term)
	// v58: a direction below the triangle is mirrored above it (v45-v57: it counted as fully blocked - the indirect light gone
	// that frame wherever a normal map tilts the shading normal: flickering black patches on cloth and decals)
	float contact = 1.0;
	if (g_contact > 0.0)
	{
		float3 cdir = cosine_direction(s.n, random3(id.xy, g_frame + 0xC0DEu).xy);
		const float below = dot(cdir, ng);
		if (below <= 0.0)
			cdir = normalize(cdir - ng * (2.0 * below - 0.01));
		const scene_hit ch = trace_scene(origin, cdir, CONTACT_RANGE);
		if (ch.hit)
		{
			const float close = 1.0 - saturate(ch.t / CONTACT_RANGE);
			contact = 1.0 - g_contact * close * close;
		}
	}
	// v47: the diffuse part apart, with the material's subsurface amount (the skin's blur)
	// v76 (user: "try using option A, use NVIDIAS library"): with "NVIDIA RTXDI lamps" (g_pass bit 2) the lamps' light is
	// RTXDI's ReSTIR DI (rtxdi_di.hlsl, run before this pass), diffuse as di_shade's. v86: its specular part apart (RE9_RTXDI .w,
	// packed): the skin's blur and RR's diffuse take the diffuse alone; with g_pass bit 4 the game's lights are in it
	float3 di_d;
	float3 di;
	if ((g_pass & 4u) != 0u)
	{
		const float4 rtxdi_light = re9_load(RE9_RTXDI, id.xy);
		di_d = rtxdi_light.rgb;
		di = rtxdi_light.rgb + unpack_r11g11b10(asuint(rtxdi_light.w));
	}
	else
		di = di_shade(id.xy, s, d, post, di_d);
	if (!all(isfinite(di)) || !all(isfinite(di_d)))         // v73: as the lights above
	{
		di = 0.0;
		di_d = 0.0;
	}
	const float3 direct_d = max(di_d + game_sun_d + lights_d, 0.0);
	re9_store(RE9_DIRECT, id.xy, float4(max(di + game_sun + lights_d + lights_s - direct_d, 0.0), contact));
	// v73 (user: "along ellie's jacket permanent black blotch, they go away when turning DD2"; the 2026-10-13 capture): no
	// subsurface amount - G-buffer 1 .x's low byte, which v47-v72 took for the skin's amount (rt.hlsl subsurface_weight), is
	// 255 on every character's clothes (and the bar's floor) and 0 on skin and hair. The blur ran at full weight on clothes,
	// never on skin, and smeared their noisy lamp light into blocks along its two axes (the reddish blotches with stepped
	// edges on the jacket). The composite's blur stays for a skin amount once one is found
	re9_store(RE9_DIRECT_D, id.xy, float4(direct_d, 0.0));
	// v83: "NVIDIA ReSTIR PT" (g_pass bit 3) - the bounce is ReSTIR PT's (rtxdi_pt.hlsl: the same path with RTXDI's initial
	// sampling and temporal reuse, its cs_pt_shade wrote the raw planes and the path reuse's sample before this pass); here the
	// direct light, the debug views' terms and the path reuse's origin. v84: its temporal and spatial reuse replace the path
	// reuse (tlou_rt.cpp leaves bit 1 off with it: no origin kept, cs_pt_shade writes no path reuse sample)
	if ((g_pass & 8u) != 0u)
	{
		re9_store(RE9_SSS_E, id.xy, float4(di, 0.0));
		re9_store(RE9_SSS_H, id.xy, float4(game_sun + lights_d + lights_s, 0.0));
		if ((g_pass & 2u) != 0u)
			reuse_origin_store(id.xy, origin, ng);
		return;
	}
	// one BSDF sample, the lobe picked by the lobes' albedos (RE9: luma(Ad) / luma(Ad + As)), kept away from 0 and 1
	// v70 (user: "dd2 has actual better material handling (specular materials, PBR) than RE9, looks like re9 doesnt interact
	// with any materials"): on a dielectric that gave the specular lobe 10-15 % of the frames (luma(As) ~0.08 against luma(Ad)
	// ~0.5), and the accumulation keeps 2 frames of a smooth surface's reflection (cs_re9_accum) - reflections were absent
	// most frames and flickered. A smooth surface takes it at least half the time (DD2: always half); rough ones (0.6 up)
	// keep RE9's floor, their specular being nearly diffuse
	const float lad = restir_lum(ad), las = restir_lum(as);
	const float spec_floor = lerp(0.5, 0.1, saturate((s.roughness - 0.2) * 2.5));
	const float p_spec = lad + las > 0.0 ? clamp(las / (lad + las), spec_floor, 0.9) : 0.5;
	const float2 xi = random3(id.xy, g_frame + 0x1E9u).xy;
	const bool spec = random1(seed + 0x5E1u) < p_spec;
	const float alpha = max(s.roughness * s.roughness, 0.002);
	const float3 dir = spec ? reflect(-s.v, sample_ggx_vndf(s.v, s.n, alpha, xi)) : cosine_direction(s.n, xi);
	// v48: one-sample MIS over the two lobes, as RE9 (f_lobe / lerp(pdfSpec, pdfDiff, pDiffuse)): either lobe's direction
	// feeds both outputs, over the mixture's pdf (v44-v47: only the picked lobe's, over its probability). The sample's
	// demodulated weights f cos / pdf: diffuse cos / pi / pdf, specular F D G1(v) G1(l) / (4 N.V) / As / pdf
	const float cos_l = dot(s.n, dir);
	float pdf_s;
	const float3 fs = re9_spec_integrand(s, dir, f0, as, pdf_s);
	const float pdf = (1.0 - p_spec) * max(cos_l, 0.0) * 0.31830989 + p_spec * pdf_s;
	float3 wd = 0.0, ws = 0.0;
	if (cos_l > 0.0 && pdf > 0.0)
	{
		wd = cos_l * 0.31830989 / pdf;
		ws = fs / pdf;
	}
	float3 L = 0.0;
	float hit_t = 0.0;
	bool traced = false;
	bool first_hit = false;                                 // v68: the path's hit and its normal (the reuse's sample)
	float3 hit_n = 0.0;
	float source = 0.0;                                     // v74: the debug view's first-bounce source (0: not traced)
	// v72 (user: "Before the integration of all DD2 stuff onto RE9, DD2 actually fixed those dark patches on ellies jacket"):
	// a direction below the triangle is traced too, as DD2's loop traces it (it checks the shading normal alone) - the ray
	// meets the surface itself and takes the game's light leaving it (dd2_seen_light; else SHaRC, DD2's estimate). v44-v71
	// left it untraced: no light, yet a sample - where a normal map tilts the shading normal (cloth folds, wet and dirt
	// decals) up to about half of them: dark patches in the textures' shapes. (Path reuse keeps such a sample whole: its hit
	// is too near for the partners' shifts, so the pixel's own MIS weight stays 1)
	if (any(wd + ws > 0.0))
	{
		traced = true;
		const float3 a4 = InT4[id.xy].rgb, a5 = InT5[id.xy].rgb;
		const float3 axis = decode_ambient_direction(GBuffer1[id.xy]);
		// v71 (user: "port 3"): DD2's first bounce (re9_dd2_bounce) on DD2's ray length - the game's light at a hit the camera
		// sees, SHaRC's memory of it, else DD2's estimate there; a miss: DD2's sky (the game's sky visibility on the open sky's
		// share only). v44-v70: rt.hlsl hit_light within Ray length (what the camera saw last frame: this renderer's own
		// result), escaped_light x t10.y on every miss. v69: the cut-out cards block as DD2's rays (v44-v68 went through)
		const scene_hit h = trace_cutouts(origin, dir, DD2_TMAX);
		const float open_k = 1.0;                           // v97: no game sky visibility (t10.y) - the Occlusion slider is gone
		const bool emit_on = g_emit > 0.0 && Data.Load(emit_list_offset() + 8) != 0u;
		L = re9_dd2_bounce(id.xy, origin, dir, h, a4, a5, axis, open_k, post, emit_on, s.roughness, spec ? s.roughness : 1.0, rng, source);
		hit_t = h.t;
		if (h.hit)
		{
			first_hit = true;
			const float3 nh = hit_normal(h.draw, h.primitive);
			hit_n = dot(nh, dir) > 0.0 ? -nh : nh;
		}
	}
	float3 sd = L * wd, ss = L * ws;
	// v70: the clamp on what the sample adds to the pixel (both lobes times their albedos) against RE9_MAX times the albedos'
	// luminance (re9_clamp_limit) - v44-v69 on the demodulated sum, whose specular part is divided by its albedo (~0.04-0.1 on
	// a dielectric) and weighs 1 / p_spec: the reflections of anything bright (lamps, windows, the sky) were cut to a fraction
	const float l = restir_lum(ad * sd + as * ss), lmax = re9_clamp_limit(ad, as);
	float kmax = 1.0;                                       // v68: the clamp's factor, for the reuse's sample too
	if (!(l <= lmax))
		kmax = l > lmax ? lmax / l : 0.0;
	sd *= kmax;
	ss *= kmax;
	re9_store(RE9_RAW_D, id.xy, float4(sd, spec ? 0.0 : 1.0));
	re9_store(RE9_RAW_S, id.xy, float4(ss, hit_t));
	// v74: the debug views' terms (cs_re9_composite, Debug view 3-5) in the planes the skin blur used (v73: no blur - its
	// passes leave pixels without a subsurface amount alone): the lamps' light (ReSTIR DI); the game's lights and the sun,
	// and where the first bounce took its light
	re9_store(RE9_SSS_E, id.xy, float4(di, 0.0));
	re9_store(RE9_SSS_H, id.xy, float4(game_sun + lights_d + lights_s, source));
	// v68: Path reuse (g_pass bit 1) - this frame's path as reuse.hlsl's sample: its hit (or the sky's direction), the light
	// it brings (clamped as the raw sample), W = 1 / the mixture's pdf; and the ray's origin here. (v48-v67: the candidate
	// of RE9's ReSTIR GI reservoirs, which reuse.hlsl replaces)
	if ((g_pass & 2u) != 0u)
	{
		reuse_sample r = (reuse_sample)0;
		if (traced && pdf > 0.0 && kmax > 0.0)
		{
			r.kind = first_hit ? REUSE_HIT : REUSE_SKY;
			r.x = first_hit ? origin + dir * hit_t + c[9].xyz : dir;
			r.n = hit_n;
			r.L = L * kmax;
			r.W = 1.0 / pdf;
		}
		reuse_store(id.xy, r);
		reuse_origin_store(id.xy, origin, ng);
	}
}

// The stand-in denoiser until Ray Reconstruction: per lobe, the reprojected history (the motion plane; depth 5% and normal
// 0.8 as cs_temporal) blended with this frame's sample - diffuse up to 32 frames, specular fewer on smooth surfaces (the
// reflected image moves with the view)
[numthreads(8, 8, 1)]
void cs_re9_accum(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint cur = g_frame & 1u, prev = cur ^ 1u;
	const float d = Depth[id.xy];
	if (d >= 1.0)
	{
		re9_store(RE9_ACC_D + cur, id.xy, 0.0);
		re9_store(RE9_ACC_S + cur, id.xy, 0.0);
		re9_store(RE9_GEO + cur, id.xy, 0.0);
		re9_store(RE9_SSS_E, id.xy, 0.0);
		return;
	}
	const float3 n = decode_normal(GBuffer0[id.xy]);
	const float3 p = camera_relative_position(float2(id.xy), d);
	// v46: hair (dithered strands, their normals all directions) takes any history at its depth and leaves a zero normal, which
	// no other surface takes
	const bool hair = (pixel_flags(id.xy) & PIXEL_HAIR) != 0;
	const float4 raw_d = re9_load(RE9_RAW_D, id.xy), raw_s = re9_load(RE9_RAW_S, id.xy);
	// v72: the contact ray's visibility (RE9_DIRECT.w - one ray per pixel and frame, both renderers) darkens this frame's sample
	// before it joins the history, and averages out with it. v45-v71 multiplied the converged light by it in the composites,
	// drawn anew every frame: with Contact shadows at 1 a 20-100 % darkening wherever geometry lies within 0.5 m (cloth folds,
	// faces, corners) - black dots changing every frame. (v97: in full - Contact shadows sets it; v49-v96 also lerp(1, contact,
	// Occlusion), and the game's own occlusion by lerp(1, AO, Occlusion) in the composite, both gone with that slider)
	const float contact = saturate(re9_load(RE9_DIRECT, id.xy).w);
	const float3 sample_d = raw_d.rgb * contact, sample_s = raw_s.rgb * contact;
	float4 hd = 0.0, hs = 0.0;
	float weight = 0.0;
	const float3 clip = previous_clip(id.xy, p);
	if (clip.z > 0.0 && Data.Load4(FRAME_INFO).x > 1)
	{
		const float2 fpos = float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float2(size) - 0.5;
		const int2 base = int2(floor(fpos));
		const float2 f = fpos - base;
		[unroll]
		for (int k = 0; k < 4; ++k)
		{
			const int2 q = base + int2(k & 1, k >> 1);
			if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y))
				continue;
			const float4 geo = re9_load(RE9_GEO + prev, uint2(q));
			if (!all(isfinite(geo)) || !(geo.x > 0.0) || abs(geo.x - clip.z) > 0.05 * clip.z + 0.05 || (!hair && dot(geo.yzw, n) < 0.8))
				continue;                                   // another surface there last frame (or never written)
			const float4 qd = re9_load(RE9_ACC_D + prev, uint2(q)), qs = re9_load(RE9_ACC_S + prev, uint2(q));
			if (!all(isfinite(qd)) || !all(isfinite(qs)))
				continue;
			const float w = ((k & 1) ? f.x : 1.0 - f.x) * ((k >> 1) ? f.y : 1.0 - f.y);
			hd += qd * w;
			hs += qs * w;
			weight += w;
		}
	}
	float4 od = float4(sample_d, 1.0), os = float4(sample_s, 1.0);
	if (weight > 0.01)
	{
		hd /= weight;
		hs /= weight;
		const float roughness = (GBuffer0[id.xy].w & 0xFFu) / 255.0;
		const float nd = min(hd.w + 1.0, RE9_HISTORY_D);
		const float ns = min(hs.w + 1.0, lerp(2.0, RE9_HISTORY_S, saturate(roughness * 4.0)));
		od = float4(lerp(hd.rgb, sample_d, 1.0 / nd), nd);
		os = float4(lerp(hs.rgb, sample_s, 1.0 / ns), ns);
	}
	// v86 (user: "i see no changes with [ReSTIR PT] on and off"): the status line's bounce noise - where the history has settled
	// (8 frames and more), this frame's sample against the history's light, relative (FRAME_PT + 40: the sum x 1000, + 44: the
	// pixels). It is what ReSTIR PT and the path reuse lower; the image does not show it once RR or this accumulation has
	// averaged the samples (both estimates converge to the same light)
	uint2 noise = 0u;
	if (weight > 0.01 && hd.w >= 8.0)
	{
		const float m = restir_lum(hd.rgb + hs.rgb), x = restir_lum(sample_d + sample_s);
		if (m > 0.002 && isfinite(x))
			noise = uint2(uint(min(abs(x - m) / m, 4.0) * 1000.0 + 0.5), 1u);
	}
	noise = WaveActiveSum(noise);
	if (WaveIsFirstLane() && noise.y != 0u)
	{
		Data.InterlockedAdd(FRAME_PT + 40, noise.x);
		Data.InterlockedAdd(FRAME_PT + 44, noise.y);
	}
	re9_store(RE9_ACC_D + cur, id.xy, od);
	re9_store(RE9_ACC_S + cur, id.xy, os);
	re9_store(RE9_GEO + cur, id.xy, float4(view_z(d), hair ? 0.0 : n));
	// v47: the skin's diffuse light for the blur, as the composite puts it together: the direct light's diffuse part
	// demodulated by the diffuse albedo, plus the indirect (occluded, by Bounce)
	// v74: only where there is a subsurface amount (none since v73) - elsewhere the plane keeps the path passes' debug terms
	const float4 dd = re9_load(RE9_DIRECT_D, id.xy);
	if (dd.w > 0.0)
	{
		const uint4 g1 = GBuffer1[id.xy];
		const float metallic = (g1.y & 0x2000u) != 0 ? ((g1.y >> 8) & 31u) / 31.0 : 0.0;
		const float3 ad = decode_albedo(GBuffer0[id.xy]) * (1.0 - metallic);
		const float2 uv = (float2(id.xy) + 0.5) * c[8].zw;
		re9_store(RE9_SSS_E, id.xy, float4(max(dd.rgb, 0.0) / max(ad, 1e-3) + max(od.rgb, 0.0) * g_bounce, dd.w));   // (v97: no game AO)
	}
}

// v47: the skin blur's horizontal pass (the composite makes the vertical one)
[numthreads(8, 8, 1)]
void cs_re9_sss(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const float4 e = re9_load(RE9_SSS_E, id.xy);
	const float d = Depth[id.xy];
	if (d < 1.0 && e.w > 0.0)                               // v74: other pixels keep the path pass's debug terms
		re9_store(RE9_SSS_H, id.xy, re9_sss_blur(RE9_SSS_E, id.xy, d, int2(1, 0)));
}

// ---- v78: DLSS Ray Reconstruction (in the helper process: tlou_rt.cpp, src\rr_shared.h) ----
// The frame's RR block in its draw table (tlou_rt.cpp k_table_rr): x the flags (RR_USE_OUTPUT: the composite takes RR's light
// of the frame before - the add-on clears it when the list is submitted and the game's queue could not wait for that light),
// y the slot this frame's inputs go to, z the slot whose output is read, w the RR region's offset in the data buffer; then the
// 8-byte rows' pitch, the 4-byte rows' pitch, one plane's bytes (tlou_rt.cpp data_layout rr*). The region: the slots' frame
// constants (rr_gpu_frame, RR_CONSTANT_BYTES each), then from RR_PLANES_AT one plane per rr_kind (rr_shared.h) - this frame's
// inputs, which the add-on copies into the slot's shared textures, and RR's output of the frame before, copied back from it
#define TABLE_RR 804608         // tlou_rt.cpp k_table_rr
#define RR_USE_OUTPUT 1u
#define RR_NOISY 2u             // v81: RR upscales in the game this frame and denoises: the composites write this frame's noisy light
                                //   (rr_noisy_light), their accumulated light goes to RE9_SETTLED for the history
#define RR_CONSTANT_BYTES 256   // TLOU_RR_CONSTANT_BYTES
#define RR_PLANES_AT 65536
#define RR_COLOR 0              // rr_kind: RGBA16F
#define RR_DIFFUSE 1            //   RGBA8
#define RR_SPECULAR 2           //   RGBA16F
#define RR_NORMALS 3            //   RGBA16F (roughness in w)
#define RR_DEPTH 4              //   R32F
#define RR_MOTION 5             //   RG16F
#define RR_SPEC_HIT 6           //   R32F
#define RR_OUTPUT 7             //   RGBA16F
#define RR_COLOR_MAX 65000.0    // within half floats

struct rr_block
{
	uint flags, in_slot, out_slot, at, pitch8, pitch4, plane;
};
rr_block rr_load_block()
{
	const uint4 b = Table.Load4(TABLE_RR), q = Table.Load4(TABLE_RR + 16);
	rr_block r;
	r.flags = b.x;
	r.in_slot = b.y;
	r.out_slot = b.z;
	r.at = b.w;
	r.pitch8 = q.x;
	r.pitch4 = q.y;
	r.plane = q.z;
	return r;
}
uint rr_texel(rr_block r, uint plane, uint2 px, bool wide)
{
	return r.at + RR_PLANES_AT + plane * r.plane + px.y * (wide ? r.pitch8 : r.pitch4) + px.x * (wide ? 8u : 4u);
}
uint2 rr_half4(float4 v)
{
	return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
}
float3 rr_unhalf3(uint2 v)
{
	return float3(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y));
}
uint rr_unorm4(float4 v)
{
	const uint4 u = uint4(saturate(v) * 255.0 + 0.5);
	return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24);
}

// The frame's constants for RR (rr_shared.h rr_gpu_frame). The jitter from the game's matrix: its camera-relative
// view-projection (rows, clip = p x M) is the camera's rotation (right, up, forward) times a projection whose jitter sits in
// its third row (clip.x = sx x + jx z) - column 0 is sx right + jx forward and column 3 the forward axis, so jx = column 0 .
// column 3 / |column 3|^2 (the same for y) - in render pixels, x right and y down (NDC y up), the image's shift by the
// jitter (as NVIDIA's samples give it to DLSS). A reset at a cut: the first frames, the camera 4 m away or turned by more
// than 45 degrees since the frame before
void rr_store_constants(uint at, uint2 size)
{
	const float4x4 vp = load_matrix(FRAME_VP_CUR), vq = load_matrix(FRAME_VP_PREV);
	const float3 c0 = float3(vp[0].x, vp[1].x, vp[2].x), c1 = float3(vp[0].y, vp[1].y, vp[2].y), c3 = float3(vp[0].w, vp[1].w, vp[2].w);
	const float3 c3p = float3(vq[0].w, vq[1].w, vq[2].w);
	const float ff = max(dot(c3, c3), 1e-20);
	float2 jitter = float2(dot(c0, c3) / ff * 0.5 * float(size.x), -dot(c1, c3) / ff * 0.5 * float(size.y));
	const bool finite = all(isfinite(jitter));
	if (!finite)
		jitter = 0.0;
	const float3 cam = asfloat(Data.Load4(FRAME_CAM_CUR)).xyz, cam_prev = asfloat(Data.Load4(FRAME_CAM_PREV)).xyz;
	const uint frames = Data.Load4(FRAME_INFO).x;
	const bool turned = dot(c3, c3p) < 0.7071 * sqrt(ff * max(dot(c3p, c3p), 1e-20));
	const bool reset = frames <= 2u || length(cam - cam_prev) > 4.0 || turned || !finite;
	Data.Store4(at, uint4(asuint(jitter), asuint(1.0), asuint(1.0)));   // the jitter, the motion's scale (pixels)
	Data.Store4(at + 16, uint4(reset ? 1u : 0u, asuint(1.0), frames, 0u));   // reset, pre-exposure, the frame
	[unroll]
	for (uint k = 0; k < 8; ++k)
		Data.Store4(at + 32 + k * 16, 0u);                  // the matrices: not given yet (v79, with the hit distance)
}

// The light of a pixel from this frame's samples alone, as the composites put it together from the accumulated ones: the
// direct light, Ad x the diffuse sample x Bounce + As x the specular sample (RE9_RAW_D / RE9_RAW_S, what cs_re9_accum
// accumulates) under the contact ray and the game's occlusion, before the lighting's factor - RR's noisy input (the helper's
// colour; v81 the light buffer itself when RR denoises in the game)
// v86: its indirect part alone - this frame's bounce from its samples, before the accumulation and RR: the Debug view
// "Bounce this frame" (where ReSTIR PT's and the path reuse's lower noise shows; the image takes it through RR or the
// accumulation, which both converge to the same light)
float3 re9_bounce_now(uint2 px, float3 ad, float3 as)
{
	const float4 raw_d = re9_load(RE9_RAW_D, px), raw_s = re9_load(RE9_RAW_S, px);
	const float contact = saturate(re9_load(RE9_DIRECT, px).w);   // (v97: in full, no game AO)
	return (ad * max(raw_d.rgb, 0.0) * g_bounce + as * max(raw_s.rgb, 0.0)) * contact;
}
float3 rr_noisy_light(uint2 px, float3 ad, float3 as)
{
	const float4 direct = re9_load(RE9_DIRECT, px), direct_d = re9_load(RE9_DIRECT_D, px);
	return max(direct.rgb, 0.0) + max(direct_d.rgb, 0.0) + re9_bounce_now(px, ad, as);
}

// RR's inputs for this frame (tlou_rt.cpp copies the planes into the slot's shared textures): the noisy light as the
// composites put it together from this frame's samples alone - the direct light, Ad x the diffuse sample x Bounce + As x the
// specular sample (RE9_RAW_D / RE9_RAW_S, what cs_re9_accum accumulates) under the contact ray and the game's occlusion,
// before the lighting's factor (the composite applies the frame's own to RR's result) - the lobes' albedos, the normal with
// the roughness (RR's packed mode), the depth buffer's value, the motion in pixels to where the surface was in the frame
// before's jittered pixels (cs_motion: the triangle's own motion; RR takes it as MVJittered) and the specular hit distance;
// the sky: the game's light, no albedo, the camera's turn. Thread (0, 0) writes the frame's constants into the slot
[numthreads(8, 8, 1)]
void cs_rr_inputs(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	const rr_block r = rr_load_block();
	if (all(id.xy == 0u))
		rr_store_constants(r.at + r.in_slot * RR_CONSTANT_BYTES, size);
	if (id.x >= size.x || id.y >= size.y)
		return;
	const float d = Depth[id.xy];
	float3 color = 0.0, ad = 0.0, as = 0.0, n = float3(0.0, 0.0, 1.0);
	float roughness = 1.0, hit = 0.0;
	float3 clip = 0.0;
	if (d < 1.0)
	{
		surface s = load_surface(id.xy, d);
		if ((pixel_flags(id.xy) & PIXEL_HAIR) != 0)
			face_viewer(s);
		float3 f0;
		re9_albedos(s, ad, as, f0);
		color = rr_noisy_light(id.xy, ad, as);
		n = s.n;
		roughness = saturate(s.roughness);
		hit = max(re9_load(RE9_RAW_S, id.xy).w, 0.0);
		clip = previous_clip(id.xy, camera_relative_position(float2(id.xy), d));
	}
	else
	{
		color = Light[id.xy].rgb;                           // the game's sky
		const float3 p = camera_relative_position(float2(id.xy), 0.99999);
		n = -normalize(p);
		const float4 cp = mul(float4(p, 1.0), load_matrix(FRAME_VP_PREV));
		clip = cp.w > 0.0 ? float3(cp.xy / cp.w, cp.w) : float3(0.0, 0.0, 0.0);
	}
	float2 motion = 0.0;
	if (clip.z > 0.0)
		motion = float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float2(size) - (float2(id.xy) + 0.5);
	if (!all(isfinite(color)))
		color = 0.0;
	if (!all(isfinite(motion)))
		motion = 0.0;
	if (!all(isfinite(n)))
		n = float3(0.0, 0.0, 1.0);
	Data.Store2(rr_texel(r, RR_COLOR, id.xy, true), rr_half4(float4(min(color, RR_COLOR_MAX), 1.0)));
	Data.Store(rr_texel(r, RR_DIFFUSE, id.xy, false), rr_unorm4(float4(ad, 1.0)));
	Data.Store2(rr_texel(r, RR_SPECULAR, id.xy, true), rr_half4(float4(saturate(as), 1.0)));
	Data.Store2(rr_texel(r, RR_NORMALS, id.xy, true), rr_half4(float4(n, roughness)));
	Data.Store(rr_texel(r, RR_DEPTH, id.xy, false), asuint(d));
	Data.Store(rr_texel(r, RR_MOTION, id.xy, false), f32tof16(motion.x) | (f32tof16(motion.y) << 16));
	Data.Store(rr_texel(r, RR_SPEC_HIT, id.xy, false), asuint(isfinite(hit) ? hit : 0.0));
}

// RR's light of the frame before (its output: unjittered) where this pixel's surface was then - the motion plane gives the
// position in that frame's jittered pixels, its jitter (the read slot's constants) the position in RR's - Catmull-Rom over
// RR's 4 x 4 pixels around it, held within the 2 x 2 nearest (no rings). Only where the 2 x 2 pixels around that position
// held this surface in the frame before (cs_re9_accum's test against its other parity's RE9_GEO: view depth 5 %, normal
// 0.8, hair any normal) - else false: the frame before did not see it, the accumulation's light stays
bool rr_reprojected(uint2 px, float d, float3 n, bool hair, out float3 light)
{
	light = 0.0;
	const rr_block r = rr_load_block();
	if ((r.flags & RR_USE_OUTPUT) == 0u)
		return false;
	const uint2 size = full_size();
	const float3 clip = previous_clip(px, camera_relative_position(float2(px), d));
	if (!(clip.z > 0.0))
		return false;
	const float2 pos = float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float2(size);
	const uint prev = (g_frame & 1u) ^ 1u;
	const int2 base = int2(floor(pos - 0.5));
	[unroll]
	for (int k = 0; k < 4; ++k)
	{
		const int2 q = base + int2(k & 1, k >> 1);
		if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y))
			return false;
		const float4 geo = re9_load(RE9_GEO + prev, uint2(q));
		if (!all(isfinite(geo)) || !(geo.x > 0.0) || abs(geo.x - clip.z) > 0.05 * clip.z + 0.05 || (!hair && dot(geo.yzw, n) < 0.8))
			return false;
	}
	const float2 jitter = asfloat(Data.Load2(r.at + r.out_slot * RR_CONSTANT_BYTES));
	const float2 f = pos - jitter - 0.5;
	const int2 b = int2(floor(f));
	const float2 t = f - float2(b);
	const float2 w0 = t * (-0.5 + t * (1.0 - 0.5 * t)), w1 = 1.0 + t * t * (-2.5 + 1.5 * t);
	const float2 w2 = t * (0.5 + t * (2.0 - 1.5 * t)), w3 = t * t * (-0.5 + 0.5 * t);
	const float wx[4] = { w0.x, w1.x, w2.x, w3.x }, wy[4] = { w0.y, w1.y, w2.y, w3.y };
	float3 sum = 0.0, lo = 1e30, hi = -1e30;
	[unroll]
	for (int j = 0; j < 4; ++j)
		[unroll]
		for (int i = 0; i < 4; ++i)
		{
			const int2 q = clamp(b + int2(i - 1, j - 1), int2(0, 0), int2(size) - 1);
			const float3 v = rr_unhalf3(Data.Load2(rr_texel(r, RR_OUTPUT, uint2(q), true)));
			sum += v * (wx[i] * wy[j]);
			if ((i == 1 || i == 2) && (j == 1 || j == 2))
			{
				lo = min(lo, v);
				hi = max(hi, v);
			}
		}
	light = clamp(sum, lo, hi);
	return all(isfinite(light));
}
// The light buffer: direct + Ad x diffuse x Bounce + As x specular (before the lighting's factor, as cs_full), the skin's
// diffuse light through the subsurface kernel (v47), the emitters' own light; blended over the game's result by Path tracing. The
// Bounce slider's extra over one physical bounce goes to RE9_BOOST, which cs_history takes out of the history (the hits
// see last frame's light: an amplified bounce would grow the loop)
[numthreads(8, 8, 1)]
void cs_re9_composite(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const float d = Depth[id.xy];
	if (d >= 1.0)
	{
		re9_store(RE9_BOOST, id.xy, 0.0);
		return;                                             // the sky keeps the game's result (v46: hair no longer)
	}
	const bool hair = (pixel_flags(id.xy) & PIXEL_HAIR) != 0;
	const uint cur = g_frame & 1u;
	surface s = load_surface(id.xy, d);
	if (hair)
		face_viewer(s);
	float3 ad, as, f0;
	re9_albedos(s, ad, as, f0);
	const float2 uv = (float2(id.xy) + 0.5) * c[8].zw;
	const float4 direct = re9_load(RE9_DIRECT, id.xy);
	// v45: the game's small-scale occlusion (SSAO, capsules, the material's cavity) and the contact ray on the indirect light
	// v49: by the Occlusion slider (v44-v48 it reached only the escaped sky's game visibility: no visible effect here)
	// v72: the contact ray (direct.w) is in the accumulated light now (cs_re9_accum); v97: the game's occlusion is gone
	const float3 diffuse = ad * max(re9_load(RE9_ACC_D + cur, id.xy).rgb, 0.0);
	const float3 specular = as * max(re9_load(RE9_ACC_S + cur, id.xy).rgb, 0.0);
	// v47: the direct light's diffuse part apart (RE9_DIRECT holds the rest); the skin's diffuse light, direct and indirect,
	// through the subsurface kernel by the material's amount (the vertical pass here, cs_re9_sss the horizontal) - in place
	// of the game's subsurface light (t7: CS_SkinDiffuse after SSSS, lit by the game's baked light and shadow maps), which
	// v44-v46 added on top of this renderer's full diffuse light: the skin's diffuse twice, half of it the game's
	const float4 direct_d = re9_load(RE9_DIRECT_D, id.xy);
	float3 total = direct.rgb + direct_d.rgb + diffuse * g_bounce + specular;
	if (direct_d.w > 0.0)
		total += ad * (re9_sss_blur(RE9_SSS_H, id.xy, d, int2(0, 1)).rgb - re9_load(RE9_SSS_E, id.xy).rgb) * direct_d.w;
	// v81: RR denoising in the game (RR_NOISY, set only without a debug view) - this frame's noisy light into the light buffer
	// (RR's colour at the game's DLSS), the accumulated light kept for the history (RE9_SETTLED)
	const bool noisy = g_pass == 0 && (rr_load_block().flags & RR_NOISY) != 0u;
	const float3 settled = total;
	// v78: Ray Reconstruction's light of the frame before, reprojected (rr_reprojected; the add-on flags it only without a
	// debug view), in place of the accumulation's where the frame before saw this surface
	float3 rr_light;
	if (noisy)
		total = rr_noisy_light(id.xy, ad, as);
	else if (g_pass == 0 && rr_reprojected(id.xy, d, decode_normal(GBuffer0[id.xy]), hair, rr_light))
		total = rr_light;
	// v74: the Debug view (g_pass; the DD2 composite has the same): 1 the direct light alone, 2 the indirect light alone, 3
	// where each pixel's first bounce took its light (green the game's light at a hit the camera sees, red a hit off screen,
	// blue the sky, yellow SHaRC, grey not traced), 4 the lamps alone (ReSTIR DI), 5 the game's lights and the sun alone, 6
	// the contact ray's visibility (white 1, black 0). The terms come from cs_re9_path (RE9_SSS_E / RE9_SSS_H). v86: 7 this
	// frame's bounce alone, from its samples (re9_bounce_now: no accumulation, no RR - ReSTIR PT's effect shows here). v87: 8
	// SHaRC's light at the camera's surfaces (sharc_debug_light), 9 SHaRC's voxels (sharc_debug_voxels)
	if (g_pass == 3 || g_pass == 6 || g_pass == 9 || g_pass == 10)   // v90: 10 ReGIR's surface cells (regir_debug_cells)
	{
		const float src = re9_load(RE9_SSS_H, id.xy).w;
		const float3 code = g_pass == 10 ? regir_debug_cells(id.xy, d) : g_pass == 9 ? sharc_debug_voxels(id.xy, d) : g_pass == 6 ? saturate(direct.w) : src == 1.0 ? float3(0.0, 0.6, 0.0) :
			src == 2.0 ? float3(0.6, 0.0, 0.0) : src == 3.0 ? float3(0.0, 0.0, 0.6) : src == 4.0 ? float3(0.6, 0.6, 0.0) : 0.05;
		re9_store(RE9_BOOST, id.xy, 0.0);
		Light[id.xy] = float4(code, 1.0);
		return;
	}
	if (g_pass == 1)
		total = direct.rgb + direct_d.rgb;
	else if (g_pass == 2)
		total = diffuse * g_bounce + specular;
	else if (g_pass == 4)
		total = re9_load(RE9_SSS_E, id.xy).rgb;
	else if (g_pass == 5)
		total = re9_load(RE9_SSS_H, id.xy).rgb;
	else if (g_pass == 7)
		total = re9_bounce_now(id.xy, ad, as);              // v86: this frame's bounce, before the accumulation and RR
	else if (g_pass == 8)
		total = sharc_debug_light(id.xy, d);                // v87: SHaRC's light leaving the camera's surfaces
	const float3 gain = lighting_factor(id.xy, d) * g_brightness;
	// v73: a result that is not a finite number keeps the game's own lighting at that pixel - v44-v72 clamped it with max(),
	// which turns NaN into 0: a black pixel
	const bool valid = all(isfinite(total)) && all(isfinite(gain));
	total = max(total, 0.0);
	// v64: the pixel's own glow (lamps; v44-v63 only with the 5% gate open) - v66: the game's own emission (G-buffer render
	// target 6), added as its lighting adds it (v65: the game's light at pixels 12x the frame's mean, over this renderer's)
	// (v74: not in the debug views - each term alone)
	const float3 self = g_pass == 0 ? emission_at(id.xy) : 0.0;
	// v50: the game's own result, the forward relight's reference (cs_re9_ratio)
	const float3 game = Light[id.xy].rgb;
	Data.Store(relight_gp_offset() + (id.y * size.x + id.x) * 8, pack_r11g11b10(game));
	Light[id.xy] = float4(lerp(game, valid ? total * gain + self : game, g_full), 1.0);
	re9_store(RE9_BOOST, id.xy, valid ? float4(diffuse * max(g_bounce * gain - min(gain, 1.0), 0.0) * g_full + self * g_full, 0.0) : 0.0);
	if (noisy)                                              // v81: the light buffer as without RR in the game, for the history
		re9_store(RE9_SETTLED, id.xy, float4(lerp(game, valid && all(isfinite(settled)) ? max(settled, 0.0) * gain + self : game, g_full), 0.0));
}

// The camera's history for the rays' hits (rt.hlsl cs_history's job in the RE9 renderer): the light buffer without the
// Bounce slider's extra and the emitters' own light (RE9_BOOST), a sixteenth of the pixels into SHaRC
[numthreads(8, 8, 1)]
void cs_re9_history(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	float3 l = Light[id.xy].rgb;
	const float d = Depth[id.xy];
	if (d < 1.0)
	{
		// v81: RR denoising in the game - the light buffer holds this frame's noisy light, the composite kept the accumulated
		// light (RE9_SETTLED): the hits of the next frames take that, as before
		if ((rr_load_block().flags & RR_NOISY) != 0u)
			l = re9_load(RE9_SETTLED, id.xy).rgb;
		const float4 boost = re9_load(RE9_BOOST, id.xy);
		if (all(isfinite(boost.rgb)))
			l = max(l - max(boost.rgb, 0.0), 0.0);
		// (v87: v23-v86's own SHaRC only - NVIDIA's SHaRC takes the game's light at the camera's surfaces in its Update pass)
		if (all((id.xy & 3u) == uint2(g_frame & 3u, (g_frame >> 2) & 3u)) && (pixel_flags(id.xy) & PIXEL_HAIR) == 0 && !sharc_nv_on())
		{
			// v64: in the DD2 renderer (g_pass 1) SHaRC keeps the game's own light leaving the surface - its lighting result
			// (kept by the composite for the relight) without the pixel's glow, over the lighting's factor: what a DD2 path
			// takes at a hit off screen (dd2.hlsl). The RE9 renderer keeps its own result there
			float3 cache = l;
			if (g_pass == 1)
				cache = max(unpack_r11g11b10(Data.Load(relight_gp_offset() + (id.y * size.x + id.x) * 8)) - emission_at(id.xy), 0.0) /
					max(lighting_factor(id.xy, d), 1e-4);
			if (all(isfinite(cache)))
				sharc_add(camera_relative_position(float2(id.xy), d) + c[9].xyz, decode_normal(GBuffer0[id.xy]), cache);
			Data.InterlockedAdd(STATS + STAT_SHARC_INJECTED * 4, 1);
		}
	}
	Data.Store(history_offset() + (id.y * size.x + id.x) * 4, pack_r11g11b10(min(l, 256.0)));
}

// v50: the forward relight's reference (src\relight.hlsl runs before the game's fog): per pixel the game's result G (kept by
// the composite) and this renderer's P (the light buffer now, bit for bit as the copy will hold it), per 16 x 16 tile their
// luminance sums, and the frame token (FRAME_INFO.x) the relight checks. v54: the sky is left out - RELIGHT_SKY for P, not
// in the sums: the game draws its sky after the composite, and v50-v53 scaled it by its tiles' ratio (16-pixel blocks
// along the skyline, flickering with the ratio)
groupshared float2 gs_relight[256];
[numthreads(16, 16, 1)]
void cs_re9_ratio(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	const uint2 size = full_size();
	float2 lum = 0.0;
	if (id.x < size.x && id.y < size.y)
	{
		const uint at = relight_gp_offset() + (id.y * size.x + id.x) * 8;
		if (Depth[id.xy] < 1.0)
		{
			const float3 p = max(Light[id.xy].rgb, 0.0);
			const float3 g = unpack_r11g11b10(Data.Load(at));
			Data.Store(at + 4, pack_r11g11b10(p));
			lum = float2(restir_lum(p), restir_lum(max(g, 0.0)));
		}
		else
			Data.Store(at + 4, RELIGHT_SKY);
	}
	gs_relight[gi] = lum;
	GroupMemoryBarrierWithGroupSync();
	[unroll]
	for (uint stride = 128; stride > 0; stride >>= 1)
	{
		if (gi < stride)
			gs_relight[gi] += gs_relight[gi + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	if (gi == 0)
	{
		// v62 (user: "the eye fix still flickers alot"): the sums blended over the frames (RELIGHT_BLEND of the new one) - one
		// frame's path-traced noise and fireflies swung the ratio, and the layers' light with it, every frame
		const uint tiles_x = (size.x + RELIGHT_TILE - 1) / RELIGHT_TILE;
		const uint at = relight_tile_offset() + 16 + (gid.y * tiles_x + gid.x) * 8;
		float2 sums = gs_relight[0];
		const float2 prev = asfloat(Data.Load2(at));
		if (all(isfinite(prev)) && all(prev >= 0.0) && all(prev < 1e9) && Data.Load(FRAME_INFO) > 2)
			sums = lerp(prev, sums, RELIGHT_BLEND);
		Data.Store2(at, asuint(sums));
		if (gid.x == 0 && gid.y == 0)
			Data.Store(relight_tile_offset(), Data.Load(FRAME_INFO));
	}
}
