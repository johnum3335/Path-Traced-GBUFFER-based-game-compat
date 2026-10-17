// TLOU RT - path reuse (v68): NVIDIA's "ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path
// Tracing" (Lin, Kettunen, Wyman 2026; sections cited as [n]) on the first bounce of both renderers. This step: paired
// spatial reuse of this frame's samples [3] with pairwise MIS and colour-weighted shading [6.3].
//   - each pixel's path is a sample (re9.hlsl reuse_store: the first hit and the light the path brings back, or the sky's
//     direction; W = 1 / pdf). The hit is the reconnection vertex: shifted to another pixel the sample is that hit seen
//     from the other pixel's origin, the light leaving it taken as the same (ReSTIR GI's reconnection shift), with the
//     Jacobian of the move (the cosine at the hit, the squared distances) and a visibility ray; a sky sample keeps its
//     direction and takes the receiving pixel's own sky light;
//   - pairing textures [3.1]: three square textures (254, 230, 210 texels) pair every texel with another, the offsets
//     Gaussian (sigma ~7.9 px, PAIR_ROUNDS), built once by cs_reuse_pairs (the paper's 2 x 2 shuffles), flipped, mirrored,
//     transposed and offset every frame [3.2]. Partners reuse each other, so a pair's two shifts are made once:
//     cs_reuse_shift shifts the pixel's own sample to each of its three partners (one ray each) for both of them;
//   - cs_reuse_resample combines the pixel's own sample (the canonical) with its partners' by defensive pairwise MIS with
//     confidence weights (unbiased: over the techniques that can make a sample the weights sum to one; a neighbour whose
//     shift fails brings nothing) and shades the colour-weighted sum of all of them, sum m_i f(y_i) W_i |J_i| [6.3] - the
//     chroma noise of picking by luminance averages out - diffuse and specular apart (the renderers' raw planes, before
//     the accumulation).
// No temporal reuse yet: fresh samples every frame, the noise does not stick to the surfaces. Next steps of the port:
// temporal reuse with duplication maps [5], the footprint reconnection criteria [4], ReSTIR DI the same way, direct and
// indirect light in one reservoir [6.1]. g_pass: 1 = the DD2 renderer (its lobes, rays and sky), 0 = RE9's.
// Revised after Daniel Qin's Novum (an OptiX ReSTIR PT Enhanced implementation, G:\r roblox\Novum-main\optixBranch): its
// defensive pairwise MIS (Lin et al. 2022 GRIS: the canonical keeps a share of its own - resolveSpatialReuse), its
// neighbour test (isSpatialNeighborValid: plane distance, 3D distance, the same material, both ways), a target with a
// firefly clamp (targetFunction) and tighter pairing textures (Novum: 16 rounds, sigma ~5.6 px; the paper's best quality
// was sigma ~10.6 at 1080p, ~6-8 px at TLOU's 632 lines: 32 rounds here).
#include "dd2.hlsl"

#define REUSE_PARTNERS 3
#define REUSE_J_MAX 8.0         // a shift whose Jacobian is above 8 (or below 1/8) fails: a reconnection beside a corner
#define REUSE_TARGET_MAX 64.0   // the target's luminance clamp (Novum's targetFunction = luminance(fireflyClamp(F))): one very
                                //   bright hit cannot swing the MIS weights (the shading keeps its own clamp, RE9_MAX)
#define PAIR_ROUNDS 32          // n_sigma rounds [eq. 3]: sigma = sqrt(2 n - 1) ~ 7.9 px (v68's first build: 128, sigma 16)
#define PAIR_MAGIC 0x50A20000u  // + the texture's size: cs_reuse_pairs has built it (the pairing block's first three words)
#define PAIR_BYTES 1938208u     // 16 + (254^2 + 230^2 + 210^2) x 12: per texture the deltas, the link indices and the link table

// The shifts (32 bytes per pixel after the origins, tlou_rt.cpp layout_for): the pixel's sample's target at the pixel,
// then per partner k: S_k = the sample's target at the partner (its lobes, the light, visibility) x the shift's Jacobian,
// T_k = visibility x Jacobian (the partner shades the sample with it)
uint reuse_shift_offset()
{
	const uint2 size = full_size();
	return align_up(reuse_origin_offset() + size.x * size.y * 16, 256);
}
uint pair_offset()
{
	const uint2 size = full_size();
	return align_up(reuse_shift_offset() + size.x * size.y * 32, 256);
}
uint pair_size(uint k)
{
	return k == 0u ? 254u : k == 1u ? 230u : 210u;
}
uint pair_texture(uint k)                                   // the deltas (int16 x, y); + S^2 x 4 the link indices, + 2 S^2 x 4 the table
{
	uint at = pair_offset() + 16;
	for (uint j = 0; j < k; ++j)
		at += pair_size(j) * pair_size(j) * 12;
	return at;
}
bool pairs_ready()
{
	const uint3 m = Data.Load3(pair_offset());
	return m.x == PAIR_MAGIC + 254u && m.y == PAIR_MAGIC + 230u && m.z == PAIR_MAGIC + 210u;
}

// Texture k's partner of pixel p this frame [3.2]: the texture flipped, mirrored, transposed and offset by the frame's
// random bits and the delta turned back the same way - still an involution (the partner's partner is p)
int2 pair_partner(int2 p, uint k)
{
	const uint S = pair_size(k);
	const uint h = pcg(g_frame * 0x9E3779B9u + k * 0x632BE5ABu + 0x7A1Bu), h2 = pcg(h ^ 0x5BD1E995u);
	const bool tr = (h & 1u) != 0u, fx = (h & 2u) != 0u, fy = (h & 4u) != 0u;
	int2 u = tr ? p.yx : p;
	u = int2(fx ? -u.x : u.x, fy ? -u.y : u.y) + int2((h >> 8) % S, (h2 >> 8) % S);
	const int s = int(S);
	const uint2 t = uint2((u % s + s) % s);
	const uint packed = Data.Load(pair_texture(k) + (t.y * S + t.x) * 4);
	int2 e = int2(int(packed << 16) >> 16, int(packed) >> 16);
	e = int2(fx ? -e.x : e.x, fy ? -e.y : e.y);
	return p + (tr ? e.yx : e);
}

// Builds the pairing textures once [3.1] (the magic words mark them built; tlou_rt.cpp dispatches it every frame the reuse
// runs - three groups that return at once after the first): texel i starts with link index i / 2 (its row neighbour has
// the same), PAIR_ROUNDS rounds shuffle every 2 x 2 block (a random permutation; odd rounds offset by one texel diagonally,
// wrapping over the edge), so the two texels of a link drift apart in a random walk; then each texel finds the other one
// with its link index (a table of two slots per link: the first comes in by compare-exchange) and stores the delta to it,
// wrapped into [-S/2, S/2] (tileable). One group of 1024 threads per texture.
[numthreads(1024, 1, 1)]
void cs_reuse_pairs(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	const uint k = gid.x;
	if (k >= REUSE_PARTNERS)
		return;
	const uint S = pair_size(k), N = S * S, B = S / 2u;
	const uint head = pair_offset() + k * 4;
	if (Data.Load(head) == PAIR_MAGIC + S)
		return;                                             // built (the group agrees: one word, read before anyone writes it)
	const uint tex = pair_texture(k), link = tex + N * 4, table = link + N * 4;
	for (uint i = gi; i < N; i += 1024u)
	{
		Data.Store(link + i * 4, i >> 1);
		Data.Store(table + i * 4, 0xFFFFFFFFu);
	}
	DeviceMemoryBarrierWithGroupSync();
	for (uint r = 0; r < PAIR_ROUNDS; ++r)
	{
		const uint o = r & 1u;
		for (uint b = gi; b < B * B; b += 1024u)
		{
			const uint x0 = ((b % B) * 2u + o) % S, y0 = ((b / B) * 2u + o) % S, x1 = (x0 + 1u) % S, y1 = (y0 + 1u) % S;
			const uint4 at = uint4(y0 * S + x0, y0 * S + x1, y1 * S + x0, y1 * S + x1);
			uint v[4] = { Data.Load(link + at.x * 4), Data.Load(link + at.y * 4), Data.Load(link + at.z * 4), Data.Load(link + at.w * 4) };
			const uint h = pcg(pcg(r * 0x9E3779B9u + k * 0x85EBCA6Bu + 0x51u) ^ b);
			uint j = h % 4u;                                    // a random permutation of the block (Fisher-Yates)
			uint t = v[3];
			v[3] = v[j];
			v[j] = t;
			j = (h >> 8) % 3u;
			t = v[2];
			v[2] = v[j];
			v[j] = t;
			j = (h >> 16) & 1u;
			t = v[1];
			v[1] = v[j];
			v[j] = t;
			Data.Store(link + at.x * 4, v[0]);
			Data.Store(link + at.y * 4, v[1]);
			Data.Store(link + at.z * 4, v[2]);
			Data.Store(link + at.w * 4, v[3]);
		}
		DeviceMemoryBarrierWithGroupSync();
	}
	for (uint i2 = gi; i2 < N; i2 += 1024u)
	{
		const uint l = Data.Load(link + i2 * 4);
		uint first;
		Data.InterlockedCompareExchange(table + l * 8, 0xFFFFFFFFu, i2, first);
		if (first != 0xFFFFFFFFu)
			Data.Store(table + l * 8 + 4, i2);
	}
	DeviceMemoryBarrierWithGroupSync();
	const int half_s = int(B);
	for (uint i3 = gi; i3 < N; i3 += 1024u)
	{
		const uint2 two = Data.Load2(table + Data.Load(link + i3 * 4) * 8);
		const uint other = two.x == i3 ? two.y : two.x;
		int dx = int(other % S) - int(i3 % S), dy = int(other / S) - int(i3 / S);
		if (dx > half_s)
			dx -= int(S);
		else if (dx < -half_s)
			dx += int(S);
		if (dy > half_s)
			dy -= int(S);
		else if (dy < -half_s)
			dy += int(S);
		Data.Store(tex + i3 * 4, (uint(dx) & 0xFFFFu) | (uint(dy) << 16));
	}
	DeviceMemoryBarrierWithGroupSync();
	if (gi == 0u)
		Data.Store(head, PAIR_MAGIC + S);
}

// A pixel as the reuse sees it: its surface (DD2: the shading normal lifted as cs_dd2_path lifts it), the lobes' albedos
// and the path's ray origin and triangle normal (the path passes' origin plane)
struct reuse_pixel
{
	uint2 px;
	surface s;
	float3 ad, as, f0;
	float3 origin;
	float3 ng;
};
bool reuse_pixel_load(uint2 px, out reuse_pixel r)
{
	r = (reuse_pixel)0;
	const float d = Depth[px];
	if (d >= 1.0 || (pixel_flags(px) & PIXEL_HAIR) != 0u)
		return false;                                       // the sky, hair: no reuse
	r.px = px;
	r.s = load_surface(px, d);
	re9_albedos(r.s, r.ad, r.as, r.f0);
	const uint4 o = Data.Load4(reuse_origin_offset() + (px.y * full_size().x + px.x) * 16);
	r.origin = asfloat(o.xyz);
	r.ng = unpack_normal(o.w);
	r.s.n = visible_normal(r.s.n, r.s.v, r.ng);             // as the paths shade (v69: the RE9 renderer's too)
	return all(isfinite(r.origin));
}

// Whether pixels p and q reuse each other's samples - the same answer for both and in both passes (Novum's
// isSpatialPairAccepted: the test both ways): on the screen, neither the sky nor hair, the same kind of surface (the
// character bit, roughness within a quarter), G-buffer normals within ~25 deg, each one on the other's plane - |(q - p).n|
// within 0.5 % of (1 m + its distance from the camera), with the triangles' normals of the paths' origin plane - and no
// further apart than 30 times that (v68's first build: view depth within 5 %, which rejected neighbours on floors seen at
// a grazing angle and took neighbours on another plane facing the camera)
bool reuse_pair_ok(uint2 p, int2 q)
{
	const uint2 size = full_size();
	if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y) || all(uint2(q) == p))
		return false;
	const uint2 uq = uint2(q);
	if (Depth[p] >= 1.0 || Depth[uq] >= 1.0 || ((pixel_flags(p) | pixel_flags(uq)) & PIXEL_HAIR) != 0u)
		return false;
	const uint4 gp = GBuffer0[p], gq = GBuffer0[uq];
	if (dot(decode_normal(gp), decode_normal(gq)) < 0.9 || abs(float(gp.w & 0xFFu) - float(gq.w & 0xFFu)) > 64.0 ||
		((GBuffer1[p].w ^ GBuffer1[uq].w) & 1u) != 0u)
		return false;
	const uint4 op = Data.Load4(reuse_origin_offset() + (p.y * size.x + p.x) * 16);
	const uint4 oq = Data.Load4(reuse_origin_offset() + (uq.y * size.x + uq.x) * 16);
	const float3 pp = asfloat(op.xyz), pq = asfloat(oq.xyz), d = pq - pp;
	const float tp = 0.005 * (1.0 + length(pp)), tq = 0.005 * (1.0 + length(pq));
	return abs(dot(d, unpack_normal(op.w))) <= tp && abs(dot(d, unpack_normal(oq.w))) <= tq && length(d) <= 30.0 * min(tp, tq);
}

// The primary's lobes toward l per unit light, each renderer's own: diffuse cos / pi (demodulated by Ad) and specular f cos
// / As - DD2: its GGX with Smith-joint-approx visibility (dd2_primary_lobes); RE9: re9_spec_integrand
void reuse_lobes(reuse_pixel r, float3 l, out float fd, out float3 fs)
{
	if ((g_pass & 1u) != 0u)
	{
		float pdf;
		dd2_primary_lobes(r.s.n, r.s.v, l, r.f0, max(r.s.roughness * r.s.roughness, 1e-4), r.as, fd, fs, pdf);
		return;
	}
	fd = max(dot(r.s.n, l), 0.0) * 0.31830989;
	float pdf_s;
	fs = re9_spec_integrand(r.s, l, r.f0, r.as, pdf_s);
}
// The target: the luminance of the light the sample brings, through both lobes and their albedos, clamped at
// REUSE_TARGET_MAX (used the same way everywhere: the MIS weights stay consistent)
float reuse_target_of(reuse_pixel r, float fd, float3 fs, float3 L)
{
	return min(restir_lum(L * (r.ad * fd + r.as * fs)), REUSE_TARGET_MAX);
}
float reuse_target(reuse_pixel r, float3 l, float3 L)
{
	float fd;
	float3 fs;
	reuse_lobes(r, l, fd, fs);
	return reuse_target_of(r, fd, fs, L);
}

// The light from out of the scene along dir at pixel r, as its renderer's path takes it when the first ray leaves the scene
// (the pixel's own baked ambient and sky visibility: a reused sky sample is lit as the receiving pixel sees the sky)
float3 reuse_sky_light(reuse_pixel r, float3 dir)
{
	const float3 a4 = InT4[r.px].rgb, a5 = InT5[r.px].rgb;
	const float3 axis = decode_ambient_direction(GBuffer1[r.px]);
	const float2 uv = (float2(r.px) + 0.5) * c[8].zw;
	const float open_k = 1.0;                               // v97: no game sky visibility (the Occlusion slider is gone)
	return dd2_escaped(r.px, dir, a4, a5, axis, open_k, 1.0) * g_sky;   // v71: both renderers' paths leave the scene as DD2's
}

// Whether nothing blocks the ray from r's origin along dir up to tmax (the renderers' bounce rays: the cut-outs solid where
// their mask is; v69 both - v68 the RE9 renderer's went through them)
bool reuse_clear(reuse_pixel r, float3 dir, float tmax)
{
	return !trace_cutouts(r.origin, dir, tmax).hit;
}

// The sample's own direction and light at its own pixel
bool reuse_own(reuse_sample x, reuse_pixel me, out float3 dir, out float3 L)
{
	L = x.L;
	dir = x.x;
	if (x.kind == REUSE_SKY)
		return true;
	const float3 y = x.x - c[9].xyz - me.origin;
	const float len = length(y);
	dir = y / max(len, 1e-6);
	return len > 1e-4;
}

// Sample x of pixel a shifted to pixel b: the direction from b's origin, the light there, the Jacobian |dT| (a hit: the
// cosines at the hit and the squared distances; the sky: 1) - false when b cannot take it (behind its surface, the hit's
// back, a Jacobian beyond REUSE_J_MAX, something in between)
bool reuse_shift(reuse_sample x, reuse_pixel a, reuse_pixel b, out float3 dir, out float3 L, out float J)
{
	dir = 0.0;
	L = 0.0;
	J = 0.0;
	if (x.kind == REUSE_SKY)
	{
		dir = x.x;
		if (dot(dir, b.ng) <= 0.0)
			return false;
		J = 1.0;
		L = reuse_sky_light(b, dir);
		return reuse_clear(b, dir, DD2_TMAX);                  // (v71: the RE9 renderer's bounce too; v68-v70 its Ray length)
	}
	if (x.kind != REUSE_HIT)
		return false;
	const float3 ya = x.x - c[9].xyz - a.origin, yb = x.x - c[9].xyz - b.origin;
	const float da = length(ya), db = length(yb);
	if (!(da > 0.05) || !(db > 0.05))
		return false;
	dir = yb / db;
	const float ca = dot(x.n, -ya / da), cb = dot(x.n, -dir);
	if (!(ca > 1e-3) || !(cb > 1e-3) || dot(dir, b.ng) <= 0.0)
		return false;
	J = (cb / ca) * (da * da) / (db * db);
	if (!(J < REUSE_J_MAX && J > 1.0 / REUSE_J_MAX))
		return false;
	L = x.L;
	return reuse_clear(b, dir, db - max(0.02, db * 0.02));   // up to the hit's surface
}

// Pass 1 of the paired reuse [3, 7]: the pixel's sample shifted to each of its three partners - what both of them need of
// that pair (the partner shades the sample with T, this pixel's MIS weight takes S) - and the sample's target here
[numthreads(8, 8, 1)]
void cs_reuse_shift(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	float own = 0.0;
	float S[REUSE_PARTNERS], T[REUSE_PARTNERS];
	[unroll]
	for (uint k0 = 0; k0 < REUSE_PARTNERS; ++k0)
	{
		S[k0] = 0.0;
		T[k0] = 0.0;
	}
	reuse_pixel me;
	if (pairs_ready() && reuse_pixel_load(id.xy, me))
	{
		const reuse_sample x = reuse_load(id.xy);
		float3 dir0, L0;
		if (x.kind != REUSE_NONE && reuse_own(x, me, dir0, L0))
		{
			own = reuse_target(me, dir0, L0);
			[unroll]
			for (uint k = 0; k < REUSE_PARTNERS; ++k)
			{
				const int2 q = pair_partner(int2(id.xy), k);
				reuse_pixel other;
				if (!reuse_pair_ok(id.xy, q) || !reuse_pixel_load(uint2(q), other))
					continue;
				float3 dir, L;
				float J;
				if (!reuse_shift(x, me, other, dir, L, J))
					continue;
				S[k] = reuse_target(other, dir, L) * J;
				T[k] = J;
			}
		}
	}
	const uint at = reuse_shift_offset() + (id.y * size.x + id.x) * 32;
	Data.Store4(at, asuint(float4(own, S[0], T[0], S[1])));
	Data.Store4(at + 16, asuint(float4(T[1], S[2], T[2], 0.0)));
}

// Pass 2: the pixel's own sample (the canonical, confidence 1) and its partners' (1 each), the defensive pairwise MIS of
// Lin et al. 2022 (GRIS) as Novum's resolveSpatialReuse weighs them - the canonical keeps c_c / C of its own; partner q's
// share c_q / C is split by the balance heuristic of (sum of the partners' c) p_<-q against c_c p_c, at the partner's
// shifted sample (p_<-q(y) = p_q(x_q) / |J|) and at the canonical's (p_<-q(x_c) = S, the canonical shifted there) - and the
// colour-weighted sum of all of them [6.3] in place of the path's raw sample (RE9_MAX clamp as the paths'; the planes' w
// kept). v68's first build split the canonical's confidence over the pairs without its own share: ~9 % less noise when
// the partners are as good as the pixel itself, much more when one is not (scratchpad mis_variants.py)
[numthreads(8, 8, 1)]
void cs_reuse_resample(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	reuse_pixel me;
	if (!pairs_ready() || !reuse_pixel_load(id.xy, me))
		return;                                             // the sky, hair: the path's own sample stays
	const uint at = reuse_shift_offset() + (id.y * size.x + id.x) * 32;
	const float4 m0 = asfloat(Data.Load4(at)), m1 = asfloat(Data.Load4(at + 16));
	const float own = m0.x;
	const float S_own[REUSE_PARTNERS] = { m0.y, m0.w, m1.y };
	int2 q[REUSE_PARTNERS];
	bool ok[REUSE_PARTNERS];
	float K = 0.0;
	[unroll]
	for (uint k0 = 0; k0 < REUSE_PARTNERS; ++k0)
	{
		q[k0] = pair_partner(int2(id.xy), k0);
		ok[k0] = reuse_pair_ok(id.xy, q[k0]);
		K += ok[k0] ? 1.0 : 0.0;
	}
	const float C = 1.0 + K;                                // the confidences: 1 each
	float mc = 1.0 / C;                                     // the canonical's own share
	float3 sum_d = 0.0, sum_s = 0.0;
	uint used = 0;                                          // partners whose sample brought light here (the counters below)
	[unroll]
	for (uint k = 0; k < REUSE_PARTNERS; ++k)
	{
		if (!ok[k])
			continue;
		const float den = own + K * S_own[k];
		mc += (den > 0.0 ? own / den : 1.0) / C;
		// the partner's sample here: its own shift to this pixel (this pixel is the partner's partner k)
		const uint qa = reuse_shift_offset() + (uint(q[k].y) * size.x + uint(q[k].x)) * 32;
		const float4 n0 = asfloat(Data.Load4(qa)), n1 = asfloat(Data.Load4(qa + 16));
		const float tq = n0.x;
		const float Tq = k == 0u ? n0.z : k == 1u ? n1.x : n1.z;
		if (!(Tq > 0.0) || !(tq > 0.0))
			continue;
		const reuse_sample y = reuse_load(uint2(q[k]));
		if (y.kind == REUSE_NONE)
			continue;
		float3 dir = y.x, L = y.L;
		if (y.kind == REUSE_SKY)
			L = reuse_sky_light(me, dir);
		else
			dir = normalize(y.x - c[9].xyz - me.origin);
		float fd;
		float3 fs;
		reuse_lobes(me, dir, fd, fs);
		const float tc = reuse_target_of(me, fd, fs, L);
		const float w = K * tq / (tc * Tq + K * tq) / C * y.W * Tq;   // m_q W_q |J| (visibility in T)
		if (!(w > 0.0) || !isfinite(w))
			continue;
		sum_d += L * (fd * w);
		sum_s += L * fs * w;
		++used;
	}
	const reuse_sample x = reuse_load(id.xy);
	float3 dir0, L0;
	if (x.kind != REUSE_NONE && reuse_own(x, me, dir0, L0))
	{
		float fd;
		float3 fs;
		reuse_lobes(me, dir0, fd, fs);
		const float w = mc * x.W;
		sum_d += L0 * (fd * w);
		sum_s += L0 * fs * w;
	}
	// the renderer's firefly clamp: DD2's on the demodulated sum (RE9_MAX); RE9's (v70) on the light added to the pixel, against
	// RE9_MAX times the albedos (re9_clamp_limit: the specular no longer cut where its albedo is small)
	const bool dd2 = (g_pass & 1u) != 0u;
	const float l = dd2 ? restir_lum(sum_d + sum_s) : restir_lum(me.ad * sum_d + me.as * sum_s);
	const float lmax = dd2 ? RE9_MAX : re9_clamp_limit(me.ad, me.as);
	if (!(l <= lmax))
	{
		const float kmax = l > lmax ? lmax / l : 0.0;
		sum_d *= kmax;
		sum_s *= kmax;
	}
	const float4 raw_d = re9_load(RE9_RAW_D, id.xy), raw_s = re9_load(RE9_RAW_S, id.xy);
	re9_store(RE9_RAW_D, id.xy, float4(max(sum_d, 0.0), raw_d.w));
	re9_store(RE9_RAW_S, id.xy, float4(max(sum_s, 0.0), raw_s.w));
	// this frame's counters for the status line (tlou_rt.cpp reads FRAME_INFO back; cs_setup zeroes its y, z, w): pixels
	// resampled, partners accepted, partners whose sample brought light - one atomic per wave
	const uint3 counts = WaveActiveSum(uint3(1u, uint(K), used));
	if (WaveIsFirstLane())
	{
		Data.InterlockedAdd(FRAME_INFO + 4, counts.x);
		Data.InterlockedAdd(FRAME_INFO + 8, counts.y);
		Data.InterlockedAdd(FRAME_INFO + 12, counts.z);
	}
}

// Offline check (scratchpad harness only, not built into the add-on): every pixel of a 64 x 64 frame looks up its partner
// in each texture and that partner's partner; per pixel and texture the partner (int16 x, y) and whether it came back
[numthreads(8, 8, 1)]
void cs_reuse_pairs_test(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	if (all(id.xy == 0u))
		Data.Store2(0, uint2(pair_offset(), reuse_shift_offset()));
	[unroll]
	for (uint k = 0; k < REUSE_PARTNERS; ++k)
	{
		const int2 q = pair_partner(int2(id.xy), k);
		const int2 back = pair_partner(q, k);
		Data.Store2(reuse_shift_offset() + ((id.y * size.x + id.x) * REUSE_PARTNERS + k) * 8,
			uint2((uint(q.x) & 0xFFFFu) | (uint(q.y) << 16), all(back == int2(id.xy)) ? 1u : 0u));
	}
}
