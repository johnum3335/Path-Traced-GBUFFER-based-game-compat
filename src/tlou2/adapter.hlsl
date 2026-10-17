// TLOU RT - the game adapter (v76; included by rt.hlsl). What the add-on's newer passes know of a game's frame comes
// through here, in the add-on's own format, so that another DX12 game needs only its side of each part. NVIDIA's RTXDI
// library (rtxdi_di.hlsl, compiled in unmodified) reads the game through it, by its bridge (rtxdi_bridge.hlsli).
// The parts:
//  1 surfaces (v76): per render pixel, this frame's and last frame's (two planes, ping-pong by frame parity) - position,
//    view depth, shading and geometric normal, albedo, roughness, metalness, specular reflectance, flags, the material's
//    occlusion, two words of the game's own. The game's side is game_surface(px, d) (TLOU's: rt.hlsl, its G-buffer
//    decoded); cs_adapt runs it for every pixel first of the add-on's passes each frame (tlou_rt.cpp, before the game's
//    lighting), and load_surface_record reads the planes - RTXDI's temporal reuse needs last frame's
//  2 the camera and the frame's constants - for now TLOU's lighting constants (rt.hlsl c[]: the render size [8], the
//    camera [9], the depth's linearisation [10], the inverse view-projection [19..22], the sun [6, 7]) and the add-on's
//    frame block (FRAME_*)
//  3 the lights - for now TLOU's light records (Lights, LightIndex) and the add-on's emitters (emit_*, lamp_*); v76: the
//    emitter list kept for two frames, with each lamp's slot in the lamps' memory (RTXDI's light index translation)
//  4 the game's lighting extras - for now TLOU's baked ambient (t4, t5 and G-buffer 1 .z's direction), its occlusion
//    terms (t6, t10, t16), the lighting's per-pixel factor (lighting_factor) and the emission (t2)
//  5 the scene's geometry (tlou_rt.cpp: stream-out of the visibility prepass, a BLAS per draw) and where the result goes
//    (the game's HDR light buffer, u0)
// The older passes still read the game directly (load_surface: the G-buffer); they move behind the adapter as needed.

#define SURFACE_BYTES 48
#define SURFACE_CHARACTER 1u    // flags: a character's pixel (skin, clothes, hair) - the game's lights may keep to or off them

struct surface_record
{
	float3 p;                   // position relative to that frame's camera (world axes)
	float view_depth;           // metres along the view axis (0: no surface - the sky)
	float3 n;                   // shading normal (world; the normal maps' detail)
	float3 ng;                  // geometric normal (world, toward the camera): from the depth buffer's neighbours
	float3 albedo;              // linear
	float roughness;            // perceptual
	float metallic;
	float spec;                 // specular reflectance, 0-255: F0 = spec x 0.08 / 255 for non-metals (128: 0.04)
	uint flags;                 // SURFACE_*
	float ao;                   // the material's own occlusion (cavity; 1 = open)
	uint game0;                 // two words for the game's own lighting (TLOU: the lights' occlusion as f16 - G-buffer 1 .x
	uint game1;                 //   bits 8-13 and the capsules' t16.x; the stencil, t23.x)
};

uint pack_normal(float3 n);                                 // rt.hlsl
float3 unpack_normal(uint v);
uint pt_offset(uint parity);

// Bytes: 0 position (float3) and view depth, 16 shading normal, 20 geometric normal (octahedral 16 + 16 each), 24 albedo
// (gamma 2.2 bytes, the G-buffer's own: TLOU's come back unchanged) and roughness (byte), 28 metalness, specular, flags,
// occlusion (bytes), 32 and 36 the game's words, 40 free
uint surface_offset(uint parity)                            // after ReSTIR PT's planes, two full-resolution planes (tlou_rt.cpp layout_for)
{
	const uint2 size = full_size();
	return align_up(pt_offset(2), 256) + parity * align_up(size.x * size.y * SURFACE_BYTES, 256);
}
uint surface_at(uint parity, uint2 px)
{
	return surface_offset(parity) + (px.y * full_size().x + px.x) * SURFACE_BYTES;
}
uint unorm8(float v)
{
	return uint(saturate(v) * 255.0 + 0.5);
}
void store_surface(uint parity, uint2 px, surface_record r)
{
	const uint at = surface_at(parity, px);
	const float3 a = pow(saturate(r.albedo), 1.0 / 2.2);
	Data.Store4(at, uint4(asuint(r.p), asuint(r.view_depth)));
	Data.Store4(at + 16, uint4(pack_normal(r.n), pack_normal(r.ng), unorm8(a.x) | (unorm8(a.y) << 8) | (unorm8(a.z) << 16) |
		(unorm8(r.roughness) << 24), unorm8(r.metallic) | (min(uint(max(r.spec, 0.0) + 0.5), 255u) << 8) | ((r.flags & 0xFFu) << 16) |
		(unorm8(r.ao) << 24)));
	Data.Store4(at + 32, uint4(r.game0, r.game1, 0u, 0u));
}
// parity: g_frame & 1 this frame's surfaces, its complement last frame's (as cs_adapt left them)
surface_record load_surface_record(uint parity, uint2 px)
{
	const uint at = surface_at(parity, px);
	const uint4 a = Data.Load4(at), b = Data.Load4(at + 16);
	const uint2 g = Data.Load2(at + 32);
	surface_record r;
	r.p = asfloat(a.xyz);
	r.view_depth = asfloat(a.w);
	r.n = unpack_normal(b.x);
	r.ng = unpack_normal(b.y);
	r.albedo = pow(float3(b.z & 0xFFu, (b.z >> 8) & 0xFFu, (b.z >> 16) & 0xFFu) / 255.0, 2.2);
	r.roughness = (b.z >> 24) / 255.0;
	r.metallic = (b.w & 0xFFu) / 255.0;
	r.spec = float((b.w >> 8) & 0xFFu);
	r.flags = (b.w >> 16) & 0xFFu;
	r.ao = (b.w >> 24) / 255.0;
	r.game0 = g.x;
	r.game1 = g.y;
	if (!all(isfinite(r.p)) || !(r.view_depth > 0.0 && r.view_depth < 1e6))
		r.view_depth = 0.0;                                 // never written, or broken
	return r;
}

surface_record game_surface(uint2 px, float d);             // the game's side (TLOU's: rt.hlsl)
float3 game_albedo(uint2 px);                               // v82: the game's albedo at a pixel (the auto-PBR layer's neighbourhood)

// ---- v82: the auto-PBR layer (user: "the autopbr could be a more generic approach to all games ... useful in case data
// could not be retrieved") ----
// What a game's frame does not hold of a surface's material - its roughness, metalness, specular reflectance, the normal
// maps' detail - is estimated from what every game's frame has: its albedo and its depth. The game's side names the
// channels it gives (GAME_SURFACE_CHANNELS, SURFACE_HAS_*: TLOU all of them); "Force auto-PBR (debug)" (TABLE_ADAPTER bit
// ADAPTER_FORCE_AUTO_PBR) takes the estimates for every channel, and load_surface hands them to the older passes too - what
// another game would get. Where the game has its own values the estimates are measured against them (one pixel in 16,
// characters left out, FRAME_AUTO_PBR): the status line's errors tune the rules on a game whose answer is known. The rules
// (v82, first cut):
//   roughness: the albedo's local contrast over the same surface (a cross of 8 taps within 2 pixels, 3 % in depth) - texture
//     detail runs with the microsurface, a uniform surface reads smoother: from 0.45 (uniform) to 0.9 (contrasted)
//   metalness 0, F0 0.04 (specular byte 128): an albedo alone does not tell a metal - a game that says so overrides it
//   the shading normal: the depth buffer's geometric normal bent by the albedo luminance's screen gradient along the
//     surface (light and dark read as a shallow relief), AUTO_PBR_BUMP strong
#define SURFACE_HAS_ROUGHNESS 1u
#define SURFACE_HAS_METALLIC 2u
#define SURFACE_HAS_SPECULAR 4u
#define SURFACE_HAS_NORMAL 8u
#define ADAPTER_FORCE_AUTO_PBR 1u
#define AUTO_PBR_BUMP 1.0

bool auto_pbr_forced()
{
	return (Table.Load(TABLE_ADAPTER) & ADAPTER_FORCE_AUTO_PBR) != 0u;
}
float auto_luminance(float3 albedo)
{
	return dot(albedo, float3(0.2126, 0.7152, 0.0722));
}
// the albedo's luminance at a neighbour on the same surface (within 3 % of this pixel's view depth), else this pixel's
float auto_neighbour_luminance(int2 q, float vz, float own)
{
	const uint2 size = full_size();
	if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y))
		return own;
	const float dq = Depth[uint2(q)];
	if (dq >= 1.0 || abs(view_z(dq) - vz) > 0.03 * vz + 0.01)
		return own;
	return auto_luminance(game_albedo(uint2(q)));
}
float auto_roughness(uint2 px, float3 albedo, float vz)
{
	const float own = auto_luminance(albedo);
	float sum = own, sum2 = own * own;
	const int2 offsets[8] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1), int2(-2, 0), int2(2, 0), int2(0, -2), int2(0, 2) };
	[unroll]
	for (int k = 0; k < 8; ++k)
	{
		const float l = auto_neighbour_luminance(int2(px) + offsets[k], vz, own);
		sum += l;
		sum2 += l * l;
	}
	const float mean = sum / 9.0;
	const float deviation = sqrt(max(sum2 / 9.0 - mean * mean, 0.0));
	return lerp(0.45, 0.9, saturate(deviation / (mean + 0.05) * 5.0));
}
float3 auto_normal(uint2 px, float3 p, float3 ng, float3 albedo, float vz, float d)
{
	const uint2 size = full_size();
	if (px.x == 0 || px.y == 0 || px.x + 1 >= size.x || px.y + 1 >= size.y)
		return ng;
	const float own = auto_luminance(albedo);
	const float gx = auto_neighbour_luminance(int2(px) + int2(1, 0), vz, own) - auto_neighbour_luminance(int2(px) - int2(1, 0), vz, own);
	const float gy = auto_neighbour_luminance(int2(px) + int2(0, 1), vz, own) - auto_neighbour_luminance(int2(px) - int2(0, 1), vz, own);
	// the surface's directions toward screen x and y: the neighbours' positions at this depth, onto the tangent plane
	const float3 tx = camera_relative_position(float2(px) + float2(1.0, 0.0), d) - p;
	const float3 ty = camera_relative_position(float2(px) + float2(0.0, 1.0), d) - p;
	const float3 t = tx - ng * dot(tx, ng), b = ty - ng * dot(ty, ng);
	if (!(dot(t, t) > 0.0) || !(dot(b, b) > 0.0))
		return ng;
	const float3 n = ng - AUTO_PBR_BUMP * (gx * normalize(t) + gy * normalize(b));
	return all(isfinite(n)) && dot(n, n) > 0.0 ? normalize(n) : ng;
}

// This frame's surfaces, every pixel (the sky: view depth 0). v82: the auto-PBR layer fills the channels the game does not
// give (all of them with "Force auto-PBR") and measures itself where the game does
[numthreads(8, 8, 1)]
void cs_adapt(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const float d = Depth[id.xy];
	surface_record r = game_surface(id.xy, d);
	uint4 error = 0;                                        // |estimate - the game's| x 1024: roughness, metalness, F0 x 10, normal
	uint compared = 0;
	if (r.view_depth > 0.0)
	{
		const uint channels = auto_pbr_forced() ? 0u : GAME_SURFACE_CHANNELS;
		const float roughness = auto_roughness(id.xy, r.albedo, r.view_depth);
		const float3 n = auto_normal(id.xy, r.p, r.ng, r.albedo, r.view_depth, d);
		const float metallic = 0.0, spec = 128.0;
		if (all((id.xy & 3u) == uint2(g_frame & 3u, (g_frame >> 2) & 3u)) && (r.flags & SURFACE_CHARACTER) == 0u)
		{
			compared = 1u;
			error = uint4(uint((GAME_SURFACE_CHANNELS & SURFACE_HAS_ROUGHNESS) != 0u ? abs(roughness - r.roughness) * 1024.0 + 0.5 : 0.0),
				uint((GAME_SURFACE_CHANNELS & SURFACE_HAS_METALLIC) != 0u ? abs(metallic - r.metallic) * 1024.0 + 0.5 : 0.0),
				uint((GAME_SURFACE_CHANNELS & SURFACE_HAS_SPECULAR) != 0u ? abs(spec - r.spec) * (0.08 / 255.0) * 10240.0 + 0.5 : 0.0),
				uint((GAME_SURFACE_CHANNELS & SURFACE_HAS_NORMAL) != 0u ? saturate(1.0 - dot(n, r.n)) * 1024.0 + 0.5 : 0.0));
		}
		if ((channels & SURFACE_HAS_ROUGHNESS) == 0u)
			r.roughness = roughness;
		if ((channels & SURFACE_HAS_METALLIC) == 0u)
			r.metallic = metallic;
		if ((channels & SURFACE_HAS_SPECULAR) == 0u)
			r.spec = spec;
		if ((channels & SURFACE_HAS_NORMAL) == 0u)
			r.n = n;
	}
	store_surface(g_frame & 1u, id.xy, r);
	error = WaveActiveSum(error);
	compared = WaveActiveSum(compared);
	if (WaveIsFirstLane() && compared != 0u)
	{
		Data.InterlockedAdd(FRAME_AUTO_PBR, error.x);
		Data.InterlockedAdd(FRAME_AUTO_PBR + 4, error.y);
		Data.InterlockedAdd(FRAME_AUTO_PBR + 8, error.z);
		Data.InterlockedAdd(FRAME_AUTO_PBR + 12, error.w);
		Data.InterlockedAdd(FRAME_AUTO_PBR + 16, compared);
	}
}
