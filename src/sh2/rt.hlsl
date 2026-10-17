// TLOU RT - path-traced indirect light for the game's own deferred lighting (CS_DeferredLighting), with the scene's
// triangles (stream-out of the visibility prepass) in hardware ray tracing.
// Runs with an extended copy of the game's lighting root signature: the game's parameters (re-bound by the add-on, so
// only bindings CS_DeferredLighting declares are used, at the same registers) plus the add-on's root descriptors and
// 8 root constants (space 100).
// Layouts (from the DXIL of MainPsGBuffer and CS_DeferredLighting, RT Recon v2/v3 and TLOU RT v10, 2026-10-10):
//  t0  G-buffer target 0, RGBA16_UINT: x = G<<8 | R, y = spec<<8 | B (albedo bytes gamma 2.2); normal octahedral, x from
//      12 bits (z & 0xFF | w & 0xF00), y from 11 bits (z >> 8 | (w >> 4) & 0x700), world axes (right-handed, Y up)
//  t4, t5  G-buffer targets 4 and 5 (R11G11B10_FLOAT): ambient irradiance (baked lightmaps / probes, sky-blue, with fine
//      occlusion). The lighting's ambient = albedo * (t4 * t6.x * (1 - ..) * min(t16.y, ..) + t5 * saturate(dot(N, ..))
//      * min(t16.x, ..)) * t10.y * zone tint (c[41..43]) + environment specular (t18 * split-sum LUT t19 32x32 RG8).
//  t8  depth, d = c10.x + c10.y / z (d == 1: sky);  t10 RGBA8: x = sun shadow mask, y = ambient occlusion
//  b0  [6] = sun direction, [7].xyz = sun colour (0 = no sun), [8] = (width, height, 1/width, 1/height), [9].xyz =
//      camera position (world), [10].xy = depth linearisation z = y / (d - x), [19..22] = rows of the inverse
//      view-projection: ([ndc.x, ndc.y, d, 1] * M).xyz / w = camera-relative position, [37].xyz = sky pixel colour
//  u0  HDR light buffer (R11G11B10_FLOAT): the lighting result, in the game's pre-exposed units
// For cosine-distributed rays the mean incoming radiance is irradiance / pi, so albedo * mean = the diffuse bounce
// radiance (PathTracer project's pt_sim2.py); the game uses the same convention (ambient = albedo * t4: t4 = E / pi).
// The 4x4 inverse and the reprojection round trip are checked in inv_check.py.
// Scene geometry: the add-on streams the clip-space xyz of every visibility prepass vertex into Geo (12 bytes per vertex,
// one region per draw, 2 frame regions); cs_zero / cs_convert turn this frame's region into absolute world positions in
// place (w = (z - c[10].y) / c[10].x, the game's projection; float32 error < 1.1 cm within 100 m, clip_check.py) and
// write one TLAS instance per draw; the add-on refits / builds one BLAS per draw (v9). A refit keeps every triangle's
// active state, so invalid vertices become zero-area triangles (not NaN) and an incomplete draw gets instance mask 0.
// v11 (Remix-style, before the game's lighting): the near field is path traced, the far field stays the game's own
// ambient. Each half-resolution pixel shoots cosine rays up to the ray length: a ray that escapes lets the game's
// ambient through (visibility V), a ray that hits returns the light that point had last frame (the previous frame's lit
// result, reprojected; B), or, hidden then, an occluder lit like this point with a dark albedo. After temporal
// accumulation and an edge-aware filter cs_ambient rewrites the lighting's inputs, copied into G-buffer targets 4 and 5
// before CS_DeferredLighting reads them, so the game's own lighting shades the path-traced light with its materials.
// After the lighting, cs_history keeps the result (minus the bounce boost above one physical bounce, so the feedback
// cannot run away) for the next frame. cs_check traces camera rays to the depth buffer's surfaces for the overlay.
// v13: the game's ambient is a flat part (t4) plus a lobe from a per-pixel dominant direction D (t5 * saturate(N.D),
// D octahedral in G-buffer 1 .z, 8 bits per axis; mostly the open sky: median D.y 0.89, 72% of the ambient in the v10
// capture). Path-traced sky shadows: rays toward D within a cone (the softness) give the lobe's visibility S, the
// cosine rays the flat part's V: t4' = t4 * V' + B * bounce, t5' = t5 * S' (V', S' = lerp(1, V or S, occlusion) * sky
// light). 4 cosine + 2 shadow rays per half-resolution pixel, and an SVGF-style filter (5 a-trous passes whose
// visibility edge-stopping follows the accumulated variance, so converged contact shadows stay sharp). Hair strands
// (alpha-tested cards, solid sheets in the BVH) are left out of these rays.
// v14 (scene data for a Remix-style path tracer): a material cache. Remix knows the material of every triangle a ray
// hits; here cs_material learns it from what the camera sees: a camera ray per sampled pixel finds the triangle under it
// (the draw's TLAS instance and the BLAS primitive, instancing included) and the G-buffer's albedo and roughness there
// are blended into that triangle's slot (one uint per triangle per cached BLAS, in the geo buffer, 0 = not seen yet).
// Ray hits anywhere then look the material up (hit_material) and take the triangle's geometric normal from the frame's
// vertices (hit_normal). Not used for shading yet (v15); the Numpad - capture shows it ("ray view" slots).
// v15 (the Remix-style integrator on that scene data): a ray's hit gives back what the camera saw there last frame, else
// the hit triangle's cached material lit by the sun (shadow ray) and by one continuation ray (hit_light), so bounce light
// carries real colours and sunlight from off-screen surfaces. Two more lighting inputs are written: the sun shadow mask
// t10.x (multiplied by the visibility of a ray toward the sun in a small cone: the game's shadow maps keep the casters the
// game did not draw this frame, the rays add the contact shadows) and the reflections t18 (a GGX ray per pixel, alpha =
// roughness^2 from the G-buffer byte the lighting's split-sum LUT is indexed with; the game's reflection stays where the
// ray hits nothing). c[6] = the direction toward the sun (the lighting's N.L), c[7] = its base sun colour.
// v18 (the game's lights, ray traced): CS_DeferredLighting's light loop is switched off (the add-on points the lighting's
// own t13 descriptor - the light count of each 16x16 tile - at zeros) and the add-on lights the frame instead, after the
// lighting: every light of the pixel's tile exactly as the game evaluates it (decoded from its DXIL: falloff, near fade,
// spot cones, box windows, cookies, the sphere light's representative point for the specular, the game's BRDF), but
// with ray-traced visibility instead of shadow maps: one ray per pixel toward a point on the sphere of the light picked in
// proportion to its unshadowed contribution (RIS, ReSTIR's first stage). Only that visibility is filtered (temporal,
// clamped to the frame's neighbourhood so moving shadows do not trail, then a-trous): the lights themselves are exact
// every frame. cs_light_add puts the result into the light buffer with the lighting's final per-pixel factor, before
// cs_history keeps it, so the bounce light carries the game's lights too.
// v20 (temporal stability on moving characters): every history (the near field's planes and the lights' visibility)
// followed the camera only, so on a character who breathes or talks the reprojected history landed on another point
// and was dropped or smeared every frame - the light flickered in patches. cs_motion now finds the triangle under each
// pixel (a camera ray into the scene) and where that triangle was last frame: the same BLAS entry's vertices in the
// previous frame region (the table's previous-data offset), interpolated with the hit's barycentrics. The temporal passes
// reproject with that position. The lights trace up to LIGHT_RAYS shadow rays per pixel: shared among the contributing
// lights when there are that many or fewer (no selection noise), else LIGHT_RAYS stratified picks in proportion to the
// lights' contributions.
// v21 (user 2026-10-10: path tracing floods rooms with light, hair glows, faces flat, leaves and hair flicker, the flashlight
// beam has a dark centre that jumps): rays leaving the near field take the baked ambient again (the probes lit rooms through
// the off-screen ceiling and walls the scene lacks); full path tracing keeps the game's small-scale occlusion and its
// large-scale ambient shadow on the indirect light; the half-resolution pixel is the front-most of its block (no
// switching between strand and skin, leaf and wall); hair keeps the game's lighting; the bounce feedback keeps one
// physical bounce under full path tracing too; shadow rays skip what a light's shadow map leaves out near the light.
// v22 (user 2026-10-10 clip: Ellie's body still shadowed her own flashlight beam - the game puts that light at her head and
// leaves the player out of its shadow map; grass cast solid slabs; small rocks' hard shadows shimmered as the flashlight
// swayed; a companion's hair glowed in the beam): lights the game shadows use its own shadow maps, read exactly as the
// lighting reads them (game_shadow_map, decoded from its DXIL), and get no rays; lights without one get the game's
// G-buffer occlusion and the rays on top; the rays see solid geometry only (cut-out cards: MASK_CUTOUT); the sun in full
// path tracing takes the rewritten t10.x (shadow maps, clouds, ray-traced contact). Light shadows 0 = the game's lights.
// v23 (user 2026-10-11, after GBRT's shots - "path tracing + NEE + MIS from RTX Remix", "SHaRC"): the path tracer's hits
// get Remix-style next event estimation of the sun and of the game's lamps (lamps_at: RIS over the frame's light list, the
// game's shadow map or a ray for the pick); a spatially hashed radiance cache (SHaRC idea, own implementation) holds the
// light leaving surfaces in world space - written by the camera's view and by the rays, read where a ray hits what the
// camera did not see and at the second bounce, so the bounces beyond and off-screen surfaces carry light; the sky lobe is
// sampled by the cosine rays and the cone rays toward its direction, combined by MIS. Full path tracing gets its own sun
// (Sun slider: strength along the game's sun direction, ray-traced shadows only, cut-outs letting light through at
// random) and a Brightness slider.

Texture2D<uint4> GBuffer0 : register(t0);
Texture2D<uint4> GBuffer1 : register(t1);
Texture2D<float4> InT4 : register(t4);
Texture2D<float4> InT5 : register(t5);
Texture2D<float4> InT10 : register(t10);                    // sun shadow mask (x), ambient occlusion (y): RGBA8
Texture2D<float4> InT18 : register(t18);                    // the game's reflections (R11G11B10)
Texture2D<float> Depth : register(t8);
Texture2D<float4> InT7 : register(t7);                      // skin's subsurface light (CS_SkinDiffuse after SSSS; s3)
Texture2D<float4> InT11 : register(t11);                    // a sun mask (loaded at the pixel; a 16x16 placeholder seen)
Texture2D<float4> InT16 : register(t16);                    // character capsule occlusion (RG8, sampled with s3)
Texture2D<float4> InT6 : register(t6);                      // the game's SSAO (x: on the flat ambient, y: on the lobe)
Texture2D<float4> InT19 : register(t19);                    // split-sum BRDF LUT 32x32 (s3)
Texture2D<float4> InT22 : register(t22);                    // screen-space reflections (w: their weight; s3)
Texture2D<uint4> InT23 : register(t23);                     // stencil (.x, as the lighting reads it)
Texture2D<uint4> InT3 : register(t3);                       // v47: material flags (.x; bit 3: subsurface - G-buffer 1 .x bits 0-7 its amount)
Texture2D<uint4> InT2 : register(t2);                       // v66: G-buffer render target 6 - the game's emission (game_emission)
RWTexture2D<float4> Light : register(u0);
cbuffer LightingConstants : register(b0)
{
	float4 c[85];                                       // 1348 bytes in CS_DeferredLighting
};

// The game's light (CS_DeferredLighting's StructuredBuffer<LightDesc> t14, 436 bytes; the struct type is kept in its DXIL,
// the meaning of the fields comes from how the lighting uses them)
struct LightDesc
{
	float3 cull_center;     //   0 (light culling only)
	float cull_radius;      //  12
	float3 cull_axis;       //  16
	int cull_misc;          //  28
	uint flags;             //  32 bits 0-3 type (0 point, 1 box, 2 spot), 4 / 5 stencil 32 gating, 6 planar (box), 7 diffuse,
	                        //     8 specular, 9-11 cookie kind (1-3 projected, 4-7 polar), 13 / 26 off, 14 characters
	                        //     only, 15 stencil 32 and not characters, 18 stencil 64 only, 28 not on characters, 29 second
	                        //     spot cone
	float3 color;           //  36
	float4 falloff;         //  48 saturate(z / (1 + x d + y d^2 + falloff_cubic d^3) + w)
	float3 position;        //  64 absolute world position
	float near_dist;        //  76 no light closer than this (along the axis for spots) ...
	float3 axis;            //  80 spot / box forward
	uint cookie;            //  92 cookie texture (space1 array index; 16 and up: none)
	float2 cone;            //  96 spot cone: saturate(dot(L, axis) * y + x)
	float falloff_cubic;    // 104
	int shadow_map;         // 108 shadow map (space2 index, -1 = none)
	float depth_b;          // 112 shadow map depth = (depth_a / w + depth_b) * depth_scale + shadow_fade.w, w = the depth
	float depth_a;          // 116   along the light's axis (projected / tetrahedral maps; planar maps are linear)
	float depth_scale;      // 120
	uint shadow_misc;       // 124
	float4 proj[3];         // 128 the light's matrix rows (x, y, w) on camera-relative positions
	float4 shadow_fade;     // 176 shadow distance fade scale, bias, minimum shadow, depth bias
	float2 cone2;           // 192 second spot cone (flags bit 29)
	float min_roughness;    // 200 roughness' = lerp(min_roughness, 1, roughness) for this light's specular
	float spec_scale;       // 204
	float cull_misc2;       // 208
	float unused212;        // 212
	float near_fade;        // 216 ... fading in over this distance
	float pcf_radius;       // 220
	float pcf_min;          // 224
	float pcf_range;        // 228
	uint shadow_curve;      // 232
	uint unused236;         // 236
	float4 cookie_a;        // 240 projected: uv offset (xy), array slice (z); polar: clamp (x), mip scale (y)
	float4 cookie_b;        // 256 projected: uv scale (xy); polar: distance terms
	float4 unused272[8];    // 272
	float box_x;            // 400 box window: saturate((box_x - |y| box_x) / box_soft) * saturate((box_y - |x| box_y) / box_soft)
	float box_y;            // 404
	float box_soft;         // 408
	float radius;           // 412 sphere light radius (the specular's representative point; the shadow rays' light size)
	float3 mask_axis;       // 416 point lights: 1 - smoothstep(mask_lo, mask_hi, dot(L, mask_axis))
	float mask_hi;          // 428
	float mask_lo;          // 432
};
StructuredBuffer<uint> LightIndex : register(t12);          // per 16x16 tile: up to 16 light indices (tile * 16 + i)
StructuredBuffer<LightDesc> Lights : register(t14);
Texture2DArray<float4> Cookies[16] : register(t0, space1);
SamplerState SamplerS0 : register(s0);                      // the lighting's samplers: t10, ...
SamplerState SamplerS2 : register(s2);                      // projected cookies, t18, t21
SamplerState SamplerS3 : register(s3);                      // t16, t7
SamplerState SamplerS4 : register(s4);                      // polar cookies

RWByteAddressBuffer Data : register(u0, space100);
RWByteAddressBuffer Geo : register(u1, space100);              // stream-out counters, TLAS instances, vertex regions (tlou_rt.cpp)
RaytracingAccelerationStructure Scene : register(t0, space100); // TLAS: one instance per draw (absolute world space)
ByteAddressBuffer Table : register(t1, space100);               // this frame's draw table (upload heap, see TABLE_*)
Texture2D<float4> HeapTextures[] : register(t0, space102);      // v30: the game's CBV/SRV/UAV heap (the cut-outs' masks by slot)
ByteAddressBuffer LightCounts : register(t2, space100);         // the game's t13 range (light count per tile), a root SRV:
                                                                // the lighting's own t13 descriptor reads zeros (v18)
// UE RT v19: the draw a hit names - TLOU's TLAS one instance per draw (its InstanceID); the UE adapter's (ue_game.hlsli) one
// instance per game instance, its InstanceID its first draw table row, a row per geometry of its BLAS
#ifdef UE_GAME
#define SCENE_DRAW_COMMITTED(q) ((q).CommittedInstanceID() + (q).CommittedGeometryIndex())
#define SCENE_DRAW_CANDIDATE(q) ((q).CandidateInstanceID() + (q).CandidateGeometryIndex())
#else
#define SCENE_DRAW_COMMITTED(q) ((q).CommittedInstanceID())
#define SCENE_DRAW_CANDIDATE(q) ((q).CandidateInstanceID())
#endif
cbuffer RtConstants : register(b0, space100)
{
	uint g_pass;          // cs_atrous / cs_light_filter iteration / cs_dump slot
	uint g_frame;         // frame counter: ping-pong parity, random seed, sample position
	float g_bounce;       // path-traced bounce light (1 = one physical bounce)
	float g_sky;          // sky light: the game's ambient where rays reach the open (1 = as the game lit it)
	float g_ray_length;   // near field, metres
	uint g_half_w;
	uint g_half_h;
	float g_reserved_occlusion; // v97: was g_occlusion (the Occlusion slider, removed - see below)
	float g_cone_cos;     // cosine of the sky shadow cone's half angle (the softness)
	float g_sun_cos;      // cosine of the sun cone's half angle (the sun shadows' softness)
	float g_reflect;      // how much of the path-traced reflections replace the game's (1 = all, where the rays hit)
	float g_light_shadows; // how much the ray-traced visibility darkens the game's lights (1 = fully)
	float g_light_radius; // smallest light radius for the shadow rays, metres (the Light size slider)
	uint g_light_tiles;   // entries of LightCounts (tiles beyond it have no lights)
	uint g_lights_on;     // the game's light loop is off this frame and LightCounts is bound
	float g_full;         // v19: share of the add-on's full path-traced lighting in the light buffer (1 = all of it)
	uint g_env_on;        // the game's environment probes (Cs_SpecularIbl's bindings, space 101) are bound
	uint g_motion_on;     // v20: cs_motion wrote this frame's motion plane (else the histories follow the camera only)
	float g_reserved_sun; // v97: was g_sun (the Sun slider's own sun, removed - see below)
	float g_brightness;   // v23: full path tracing's level (x its lit result)
	float g_baked;        // v28: share of the game's baked ambient in the light of rays that leave the near field (1 = all;
	                      // 0 = GBRT-style: the open sky - the game's distant probe - and only what really arrives)
	float g_contact;      // v40: contact shadows' strength on the bounce and sky light (full path tracing)
	float g_emit;         // v39: the lamps' strength (the emitters' light in full path tracing; 0 = off) - see below
	uint g_alpha_on;      // v30: HeapTextures is bound at the game's heap: the cut-outs' masks can be read
};
// v97 (user: "remove the occlusion code and slider"; "remove the sun slider and code (NOT the already stabilished DD2 sun
// one)"): no game ambient occlusion on the path-traced light (its SSAO, the characters' capsules, the materials' own, the
// large-scale ambient shadow t10.y) - the contact rays keep their own slider, the ray-traced visibility darkens the game's
// baked light fully (Occlusion's old 1) - and no add-on sun: the game's sun lit the DD2 way (the Sun slider's old 0). The
// names below stay only for dd2.hlsl (left as is), at the values that switch them off
#define g_occlusion 0.0
#define g_sun 0.0
float game_ambient_occlusion(uint2 px, uint4 g1, float2 uv)
{
	return 1.0;
}

// ---- the game's environment probes (v19; Cs_SpecularIbl's DXIL, its bindings mirrored into space 101) ----
// Cs_SpecularIbl lights the reflections (t18) from up to 250 cubemaps: each 16x16 tile has a 256-bit mask of the probes
// whose influence boxes reach it; per probe a box in view space (|q| < 1 inside, fading outside an inner box), a box
// projection of the lookup direction (parallax correction; distant probes have none) and front-to-back blending until a
// probe covers the point fully. The add-on looks up the same probes in any direction: the light of rays that leave the
// scene (the sky, the distant surroundings) - the counterpart of GBRT's sky panorama.
struct CubemapInfo
{
	float3 inner;           //   0 inner box (full weight inside), influence coordinates
	int unused12;           //  12
	float3 fade;            //  16 weight lost per unit outside the inner box
	float weight;           //  28 weight inside the inner box / of a distant probe
	float4 influence[3];    //  32 view-space position -> influence coordinates
	float4 box[3];          //  80 view-space position -> projection box coordinates
	float4 dir[3];          // 128 world direction -> projection box coordinates (xyz)
	float3 back[3];         // 176 projection box coordinates -> cubemap direction
	float3 offset;          // 212 the capture point in projection box coordinates
	int distant;            // 224 not 0: no box projection
	float3 unused228[5];    // 228
};
struct CubemapBits
{
	uint bits[8];
};
StructuredBuffer<CubemapInfo> IblInfo : register(t0, space101);
StructuredBuffer<CubemapBits> IblBits : register(t1, space101);
TextureCube<float4> IblCubes[250] : register(t8, space101);
cbuffer IblConstants : register(b0, space101)
{
	int4 ibl_c0;            // z: highest probe index (bit b of word w is probe z - (w * 32 + b)), w: tiles per row
	float4 ibl_c1;          // y: intensity
	int4 ibl_c2;
	float4 ibl_c3;          // the camera position in the probes' world space
	float4 ibl_c4;
	float4 ibl_c5[3];       // view -> world rows
};
SamplerState EnvSampler : register(s0, space100);           // the add-on's static sampler: trilinear, clamp

// Draw table: header (uint draws, uint counters to zero: first byte offset in Geo, count, TLAS instance array byte offset
// in Geo), then 32 bytes per draw from byte 128: vertex data byte offset in Geo, vertex count, counter byte offset in Geo,
// flags (bit 0: hair, bit 1: new BLAS entry, its material slots are cleared, bit 2: cut-out), BLAS GPU address (low,
// high; 0 = none),
// first material slot of the entry (0xFFFFFFFF = none), v20: the vertex data byte offset in Geo the same BLAS entry had
// in the previous frame region (converted to world positions then; 0xFFFFFFFF = none)
#define TABLE_DRAWS 128
#define TABLE_STRIDE 48         // v30: was 32
#define TABLE_UV 32             // v30: + 32 the draw's mask coordinates in Geo (float2 per vertex), + 36 its mask's heap slot
#define NO_UV 0xFFFFFFFFu
#define VERTEX_BYTES 12
#define GEO_MATERIAL 706478080u // byte offset of the material slots in Geo (k_material_base in tlou_rt.cpp)
#define GEO_MATERIAL_PBR 975372288u // v82: a second slot per triangle - metalness and specular reflectance (k_material_pbr_base)
#define NO_MATERIAL 0xFFFFFFFFu
#define NO_PREVIOUS 0xFFFFFFFFu
#define TABLE_EPOCH 16          // v36: header uint 4, the epoch's number (cs_late's tags)
#define TABLE_ADAPTER 20        // v82: header uint 5, the game adapter's switches (adapter.hlsl ADAPTER_*)
#define GEO_LATE 974913536u     // v36: one uint per stream-out counter slot (region x 16384 + slot): the epoch + 1 whose draw
                                //   in that slot the lighting before found already run (cs_late; k_late_base in tlou_rt.cpp)
#define COUNTER_BYTES 262144u   // one region's stream-out counters (16384 slots x 16 bytes, k_counter_bytes)
#define REGIONS 3u              // k_regions
#define TABLE_SOURCE 40         // v37: + 40 the mask's texture identity (its staging descriptor, hashed; 0 = unknown)
#define TABLE_KEEP 44           // v94: + 44 the draw's BLAS entry's stillness record in GEO_KEEP + 1 (0: none)
#define GEO_KEEP 1042481152u    // v94: per BLAS entry (k_keep_base in tlou_rt.cpp), KEEP_RECORD_BYTES: the frames it has stood
#define KEEP_RECORD_BYTES 208u  //   still and the anchors taken (uint4), then KEEP_SAMPLES anchors - where its sample vertices
#define KEEP_SAMPLES 16u        //   were when the stillness began (float3 each)
#define KEEP_STILL_FRAMES 60u   // v94: frames an object stands still before its geometry may outlive its draws
#define DRAW_KEPT 16u           // v94: table flags bit 4 - an entry the game no longer draws, listed for the shadow rays (+ bit 3)
#define DRAW_NEW_RECORD 32u     // v94: table flags bit 5 - the entry's record is new (cs_convert starts it)
#define GEO_MASK_SLOT 975110144u  // v37: per listed draw, the heap slot its mask is read from this frame (cs_mask) ...
#define GEO_MASK_HASH 975175680u  // ... and this frame's texture identity -> slot, from the cut-outs the game drew on time
#define MASK_HASH_SLOTS 8192u   //   (8 bytes each: identity, slot; cleared by cs_late)
#define STALE_MASK 0xFFFFFFFEu  // a late or retained cut-out whose texture no draw of this frame names: the light goes through
#define GEO_MASK_VOTES 975241216u // v55: per listed draw, this frame's votes on its mask by the camera rays (cs_motion): agree,
#define MASK_NO_VOTE 0x80000000u  //   disagree (top bit: a late or retained draw, not in this frame's depth buffer - no votes)
#define MASK_VOTES_MIN 24.0     //   votes before a verdict, and the share of disagreeing ones above which the rays let the
#define MASK_DISAGREE 0.35      //   light through the draw (its alpha test is not the one modelled: another texture, channel,
                                //   coordinates or cutoff)
#ifdef UE_GAME
// UE RT v19: the UE adapter's Geo buffer (ue_rt.cpp k_ue_geo_bytes) - no stream-out regions (the game's own vertex buffers
// hold the geometry: ue_game.hlsli ue_hit_normal), its regions packed from 0: 16M material slots of each kind, then the
// masks' and the stillness records' (the UE adapter writes none; kept for the passes that name them)
#undef GEO_MATERIAL
#undef GEO_MATERIAL_PBR
#undef GEO_LATE
#undef GEO_KEEP
#undef GEO_MASK_SLOT
#undef GEO_MASK_HASH
#undef GEO_MASK_VOTES
#define GEO_MATERIAL 0u
#define GEO_MATERIAL_PBR 67108864u
#define GEO_MASK_SLOT 134217728u
#define GEO_MASK_HASH 134283264u
#define GEO_MASK_VOTES 134348800u
#define GEO_LATE 134479872u
#define GEO_KEEP 134676480u
#endif

// Data buffer: a 1024-byte frame block, PLANES half-resolution planes of float4, the full-resolution history (packed
// R11G11B10 per pixel, 512-aligned), then four copy-out images for t4', t5' (R11G11B10), t10' (RGBA8) and t18'
// (R11G11B10), rows of copy_pitch() bytes, 512-aligned: the placed footprints CopyTextureRegion reads; then the game's
// lights' full-resolution planes (v18: LIGHT_DWORDS planes of one dword per pixel, two of float4), then the motion plane
// (v20: MOTION_BYTES per full-resolution pixel, motion_offset). tlou_rt.cpp computes the same offsets.
#define FRAME_BYTES 1280        // v82: was 1024 (+ FRAME_AUTO_PBR)
#define FRAME_VP_CUR 0          // forward view-projection of the camera-relative space (rows), this frame
#define FRAME_VP_PREV 64        // ... and the previous frame
#define FRAME_CAM_CUR 128       // camera position (world), this frame
#define FRAME_CAM_PREV 144      // ... and the previous frame
#define FRAME_INFO 160          // x = frames set up so far
#define FRAME_MATERIAL_MEAN 176 // mean albedo of last frame's material samples (rgb): triangles not seen yet use it
#define FRAME_AUTO_PBR 1024     // v82: the auto-PBR layer's estimates against the game's own values (adapter.hlsl cs_adapt): the
                                //   sums of |error| x 1024 (roughness, metalness, F0 x 10, 1 - the normals' cosine), then the
                                //   pixels compared (x of the second uint4); zeroed by cs_setup, read back with the stats
#define FRAME_PT 1056           // v83: NVIDIA ReSTIR PT's counters (rtxdi_pt.hlsl): pixels, samples from last frame, decorrelated,
#define FRAME_REGIR 1104        // v90: ReGIR's surface cells' counters (regir_cells.hlsli): cells in use, bounce hits served by a
                                //   cell / by the onion, insertions refused (the table full) - after FRAME_PT's 48 bytes
#define FRAME_KEPT (FRAME_REGIR + 24)   // v94: kept entries that held still long enough to be in the scene (cs_convert; that
                                        //   block's spare word: zeroed by cs_setup, read back with it)
                                //   firefly-flagged; then the final samples' summed age, reconnected at their hit, replayed, sky;
                                //   v84 (48 bytes): then the paths taken from a neighbour, CGNS's neighbours found; v86: the
                                //   bounce noise (re9.hlsl cs_re9_accum, either renderer: relative deviation sum x 1000, pixels)
#define FRAME_C10_PREV 1012    // v36: the last converted frame's depth linearisation (c[10].xy) ...
#define FRAME_INV_PREV 448      // ... and its clip -> camera-relative rows (c[19..22]), written by cs_late after the conversion:
                                //   the next frame's cs_convert places the draws that ran before this lighting with them
#define SKY_A4 400              // v35: the add-on's sky (cs_sky, kept over the frames): the outdoor flat ambient (rgb), w =
                                //   the frames it has been fed (0: no sky yet) ...
#define SKY_A5 416              // ... its directional lobe (rgb), w = frames since an open pixel was last seen ...
#define SKY_AXIS 432            // ... the lobe's direction (xyz)
#define LIGHT_MASK 512          // 32 uints: the lights the tiles named this frame (cs_light_census), zeroed by cs_setup
#define LIGHT_LIST_COUNT 1008   // v23: how many lights cs_light_census listed this frame (zeroed by cs_setup) ...
#define LIGHT_LIST 640          // ... and their indices (the bounce hits' lamps draw from it)
#define LIGHT_LIST_MAX 92       // v37: was 96 (its last 16 bytes hold LIGHT_LIST_COUNT and FRAME_C10_PREV)
#define STATS 192               // 52 uint counters (v37: was 48), zeroed by cs_setup, copied to the add-on's readback buffer every frame:
#define STAT_TRIANGLES 0        //   triangles of complete draws (hit-able)
#define STAT_INCOMPLETE 1       //   draws whose stream-out counter did not match (hidden this frame)
#define STAT_ON_SURFACE 2       //   camera rays hitting within 1% of the depth buffer's distance
#define STAT_IN_FRONT 3         //   ... hitting before it (alpha-tested cut-outs: leaves, hair, fences)
#define STAT_BEHIND 4           //   ... hitting up to 5% behind it
#define STAT_MISSED 5           //   ... hitting nothing: geometry missing
#define STAT_LATE 6             //   v36: draws placed with the frame before's camera (cs_late; v7-v35: sky hits, always 0)
#define STAT_MATERIAL_KNOWN 7   //   bounce ray hits whose triangle's material is cached ...
#define STAT_MATERIAL_HITS 8    //   ... of all bounce ray hits
#define STAT_ALBEDO_SUM 9       //   9-11: albedo of this frame's material samples (sum x 1024), 12: their count
#define STAT_ALBEDO_COUNT 12
#define STAT_MATERIAL_WRITES 13 //   pixels whose triangle got a material sample (cs_material)
#define STAT_LIGHTS 14          //   v18: lights the tiles name (each counted once, indices below 1024) ...
#define STAT_LIGHT_POINT 15     //   ... point lights, ...
#define STAT_LIGHT_SPOT 16      //   ... spot lights (and other types), ...
#define STAT_LIGHT_BOX 17       //   ... box lights, ...
#define STAT_LIGHT_SHADOWMAP 18 //   ... lights the game gave a shadow map, ...
#define STAT_LIGHT_COOKIE 19    //   ... lights with a cookie (projected texture), ...
#define STAT_LIGHT_RADIUS 20    //   ... the sum of their radii (mm) ...
#define STAT_LIGHT_RADIUS_MAX 21 //  ... and the largest (mm)
#define STAT_LIGHT_TILES 22     //   tiles with lights
#define STAT_LIGHT_MAX_INDEX 23 //   largest light index + 1
#define STAT_LIT_PIXELS 24      //   pixels the game's lights reach
#define STAT_LIGHT_EVALS 25     //   lights evaluated at those pixels (sum)
#define STAT_LIGHT_BLOCKED 26   //   lit pixels whose shadow rays were mostly blocked (visibility below 0.5)
#define STAT_LIGHT_CHARACTER 27 //   lit character pixels (G-buffer 1 .w bit 0)
#define STAT_MOTION_MOVING 28   //   v20 (cs_motion): pixels on a triangle that moved since last frame ...
#define STAT_MOTION_STATIC 29   //   ... on a triangle that kept its place ...
#define STAT_MOTION_CAMERA 30   //   ... without a triangle last frame (new, a cut-out in front, too far): the camera only
#define STAT_SHADOW_NEAR_MAX 31 //   v21: the farthest shadow-map near plane of the named lights (mm) ...
#define STAT_SHADOW_NEAR_MIN 32 //   ... and the nearest (stored complemented: ~mm, 0 = none)
#define STAT_CUTOUT_DRAWS 33    //   v22: draws with the cut-out instance mask
#define STAT_SHARC_QUERIES 34   //   v23 SHaRC: lookups at hits the camera did not see ...
#define STAT_SHARC_FOUND 35     //   ... that found light in the cache,
#define STAT_SHARC_WRITES 36    //   estimates the rays wrote, ...
#define STAT_SHARC_INJECTED 37  //   ... pixels the camera's view wrote,
#define STAT_SHARC_FULL 38      //   samples dropped (bucket full),
#define STAT_SHARC_LIVE 39      //   live entries after the resolve
#define STAT_SKY_ESCAPED 40     // v28: camera rays that left the near field with Baked light below 1 ...
#define STAT_SKY_FOUND 41       //   ... and found the game's distant (sky) probe in their tile
#define STAT_MASK_VOTES 40      // v55 (40 / 41 free since v51 removed cs_trace_rt): cut-out hits the camera rays voted on ...
#define STAT_MASK_DISAGREE 41   //   ... and those whose mask disagreed with the depth buffer (cs_motion)
#define STAT_ALPHA_CALLS 42     // v31: cut-out hits tested (one in 16, by triangle) ...
#define STAT_SKY_OPEN 43        // v35: cs_sky - this frame's open-sky pixels (weight x 100) ...
#define STAT_SKY_LUM 44         //   ... the sky's luminance toward the zenith (x 1000) ...
#define STAT_SKY_AGE 45         //   ... frames since it was last updated (0 = this frame)
#define STAT_ALPHA_FALLBACK 46  // v31-v34 four reasons, v35 one: no mask, heap off, coordinates incomplete, not a texture
#define STAT_ALPHA_SOLID 47     //   ... the texel solid (of the rest: holes)
#define STAT_MASK_REMAPPED 48   // v37: late or retained cut-out draws given this frame's slot for their mask's texture ...
#define STAT_MASK_STALE 49      //   ... and those whose texture no draw of this frame names (light through)
#define STAT_EMIT_PIXELS 50     // v39: emissive pixels before the lighting (cs_emit) ...
#define STAT_EMIT_LIST 51       //   ... and the emitters listed from them (cs_emit_list)
#define STAT_COUNT 52
#define PLANE_TRACE 0          // this frame's rays: bounce radiance B (rgb), visibility V of the flat ambient (1 = escaped)
#define PLANE_TRACE_S 1         // ... the sky shadow S (x: visibility toward the ambient's dominant direction), sun (y)
#define PLANE_TRACE_R 2         // ... the reflection's light (rgb) and whether it hit something (w)
#define PLANE_ACCUM 3           // 3, 4: accumulated (B, V) (ping-pong by frame parity)
#define PLANE_MOMENTS 5         // 5, 6: accumulated S, second moments of V and S, sample count (ping-pong)
#define PLANE_SUN 7             // 7, 8: accumulated sun visibility, its second moment (ping-pong)
#define PLANE_ACCUM_R 9         // 9, 10: accumulated reflections (ping-pong)
#define PLANE_GEO 11            // 11, 12: view depth, normal (ping-pong by frame parity)
#define PLANE_DENOISE 13        // 13, 14: filtered (B, V) ping-pong, the result ends up in plane 13 (odd pass count)
#define PLANE_DENOISE_S 15      // 15, 16: filtered (S, variance of V, variance of S) ping-pong, result in plane 15
#define PLANE_DENOISE_SUN 17    // 17, 18: filtered (sun, its variance) ping-pong, result in plane 17
#define PLANE_DENOISE_R 19      // 19, 20: filtered reflections ping-pong, result in plane 19
#define PLANE_TRACE_E 21        // v19: this frame's escaped cosine rays' light (rgb): the sky / environment E
#define PLANE_ACCUM_E 22        // 22, 23: accumulated E (ping-pong)
#define PLANE_DENOISE_E 24      // 24, 25: filtered E ping-pong (the (B, V) weights), result in plane 24
#define PLANES 26
#define ENV_DIFFUSE_MIP 5.0     // probe mip for the cosine rays (the probes' mips are roughness * 7): prefiltered, less noise
#define SUN_RAY_LENGTH 500.0    // metres: sun shadows from far casters too (as far as the game drew them)
#define HIT_SUN_RAY_LENGTH 100.0
#define REFLECTION_RAY_LENGTH 200.0
#define ATROUS_PASSES 3         // steps 1, 2, 4 (tlou_rt.cpp runs this many; v13-v39: 5, steps up to 16 - ReSTIR GI's
                                //   samples need less, and the indirect light's shadows stay)
#define CONTACT_RANGE 0.5       // v40: metres a contact ray looks for an occluder (v41: was 1, with a linear falloff)
#define SKY_LOBE_SHARE 0.72     // v41: the ambient's directional lobe, share of the sky light facing it (v10 capture: 72%)
#define HEMI_RAYS 4             // cosine rays per half-resolution pixel and frame (B, V)
#define SHADOW_RAYS 2           // rays toward the dominant direction per half-resolution pixel and frame (S)
#define MASK_SCENE 0x01u        // TLAS instance masks (cs_convert): the scene ...
#define MASK_HAIR 0x02u         // ... and hair strands (alpha-tested cards), left out of the near-field rays
#define MASK_CUTOUT 0x04u       // v22: other alpha-tested cards (grass, leaves, fences: the prepass shader binds a sampler),
                                // solid sheets in the scene: only the camera rays (MASK_DRAWN) see them, as hair
#define MASK_KEPT 0x08u         // v94: still objects the game stopped drawing (tlou_rt.cpp keeps their BLASes "Remember
                                //   geometry" seconds): only the shadow rays see them (trace_visible, sun_visible)
#define MASK_DRAWN (MASK_SCENE | MASK_HAIR | MASK_CUTOUT)   // v94: what the game drew (the camera and material rays; was 0xFF)
#define HIDDEN_ALBEDO 0.1       // a hit the camera did not see: an occluder lit like this point with this albedo
#define LIGHT_SUM 0             // v18 full-resolution dword planes: the tile's lights unshadowed (packed R11G11B10),
#define LIGHT_RAW 1             //   this frame's visibility sample (-1: no light here),
#define LIGHT_NRM 2             //   2, 3: normal (octahedral 16+16, ping-pong with the accumulation),
#define LIGHT_FS 4              //   4, 5: filtered visibility (cs_light_filter ping-pong, the result ends up in 4),
#define LIGHT_FV 6              //   6, 7: its variance
#define LIGHT_DWORDS 8
#define LIGHT_ACC 0             // float4 planes 0, 1: accumulated visibility, its second moment, frames, view depth
#define LIGHT_HISTORY 16.0      // frames of visibility history at most
#define LIGHT_FILTER_PASSES 3   // steps 1, 2, 4 (tlou_rt.cpp runs this many)
#define LIGHT_SKIP 0.35         // metres: occluders this close to a light are its own lamp or fixture (no shadow); v63: was
                                //   0.15 - hanging lanterns and basket lamps are wider, their frames blocked their own lights
#define LIGHT_RAYS 4            // v20: shadow rays per lit pixel and frame
#define MOTION_MAX 0.5          // metres a triangle may move between frames; farther: another object's entry (camera only)
#define MOTION_BYTES 16         // motion plane per full-resolution pixel: previous (ndc.xy, w), PIXEL_* flags (v21)
#define PIXEL_TRIANGLE 1u       //   the camera ray found the surface's triangle ...
#define PIXEL_HAIR 2u           //   ... and it belongs to a hair draw (strand cards, dithered in the G-buffer)
#define PIXEL_CUTOUT_IN_FRONT 4u //  a cut-out card (leaf, hair, fence) lies in front of the surface in the scene

uint align_up(uint v, uint a)
{
	return (v + a - 1) & ~(a - 1);
}
uint2 full_size()
{
	return uint2(c[8].xy + 0.5);
}
uint plane_offset(uint plane)
{
	return FRAME_BYTES + plane * g_half_w * g_half_h * 16;
}
uint history_offset()
{
	return align_up(plane_offset(PLANES), 512);
}
uint copy_pitch()
{
	return align_up(full_size().x * 4, 256);
}
uint copy_offset(uint image)
{
	return align_up(history_offset() + full_size().x * full_size().y * 4, 512) + image * align_up(copy_pitch() * full_size().y, 512);
}
uint light_dword(uint plane)                                // v18 planes after the four copy-out images
{
	return copy_offset(4) + plane * full_size().x * full_size().y * 4;
}
uint light_quad(uint plane)
{
	return light_dword(LIGHT_DWORDS) + plane * full_size().x * full_size().y * 16;
}
uint motion_offset()                                        // v20: after the lights' two float4 planes
{
	return light_quad(2);
}
float4 load_px(uint plane, uint2 hp)
{
	return asfloat(Data.Load4(plane_offset(plane) + (hp.y * g_half_w + hp.x) * 16));
}
void store_px(uint plane, uint2 hp, float4 v)
{
	Data.Store4(plane_offset(plane) + (hp.y * g_half_w + hp.x) * 16, asuint(v));
}
float4x4 load_matrix(uint offset)
{
	return float4x4(asfloat(Data.Load4(offset)), asfloat(Data.Load4(offset + 16)), asfloat(Data.Load4(offset + 32)), asfloat(Data.Load4(offset + 48)));
}

// R11G11B10_FLOAT (via half floats, as MiniEngine's PixelPacking_R11G11B10; checked against an exact encoder in
// r11g11b10_check.py: within rounding, clamped to 64512 so nothing rounds up to inf, negatives to 0)
uint pack_r11g11b10(float3 rgb)
{
	rgb = clamp(rgb, 0.0, asfloat(0x477C0000u));
	const uint r = ((f32tof16(rgb.x) + 8) >> 4) & 0x7FFu;
	const uint g = ((f32tof16(rgb.y) + 8) << 7) & 0x3FF800u;
	const uint b = ((f32tof16(rgb.z) + 16) << 17) & 0xFFC00000u;
	return r | g | b;
}
float3 unpack_r11g11b10(uint v)
{
	return float3(f16tof32((v << 4) & 0x7FF0u), f16tof32((v >> 7) & 0x7FF0u), f16tof32((v >> 17) & 0x7FE0u));
}

float3 decode_normal(uint4 g)
{
	const uint u = (g.z & 0xFFu) | (g.w & 0xF00u);
	const uint v = (g.z >> 8) | ((g.w >> 4) & 0x700u);
	float x = u * (2.0 / 4095.0) - 1.0;
	float y = v * (2.0 / 2047.0) - 1.0;
	const float z = 1.0 - abs(x) - abs(y);
	if (z < 0.0)
	{
		const float ox = x;
		x = (1.0 - abs(y)) * (ox >= 0.0 ? 1.0 : -1.0);
		y = (1.0 - abs(ox)) * (y >= 0.0 ? 1.0 : -1.0);
	}
	return normalize(float3(x, y, z));
}

float3 decode_albedo(uint4 g)
{
	return pow(float3(g.x & 0xFFu, g.x >> 8, g.y & 0xFFu) / 255.0, 2.2);
}

// The ambient lobe's dominant direction D from G-buffer 1 .z, as CS_DeferredLighting decodes it: octahedral, x from the
// low byte, y from the high byte (2/255 steps), z = 1 - |x| - |y| kept when the lower half folds over
float3 decode_ambient_direction(uint4 g1)
{
	float x = (g1.z & 0xFFu) * (2.0 / 255.0) - 1.0;
	float y = ((g1.z >> 8) & 0xFFu) * (2.0 / 255.0) - 1.0;
	const float z = 1.0 - abs(x) - abs(y);
	if (z < 0.0)
	{
		const float ox = x;
		x = (1.0 - abs(y)) * (ox >= 0.0 ? 1.0 : -1.0);
		y = (1.0 - abs(ox)) * (y >= 0.0 ? 1.0 : -1.0);
	}
	return normalize(float3(x, y, z));
}

// ---- material cache (v14) ----
// A slot holds albedo as 8 bits per channel with gamma 2.2 (the G-buffer's own encoding) and roughness as 1 + 254 * r in
// the top byte, so a seen triangle is never 0
uint pack_material(float3 albedo, float roughness)
{
	const uint3 a = uint3(pow(saturate(albedo), 1.0 / 2.2) * 255.0 + 0.5);
	return a.x | (a.y << 8) | (a.z << 16) | ((uint(saturate(roughness) * 254.0 + 0.5) + 1u) << 24);
}
float4 unpack_material(uint m)
{
	return float4(pow(float3(m & 0xFFu, (m >> 8) & 0xFFu, (m >> 16) & 0xFFu) / 255.0, 2.2), float((m >> 24) - 1u) / 254.0);
}
// The raw slot of a hit triangle (draw = TLAS instance ID, primitive = BLAS primitive index); 0 = not seen yet
uint material_slot(uint draw, uint primitive)
{
	const uint base = Table.Load(TABLE_DRAWS + draw * TABLE_STRIDE + 24);
	return base != NO_MATERIAL ? Geo.Load(GEO_MATERIAL + (base + primitive) * 4) : 0u;
}
// The cached material of a hit triangle: linear albedo (rgb) and roughness (w); w < 0 for a triangle not seen yet, which
// gets the mean albedo of last frame's samples
float4 hit_material(uint draw, uint primitive)
{
	const uint m = material_slot(draw, primitive);
	return m != 0 ? unpack_material(m) : float4(asfloat(Data.Load3(FRAME_MATERIAL_MEAN)), -1.0);
}
// v82: the triangle's second slot (GEO_MATERIAL_PBR, learnt with the first by cs_material from the game adapter's
// surfaces - the game's values or the auto-PBR layer's): metalness in the low byte, the specular reflectance byte (F0 x
// 255 / 0.08 for non-metals, as the adapter's) above it, bit 16 once seen. Bounce hits shade with it (v44-v81: metalness 0
// and F0 0.04 at every hit - metals reflected as grey plastic)
uint pack_material_pbr(float metallic, float spec)
{
	return uint(saturate(metallic) * 255.0 + 0.5) | (min(uint(max(spec, 0.0) + 0.5), 255u) << 8) | 0x10000u;
}
uint blend_material_pbr(uint old, float metallic, float spec)
{
	if ((old & 0x10000u) == 0u)
		return pack_material_pbr(metallic, spec);
	return pack_material_pbr(lerp((old & 0xFFu) / 255.0, metallic, 0.25), lerp(float((old >> 8) & 0xFFu), spec, 0.25));
}
// The hit triangle's metalness (x) and dielectric F0 (y); a triangle not seen yet: 0 and 0.04
float2 hit_material_pbr(uint draw, uint primitive)
{
	const uint base = Table.Load(TABLE_DRAWS + draw * TABLE_STRIDE + 24);
	const uint m = base != NO_MATERIAL ? Geo.Load(GEO_MATERIAL_PBR + (base + primitive) * 4) : 0u;
	return (m & 0x10000u) != 0u ? float2((m & 0xFFu) / 255.0, float((m >> 8) & 0xFFu) * (0.08 / 255.0)) : float2(0.0, 0.04);
}
// The geometric normal of a hit triangle from this frame's vertices (absolute world space). The winding is not known:
// callers turn it toward the ray.
#ifdef UE_GAME
#include "../ue_game.hlsli"
#endif
float3 hit_normal(uint draw, uint primitive)
{
#ifdef UE_GAME
	return ue_hit_normal(draw, primitive);             // UE RT v19: the game's vertex buffers, the instance's transform
#else
	const uint a = Table.Load(TABLE_DRAWS + draw * TABLE_STRIDE) + primitive * 3 * VERTEX_BYTES;
	const float3 p0 = asfloat(Geo.Load3(a)), p1 = asfloat(Geo.Load3(a + VERTEX_BYTES)), p2 = asfloat(Geo.Load3(a + 2 * VERTEX_BYTES));
	const float3 n = cross(p1 - p0, p2 - p0);
	const float len = length(n);
	return len > 1e-12 ? n / len : float3(0.0, 1.0, 0.0);
#endif
}

float view_z(float d)
{
	return c[10].y / (d - c[10].x);
}

float3 camera_relative_position(float2 pixel, float d)
{
	const float2 uv = (pixel + 0.5) * c[8].zw;
	const float4 p = (uv.x * 2.0 - 1.0) * c[19] + ((1.0 - uv.y) * 2.0 - 1.0) * c[20] + d * c[21] + c[22];
	return p.xyz / p.w;
}

// The full-resolution pixel a half-resolution pixel stands for: v21 the front-most of its 2x2 block (smallest depth), the
// same surface every frame. v13-v20 cycled through the block, which on hair and cut-out leaves switched between the strand
// and the skin, the leaf and the wall behind, so the history was rejected every frame (the flicker there).
uint2 sample_pixel(uint2 hp)
{
	const uint2 last = full_size() - 1, base = hp * 2;
	uint2 best = min(base, last);
	float nearest = Depth[best];
	[unroll]
	for (uint k = 1; k < 4; ++k)
	{
		const uint2 q = min(base + uint2(k & 1, k >> 1), last);
		const float d = Depth[q];
		if (d < nearest)
		{
			nearest = d;
			best = q;
		}
	}
	return best;
}
// The material cache still cycles through the block (it learns the triangles under all four pixels over the frames)
uint2 cycle_pixel(uint2 hp)
{
	return min(hp * 2 + uint2(g_frame & 1, (g_frame >> 1) & 1), full_size() - 1);
}

// v24: the surface's true orientation at a pixel, from its neighbours' depth (the G-buffer normal carries the normal maps:
// cloth folds, wood grain - v19-v23 decided the sun's side by it, and the details turned away stored "lit", which the
// filters spread over the room). Per axis the neighbour nearer in position (not across an edge); facing the camera; the
// G-buffer normal where neither neighbour is a surface.
float3 geometric_normal(uint2 px, float3 p, float3 fallback)
{
	const uint2 last = full_size() - 1;
	float3 axes[2];
	[unroll]
	for (uint a = 0; a < 2; ++a)
	{
		const uint2 step = a == 0 ? uint2(1, 0) : uint2(0, 1);
		const uint2 qa = min(px + step, last), qb = px - min(px, step);
		const float da = Depth[qa], db = Depth[qb];
		const bool valid_a = da < 1.0 && any(qa != px), valid_b = db < 1.0 && any(qb != px);
		if (!valid_a && !valid_b)
			return fallback;
		const float3 pa = camera_relative_position(float2(qa), da) - p, pb = p - camera_relative_position(float2(qb), db);
		axes[a] = valid_a && (!valid_b || dot(pa, pa) < dot(pb, pb)) ? pa : pb;
	}
	const float3 n = cross(axes[0], axes[1]);
	if (dot(n, n) < 1e-14)
		return fallback;
	const float3 u = normalize(n);
	return dot(u, p) > 0.0 ? -u : u;
}

// ---- v76: the game adapter (adapter.hlsl): this frame's and last frame's surfaces (RTXDI's bridge reads them); TLOU's
// side below ----
// v82: the surface channels TLOU's G-buffer holds (adapter.hlsl SURFACE_HAS_*): all of them - the auto-PBR layer estimates
// none (it measures itself against them, and stands in for all with "Force auto-PBR")
#define GAME_SURFACE_CHANNELS 15u
#include "adapter.hlsl"

// TLOU's surface: its G-buffer as CS_DeferredLighting decodes it (t0, t1: the layouts at the top), for its lights the
// capsules' occlusion (t16) and the stencil (t23); the geometric normal from the depth buffer's neighbours
surface_record game_surface(uint2 px, float d)
{
	const uint4 g0 = GBuffer0[px], g1 = GBuffer1[px];
	surface_record r;
	r.p = d < 1.0 ? camera_relative_position(float2(px), d) : 0.0;
	r.view_depth = d < 1.0 ? view_z(d) : 0.0;
	r.n = decode_normal(g0);
	r.ng = d < 1.0 ? geometric_normal(px, r.p, r.n) : r.n;
	r.albedo = decode_albedo(g0);
	r.roughness = (g0.w & 0xFFu) / 255.0;
	r.metallic = (g1.y & 0x2000u) != 0 ? ((g1.y >> 8) & 31u) / 31.0 : 0.0;
	r.spec = float((g0.y >> 8) & 0xFFu);
	r.flags = (g1.w & 1u) != 0 ? SURFACE_CHARACTER : 0u;
	r.ao = ((g1.w >> 2) & 63u) / 63.0;
	r.game0 = f32tof16(min(((g1.x >> 8) & 63u) / 63.0, InT16.SampleLevel(SamplerS3, (float2(px) + 0.5) * c[8].zw, 0.0).x));
	r.game1 = InT23[px].x;
	return r;
}
// v82: TLOU's albedo at a pixel (the auto-PBR layer reads the neighbourhood)
float3 game_albedo(uint2 px)
{
	return decode_albedo(GBuffer0[px]);
}

uint pcg(uint v)
{
	const uint state = v * 747796405u + 2891336453u;
	const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
float3 random3(uint2 p, uint frame)
{
	const uint h1 = pcg(p.x + pcg(p.y + pcg(frame)));
	const uint h2 = pcg(h1);
	return float3(h1 & 0xFFFFu, h1 >> 16, h2 & 0xFFFFu) / 65536.0;
}

float3 cosine_direction(float3 n, float2 xi)
{
	const float r = sqrt(xi.x);
	const float phi = 6.2831853 * xi.y;
	const float3 t = normalize(abs(n.y) < 0.999 ? cross(n, float3(0.0, 1.0, 0.0)) : cross(n, float3(1.0, 0.0, 0.0)));
	const float3 b = cross(n, t);
	return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - xi.x)));
}

// Uniform direction within the cone of half angle acos(cos_max) around axis
float3 cone_direction(float3 axis, float cos_max, float2 xi)
{
	const float cos_t = lerp(1.0, cos_max, xi.x);
	const float sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
	const float phi = 6.2831853 * xi.y;
	const float3 t = normalize(abs(axis.y) < 0.999 ? cross(axis, float3(0.0, 1.0, 0.0)) : cross(axis, float3(1.0, 0.0, 0.0)));
	const float3 b = cross(axis, t);
	return normalize(t * (sin_t * cos(phi)) + b * (sin_t * sin(phi)) + axis * cos_t);
}

// Low-discrepancy 2D points (R2 sequence), rotated per pixel (Cranley-Patterson): the rays of a pixel cover the
// hemisphere evenly within a frame and across frames
float2 sequence(uint i, float2 rotation)
{
	return frac(float2(0.7548776662466927, 0.5698402909980532) * float(i & 4095u) + rotation);
}

float4x4 inverse(float4x4 m)
{
	const float b00 = m[0][0] * m[1][1] - m[0][1] * m[1][0], b01 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
	const float b02 = m[0][0] * m[1][3] - m[0][3] * m[1][0], b03 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
	const float b04 = m[0][1] * m[1][3] - m[0][3] * m[1][1], b05 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
	const float b06 = m[2][0] * m[3][1] - m[2][1] * m[3][0], b07 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
	const float b08 = m[2][0] * m[3][3] - m[2][3] * m[3][0], b09 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
	const float b10 = m[2][1] * m[3][3] - m[2][3] * m[3][1], b11 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
	const float inv_det = 1.0 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
	return float4x4(
		m[1][1] * b11 - m[1][2] * b10 + m[1][3] * b09, m[0][2] * b10 - m[0][1] * b11 - m[0][3] * b09, m[3][1] * b05 - m[3][2] * b04 + m[3][3] * b03, m[2][2] * b04 - m[2][1] * b05 - m[2][3] * b03,
		m[1][2] * b08 - m[1][0] * b11 - m[1][3] * b07, m[0][0] * b11 - m[0][2] * b08 + m[0][3] * b07, m[3][2] * b02 - m[3][0] * b05 - m[3][3] * b01, m[2][0] * b05 - m[2][2] * b02 + m[2][3] * b01,
		m[1][0] * b10 - m[1][1] * b08 + m[1][3] * b06, m[0][1] * b08 - m[0][0] * b10 - m[0][3] * b06, m[3][0] * b04 - m[3][1] * b02 + m[3][3] * b00, m[2][1] * b02 - m[2][0] * b04 - m[2][3] * b00,
		m[1][1] * b07 - m[1][0] * b09 - m[1][2] * b06, m[0][0] * b09 - m[0][1] * b07 + m[0][2] * b06, m[3][1] * b01 - m[3][0] * b03 - m[3][2] * b00, m[2][0] * b03 - m[2][1] * b01 + m[2][2] * b00) * inv_det;
}

// ---- passes ----

[numthreads(1, 1, 1)]
void cs_setup()
{
	[unroll]
	for (uint i = 0; i < 4; ++i)
		Data.Store4(FRAME_VP_PREV + i * 16, Data.Load4(FRAME_VP_CUR + i * 16));
	Data.Store4(FRAME_CAM_PREV, Data.Load4(FRAME_CAM_CUR));
	const float4x4 vp = inverse(float4x4(c[19], c[20], c[21], c[22]));
	[unroll]
	for (uint k = 0; k < 4; ++k)
		Data.Store4(FRAME_VP_CUR + k * 16, asuint(vp[k]));
	Data.Store4(FRAME_CAM_CUR, asuint(float4(c[9].xyz, 1.0)));
	Data.Store4(FRAME_INFO, uint4(Data.Load4(FRAME_INFO).x + 1, 0, 0, 0));
	const uint4 albedo = Data.Load4(STATS + STAT_ALBEDO_SUM * 4);   // last frame's material samples: sum r, g, b, count
	if (albedo.w >= 64)
		Data.Store4(FRAME_MATERIAL_MEAN, asuint(float4(float3(albedo.xyz) / (1024.0 * float(albedo.w)), 0.0)));
	[unroll]
	for (uint z = 0; z < STAT_COUNT / 4; ++z)
		Data.Store4(STATS + z * 16, uint4(0, 0, 0, 0));
	[unroll]
	for (uint m = 0; m < 8; ++m)
		Data.Store4(LIGHT_MASK + m * 16, uint4(0, 0, 0, 0));
	Data.Store(LIGHT_LIST_COUNT, 0u);
	Data.Store4(FRAME_AUTO_PBR, uint4(0, 0, 0, 0));         // v82
	Data.Store4(FRAME_AUTO_PBR + 16, uint4(0, 0, 0, 0));
	Data.Store4(FRAME_PT, uint4(0, 0, 0, 0));               // v83
	Data.Store4(FRAME_PT + 16, uint4(0, 0, 0, 0));
	Data.Store4(FRAME_PT + 32, uint4(0, 0, 0, 0));          // v84
	Data.Store4(FRAME_REGIR, uint4(0, 0, 0, 0));            // v90
	Data.Store4(FRAME_REGIR + 16, uint4(0, 0, 0, 0));       // v93: the visibility cache's entries, rays recorded
}

// Resets the stream-out counters this frame's region used, after cs_convert read them (v27; v7-v26 reset the other region's
// before the conversion, wiping the counters of draws the game recorded late for the next frame but ran before this lighting)
[numthreads(64, 1, 1)]
void cs_zero(uint3 id : SV_DispatchThreadID)
{
	const uint4 h = Table.Load4(0);
	if (id.x < h.z)
		Geo.Store4(h.y + id.x * 16, uint4(0, 0, 0, 0));    // v30: the coordinates' counter (+ 8) too
}

bool finite_and_near(float3 p)
{
	const uint3 e = asuint(p) & 0x7F800000u;                // integer test: not NaN / inf whatever the float model
	return all(e != 0x7F800000u) && all(abs(p) < 1.0e7);
}

// v37: this frame's texture identity -> heap slot (open addressing; identities are never 0)
uint mask_hash(uint source)
{
	return (source * 2654435761u) >> 19;                    // 13 bits: MASK_HASH_SLOTS
}
void mask_insert(uint source, uint slot)
{
	const uint h = mask_hash(source);
	for (uint probe = 0; probe < 16; ++probe)
	{
		const uint at = GEO_MASK_HASH + ((h + probe) & (MASK_HASH_SLOTS - 1u)) * 8u;
		uint prev;
		Geo.InterlockedCompareExchange(at, 0u, source, prev);
		if (prev == 0u)
		{
			Geo.Store(at + 4, slot);
			return;
		}
		if (prev == source)
			return;
	}
}
uint mask_lookup(uint source)
{
	const uint h = mask_hash(source);
	for (uint probe = 0; probe < 16; ++probe)
	{
		const uint at = GEO_MASK_HASH + ((h + probe) & (MASK_HASH_SLOTS - 1u)) * 8u;
		const uint key = Geo.Load(at);
		if (key == source)
			return Geo.Load(at + 4);
		if (key == 0u)
			break;
	}
	return STALE_MASK;
}

// v94: a BLAS entry's stillness (GEO_KEEP), by thread 0 of cs_convert once its group has converted the draw: the first
// vertices of KEEP_SAMPLES of its triangles against their anchors, where they were when the stillness began (2 mm + 0.02%
// of the distance: twice the clip -> world rounding cs_motion allows for). Each frame the entry is drawn within them counts;
// one beyond starts over from where the samples are now - a moving or animated object (a character breathing or shifting
// its weight drifts more than that within a second, a door swinging), or another copy of the mesh in the entry.
void keep_track(uint record_at, uint first, uint triangles, float3 cam, bool start)
{
	const uint samples = min(triangles, KEEP_SAMPLES);
	const uint2 h = Geo.Load2(record_at);                   // frames still, anchors taken
	bool moved = start || h.y != samples;
	for (uint k = 0u; k < samples && !moved; ++k)
	{
		const float3 now = asfloat(Geo.Load3(first + ((k * triangles) / samples) * 3u * VERTEX_BYTES));
		if (!(length(now - asfloat(Geo.Load3(record_at + 16u + k * 12u))) <= 0.002 + 0.0002 * length(now - cam)))
			moved = true;
	}
	if (!moved)
	{
		Geo.Store(record_at, min(h.x + 1u, 0x7FFFFFFFu));
		return;
	}
	for (uint j = 0u; j < samples; ++j)
		Geo.Store3(record_at + 16u + j * 12u, Geo.Load3(first + ((j * triangles) / samples) * 3u * VERTEX_BYTES));
	Geo.Store2(record_at, uint2(0u, samples));
}

// One group per draw: clip-space xyz -> absolute world position (camera-relative + c[9]), in place, and the draw's TLAS
// instance. A refit must keep each triangle's active state, so nothing becomes NaN: a vertex that cannot be converted
// (w ~ 0) takes the position of a valid vertex of its triangle (zero area, never hit), and a draw whose counter does not
// show all of its vertices (not executed yet when this frame's lighting ran, or incomplete) collapses to the camera
// position and gets instance mask 0.
[numthreads(64, 1, 1)]
void cs_convert(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
	const uint entry = TABLE_DRAWS + group.x * TABLE_STRIDE;
	const uint4 e = Table.Load4(entry);
	const uint2 blas = Table.Load2(entry + 16);
	const bool retained = (e.w & 8u) != 0u;                  // v33: last frame's draw kept: already converted, counters gone
	const bool complete = retained || Geo.Load(e.z) == e.y * VERTEX_BYTES;
	// v36: a draw the lighting before found already run (cs_late) is the frame before's: recorded after that lighting took
	// its epoch but executed before it, its clip space is that frame's camera (v7-v35 used this frame's: a copy shifted by
	// the camera's motion and jitter, whose shadows came and went with the game's recording order)
	const bool late = !retained && Geo.Load(GEO_LATE + (e.z / 16u) * 4u) == Table.Load(TABLE_EPOCH) + 1u;
	float4 r0 = c[19], r1 = c[20], r2 = c[21], r3 = c[22];
	float2 lin = c[10].xy;
	float3 cam = c[9].xyz;
	if (late)
	{
		r0 = asfloat(Data.Load4(FRAME_INV_PREV));
		r1 = asfloat(Data.Load4(FRAME_INV_PREV + 16));
		r2 = asfloat(Data.Load4(FRAME_INV_PREV + 32));
		r3 = asfloat(Data.Load4(FRAME_INV_PREV + 48));
		lin = asfloat(Data.Load2(FRAME_C10_PREV));
		cam = asfloat(Data.Load3(FRAME_CAM_PREV));
	}
	const uint triangles = e.y / 3;
	const uint material = (e.w & 2u) != 0 ? Table.Load(entry + 24) : NO_MATERIAL;   // a new entry: clear its material slots
	uint valid = 0;
	for (uint i = thread.x; i < triangles && !retained; i += 64)
	{
		if (material != NO_MATERIAL)
		{
			Geo.Store(GEO_MATERIAL + (material + i) * 4, 0u);
			Geo.Store(GEO_MATERIAL_PBR + (material + i) * 4, 0u);   // v82: its second slot too
		}
		const uint a = e.x + i * 3 * VERTEX_BYTES;
		float3 p[3];
		bool ok[3];
		[unroll]
		for (uint k = 0; k < 3; ++k)
		{
			const float3 clip = asfloat(Geo.Load3(a + k * VERTEX_BYTES));
			const float w = (clip.z - lin.y) / lin.x;
			const float4 h = clip.x * r0 + clip.y * r1 + clip.z * r2 + w * r3;
			p[k] = h.xyz / h.w;
			ok[k] = complete && finite_and_near(p[k]);
		}
		const float3 fallback = ok[0] ? p[0] : ok[1] ? p[1] : ok[2] ? p[2] : float3(0.0, 0.0, 0.0);
		[unroll]
		for (uint j = 0; j < 3; ++j)
			Geo.Store3(a + j * VERTEX_BYTES, asuint((ok[j] ? p[j] : fallback) + cam));
		valid += ok[0] && ok[1] && ok[2] ? 1u : 0u;
	}
	valid = WaveActiveSum(valid);
	if (WaveIsFirstLane() && valid != 0)
		Data.InterlockedAdd(STATS + STAT_TRIANGLES * 4, valid);
	DeviceMemoryBarrierWithGroupSync();                     // v94: the group's converted vertices, for keep_track
	const uint record = Table.Load(entry + TABLE_KEEP);
	const bool kept = (e.w & DRAW_KEPT) != 0u;              // v94: an entry the game no longer draws (retained too)
	if (thread.x == 0)
	{
		if (!complete)
			Data.InterlockedAdd(STATS + STAT_INCOMPLETE * 4, 1);
		if (late)
			Data.InterlockedAdd(STATS + STAT_LATE * 4, 1);
		if (record != 0u && complete && !retained)
			keep_track(GEO_KEEP + (record - 1u) * KEEP_RECORD_BYTES, e.x, triangles, cam, (e.w & DRAW_NEW_RECORD) != 0u);
		// D3D12_RAYTRACING_INSTANCE_DESC: identity 3x4 transform, InstanceID:24 (the draw) | InstanceMask:8,
		// InstanceContributionToHitGroupIndex:24 | Flags:8 (TRIANGLE_CULL_DISABLE), BLAS address
		const uint inst = Table.Load(12) + group.x * 64;
		uint mask = complete && any(blas != 0u) ? ((e.w & 1u) != 0 ? MASK_HAIR : (e.w & 4u) != 0 ? MASK_CUTOUT : MASK_SCENE) : 0u;   // e.w bit 0: hair, bit 2: cut-out
		if (kept)                                           // v94: in the scene once still for KEEP_STILL_FRAMES (shadow rays only)
			mask = any(blas != 0u) && record != 0u && Geo.Load(GEO_KEEP + (record - 1u) * KEEP_RECORD_BYTES) >= KEEP_STILL_FRAMES ? MASK_KEPT : 0u;
		if (mask == MASK_CUTOUT)
			Data.InterlockedAdd(STATS + STAT_CUTOUT_DRAWS * 4, 1);
		if (mask == MASK_KEPT)
			Data.InterlockedAdd(FRAME_KEPT, 1u);
		Geo.Store4(inst, asuint(float4(1.0, 0.0, 0.0, 0.0)));
		Geo.Store4(inst + 16, asuint(float4(0.0, 1.0, 0.0, 0.0)));
		Geo.Store4(inst + 32, asuint(float4(0.0, 0.0, 1.0, 0.0)));
		// flags: TRIANGLE_CULL_DISABLE, v23 + FORCE_NON_OPAQUE for cut-outs and hair (sun_visible lets light through them);
		// v32: InstanceContributionToHitGroupIndex bit 0 = the mask's coordinates are all written (checked here: cs_zero
		// clears the counters right after this pass - v30-v31 checked them later in the rays, found 0 and never used a mask)
		const uint uv = Table.Load(entry + TABLE_UV);
		const bool uv_ok = uv != NO_UV && (retained || (complete && Geo.Load(e.z + 8) == e.y * 8u));
		// (v94: kept geometry TRIANGLE_CULL_DISABLE | FORCE_OPAQUE - the shadow rays never test it as a card: its data is gone)
		Geo.Store4(inst + 48, uint4(group.x | (mask << 24), ((mask == MASK_CUTOUT || mask == MASK_HAIR ? 9u : mask == MASK_KEPT ? 5u : 1u) << 24) | (uv_ok ? 1u : 0u), blas.x, blas.y));
		// v37: a cut-out the game drew on time this frame: its slot names its texture for this frame (cs_mask)
		const uint source = Table.Load(entry + TABLE_SOURCE);
		if (uv_ok && !late && !retained && source != 0u && (mask == MASK_CUTOUT || mask == MASK_HAIR))
			mask_insert(source, Table.Load(entry + TABLE_UV + 4));
	}
}

// v37: the heap slot each listed draw's mask is read from this frame. The game refills its shader-visible heap from staging
// descriptors every frame, so a slot is sure only in the frame that drew with it: a late draw (the frame before's) or a
// retained one takes the slot a draw of this frame uses for the same texture (cs_convert's table), else lets the light
// through (v30-v36 read the old slot - by then often another texture: the cards' whole triangles shadowed in those frames)
[numthreads(64, 1, 1)]
void cs_mask(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= Table.Load(0))
		return;
	const uint entry = TABLE_DRAWS + id.x * TABLE_STRIDE;
	const uint4 e = Table.Load4(entry);
	uint slot = Table.Load(entry + TABLE_UV + 4);
	const uint source = Table.Load(entry + TABLE_SOURCE);
	const bool retained = (e.w & 8u) != 0u;
	const bool late = !retained && Geo.Load(GEO_LATE + (e.z / 16u) * 4u) == Table.Load(TABLE_EPOCH) + 1u;
	if (slot != NO_UV && source != 0u && (late || retained))
	{
		slot = mask_lookup(source);
		Data.InterlockedAdd(STATS + (slot == STALE_MASK ? STAT_MASK_STALE : STAT_MASK_REMAPPED) * 4, 1);
	}
	Geo.Store(GEO_MASK_SLOT + id.x * 4u, slot);
	Geo.Store2(GEO_MASK_VOTES + id.x * 8u, uint2(0u, late || retained ? MASK_NO_VOTE : 0u));   // v55: cs_motion's votes start here
}

// v36: after the conversion: the next epoch's draws that already ran - recorded after this lighting took its epoch, executed
// before it (this frame's draws: the game records some late) - found by their stream-out counters (the next region's,
// zeroed after its last conversion three epochs ago) and tagged with the next epoch for its cs_convert; and this frame's
// camera for them (FRAME_INV_PREV / FRAME_C10_PREV - this frame's own conversion has read the last ones before this pass)
[numthreads(64, 1, 1)]
void cs_late(uint3 id : SV_DispatchThreadID)
{
	const uint next = (Table.Load(4) + COUNTER_BYTES) % (REGIONS * COUNTER_BYTES);
	Geo.Store(GEO_LATE + (next / 16u + id.x) * 4u, Geo.Load(next + id.x * 16u) != 0u ? Table.Load(TABLE_EPOCH) + 2u : 0u);
	if (id.x < MASK_HASH_SLOTS)                             // v37: the next frame's texture table starts empty
		Geo.Store2(GEO_MASK_HASH + id.x * 8u, uint2(0u, 0u));
	if (id.x == 0)
	{
		Data.Store4(FRAME_INV_PREV, asuint(c[19]));
		Data.Store4(FRAME_INV_PREV + 16, asuint(c[20]));
		Data.Store4(FRAME_INV_PREV + 32, asuint(c[21]));
		Data.Store4(FRAME_INV_PREV + 48, asuint(c[22]));
		Data.Store2(FRAME_C10_PREV, asuint(c[10].xy));
	}
}

// ---- path tracing (v15: the Remix-style integrator on the scene data) ----

// Radiance of the open sky in direction w as the shading point's baked ambient sees it: the flat part plus the lobe
// toward its dominant direction (1.5 * t5 * max(w.D, 0) gives back t5 * (N.D) as irradiance / pi for N = D)
float3 sky_radiance(float3 w, float3 a4, float3 a5, float3 axis)
{
	return a4 + 1.5 * a5 * saturate(dot(w, axis));
}

// The probes' view space from a camera-relative position (the probes' world = their camera position + p_rel; the
// inverse of the view -> world rows)
float3 ibl_view_position(float3 p_rel)
{
	const float3 w = ibl_c3.xyz + p_rel - float3(ibl_c5[0].w, ibl_c5[1].w, ibl_c5[2].w);
	const float3 r0 = ibl_c5[0].xyz, r1 = ibl_c5[1].xyz, r2 = ibl_c5[2].xyz;
	const float3 c0 = cross(r1, r2), c1 = cross(r2, r0), c2 = cross(r0, r1);
	const float det = dot(r0, c0);
	return float3(dot(c0, w), dot(c1, w), dot(c2, w)) / (abs(det) > 1e-12 ? det : 1.0);
}

// The game's environment in direction dir seen from p_rel (camera-relative), with the probes of pixel px's tile, as
// Cs_SpecularIbl blends them (mip: the probes' prefiltered levels, 0-7)
float3 env_radiance(uint2 px, float3 p_rel, float3 dir, float mip)
{
	const uint tile = (px.y >> 4) * uint(ibl_c0.w) + (px.x >> 4);
	const float4 v1 = float4(ibl_view_position(p_rel), 1.0);
	float3 sum = 0.0, last = 0.0;
	float remaining = 1.0;
	bool done = false;
	for (int word = 7; word >= 0 && !done; --word)
	{
		uint bits = IblBits[tile].bits[word];
		while (bits != 0 && !done)
		{
			const uint bit = firstbithigh(bits);
			bits &= ~(1u << bit);
			const int index = ibl_c0.z - int(uint(word) * 32u + bit);
			if (index < 0 || index >= 250)
				continue;
			const CubemapInfo info = IblInfo[index];
			const float3 q = abs(float3(dot(info.influence[0], v1), dot(info.influence[1], v1), dot(info.influence[2], v1)));
			if (!(q.x < 1.0 && q.y < 1.0 && q.z < 1.0))
				continue;                                   // outside this probe's influence box
			const bool outer = q.x > info.inner.x || q.y > info.inner.y || q.z > info.inner.z;
			float3 lookup = dir;
			float w = info.weight;
			if (info.distant == 0)
			{
				// box projection: where the direction leaves the projection box, seen from the capture point
				const float3 p = float3(dot(info.box[0], v1), dot(info.box[1], v1), dot(info.box[2], v1));
				const float3 d = float3(dot(info.dir[0].xyz, dir), dot(info.dir[1].xyz, dir), dot(info.dir[2].xyz, dir));
				const float3 t3 = (sign(d) - p) / float3(d.x == 0.0 ? 1e-5 : d.x, d.y == 0.0 ? 1e-5 : d.y, d.z == 0.0 ? 1e-5 : d.z);
				const float3 h = p + d * min(t3.x, min(t3.y, t3.z)) - info.offset;
				lookup = float3(dot(info.back[0], h), dot(info.back[1], h), dot(info.back[2], h));
				if (outer)
				{
					const float3 f = 1.0 - max(q - info.inner, 0.0) * info.fade;
					w = f.x * f.y * f.z;
				}
			}
			last = IblCubes[NonUniformResourceIndex(index)].SampleLevel(EnvSampler, lookup, mip).rgb;
			if (outer || info.distant != 0)
			{
				sum += last * (w * remaining);
				remaining *= 1.0 - w;
			}
			else
			{
				sum += last * remaining;                    // inside a box-projected probe's inner box: it covers the point
				done = true;
			}
		}
	}
	if (!done)
		sum += last * remaining;
	return sum * ibl_c1.y;
}

// The open sky in direction dir. v28 looked for the game's distant probe, which the game does not have (v28's log: found by
// 0.0% of the rays). v35: the add-on's own sky (cs_sky): the game's outdoor ambient as its open, upward-facing surfaces get
// it - the flat part plus the lobe toward its dominant direction, as the lighting takes it (sky_radiance) - darker toward
// the ground below the horizon. False until cs_sky has seen open sky (then the baked ambient stays).
#define SKY_GROUND 0.4          // below the horizon: the ground's share of the sky's light
bool sky_env(uint2 px, float3 dir, out float3 radiance)
{
	const float4 a4 = asfloat(Data.Load4(SKY_A4));
	if (!(a4.w >= 1.0))
	{
		radiance = 0.0;
		return false;
	}
	const float3 a5 = asfloat(Data.Load3(SKY_A5)), axis = asfloat(Data.Load3(SKY_AXIS));
	radiance = a4.rgb + 1.5 * a5 * saturate(dot(dir, axis));
	if (dir.y < 0.0)
		radiance *= lerp(1.0, SKY_GROUND, saturate(-dir.y * 4.0));
	return true;
}

// v35: the add-on's sky from the game's outdoor ambient. One group samples a 64 x 36 grid of the frame (jittered per
// frame): pixels facing up (normal y > 0.7) that the game sees open to the sky (its large-scale ambient occlusion t10.y
// near 1; weight open^8 x up), their flat ambient t4, lobe t5 and its direction (G-buffer 1), weighted means blended into
// SKY_* (1 / frames, at least 5%). Indoors no pixel is open: the sky stays as it was last seen.
#define SKY_GRID_X 64
#define SKY_GRID_Y 36
groupshared float4 gs_sky[3][256];
[numthreads(256, 1, 1)]
void cs_sky(uint3 t : SV_GroupThreadID)
{
	const uint2 size = full_size();
	const float2 jitter = random3(uint2(7, 11), g_frame).xy;
	float4 a4s = 0.0, a5s = 0.0, axs = 0.0;
	for (uint k = t.x; k < SKY_GRID_X * SKY_GRID_Y; k += 256)
	{
		const float2 cell = float2(k % SKY_GRID_X, k / SKY_GRID_X) + jitter;
		const uint2 px = min(uint2(cell * float2(size) / float2(SKY_GRID_X, SKY_GRID_Y)), size - 1);
		if (Depth[px] >= 1.0)
			continue;
		const float3 n = decode_normal(GBuffer0[px]);
		const float open = saturate(InT10[px].y);
		const float w = pow(open, 8.0) * saturate((n.y - 0.7) / 0.3);
		if (!(w > 0.0))
			continue;
		const float3 a4 = InT4[px].rgb, a5 = InT5[px].rgb;
		a4s += float4(a4 * w, w);
		a5s += float4(a5 * w, w);
		axs += float4(decode_ambient_direction(GBuffer1[px]) * (w * dot(a5, float3(0.2126, 0.7152, 0.0722))), w);
	}
	gs_sky[0][t.x] = a4s;
	gs_sky[1][t.x] = a5s;
	gs_sky[2][t.x] = axs;
	GroupMemoryBarrierWithGroupSync();
	[unroll]
	for (uint step = 128; step > 0; step >>= 1)
	{
		if (t.x < step)
		{
			[unroll]
			for (uint q = 0; q < 3; ++q)
				gs_sky[q][t.x] += gs_sky[q][t.x + step];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (t.x != 0)
		return;
	a4s = gs_sky[0][0];
	a5s = gs_sky[1][0];
	axs = gs_sky[2][0];
	const float4 old4 = asfloat(Data.Load4(SKY_A4)), old5 = asfloat(Data.Load4(SKY_A5));
	const float3 old_axis = asfloat(Data.Load3(SKY_AXIS));
	if (a4s.w >= 2.0)                                       // a few fully open pixels (or many partly open)
	{
		const float3 a4 = a4s.rgb / a4s.w, a5 = a5s.rgb / a5s.w;
		const float3 axis = dot(axs.xyz, axs.xyz) > 1e-12 ? normalize(axs.xyz) : float3(0.0, 1.0, 0.0);
		const float frames = max(old4.w, 0.0);
		const float k = frames < 1.0 ? 1.0 : max(1.0 / (frames + 1.0), 0.05);
		const float3 axis_new = frames < 1.0 ? axis : normalize(lerp(old_axis, axis, k) + float3(0.0, 1e-4, 0.0));
		Data.Store4(SKY_A4, asuint(float4(lerp(old4.rgb, a4, k), min(frames + 1.0, 1e6))));
		Data.Store4(SKY_A5, asuint(float4(lerp(old5.rgb, a5, k), 0.0)));
		Data.Store3(SKY_AXIS, asuint(axis_new));
	}
	else if (old4.w >= 1.0)
		Data.Store(SKY_A5 + 12, asuint(old5.w + 1.0));      // not seen this frame: older by one
	const float4 now4 = asfloat(Data.Load4(SKY_A4));
	const float3 now5 = asfloat(Data.Load3(SKY_A5)), now_axis = asfloat(Data.Load3(SKY_AXIS));
	const float3 zenith = now4.rgb + 1.5 * now5 * saturate(now_axis.y);
	Data.Store(STATS + STAT_SKY_OPEN * 4, uint(min(a4s.w * 100.0, 4e9)));
	Data.Store(STATS + STAT_SKY_LUM * 4, now4.w >= 1.0 ? uint(min(dot(zenith, float3(0.2126, 0.7152, 0.0722)) * 1000.0, 4e9)) : 0u);
	Data.Store(STATS + STAT_SKY_AGE * 4, now4.w >= 1.0 ? uint(asfloat(Data.Load(SKY_A5 + 12))) : 0xFFFFFFFFu);
}

// The light of a ray that leaves the near field: the shading point's baked ambient seen as the sky. (v19 took the game's
// environment probes there, but the scene holds only what the camera sees: rays leave through the ceiling and the walls
// behind the camera, and the probes - which see that room lit - flooded interiors with light. The baked ambient knows
// those surfaces. env_radiance stays for later use.) v28: blended toward the open sky (sky_env) as Baked light goes to 0.
float3 escaped_light(uint2 px, float3 p_rel, float3 dir, float3 a4, float3 a5, float3 axis)
{
	const float3 baked = sky_radiance(dir, a4, a5, axis);
	float3 sky;
	if (g_baked >= 1.0 || !sky_env(px, dir, sky))
		return baked;
	return lerp(sky, baked, g_baked);
}

struct scene_hit
{
	bool hit;
	float t;
	uint draw;
	uint primitive;
};

// Closest hit in the scene (everything but hair) up to tmax; origin is camera-relative, the scene absolute (+ c[9])
scene_hit trace_scene(float3 origin, float3 dir, float tmax)
{
	RayDesc ray;
	ray.Origin = origin + c[9].xyz;
	ray.TMin = 0.0;
	ray.Direction = dir;
	ray.TMax = tmax;
	RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
	q.TraceRayInline(Scene, RAY_FLAG_NONE, MASK_SCENE, ray);
	q.Proceed();
	scene_hit h;
	h.hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
	h.t = h.hit ? q.CommittedRayT() : tmax;
	h.draw = h.hit ? SCENE_DRAW_COMMITTED(q) : 0u;
	h.primitive = h.hit ? q.CommittedPrimitiveIndex() : 0u;
	return h;
}

// v30: how much light passes cut-out triangle prim of draw at barycentrics bc, dist metres from the camera: the mask the
// game's own depth pass tests (its first material texture's first channel at the coordinates it reads, tlou_rt.cpp
// alpha_inputs), one texel at the mip the hit's footprint covers: 1 through a hole, 0 on the leaf. A draw without a mask (or
// whose coordinates are not all written, or g_alpha_on 0): CUTOUT_TRANSMISSION (v22-v29 for every card, the slabs).
#define CUTOUT_TRANSMISSION 0.5
#define ALPHA_CUTOFF 0.25       // the depth passes test 0.01, 0.2 or 0.5 (most masks are near 0 or 1)
void alpha_stat(bool sampled, uint stat)
{
	if (sampled)
		Data.InterlockedAdd(STATS + stat * 4, 1);
}
float cutout_mask(uint draw, uint prim, float2 bc, float dist, uint contribution, bool count = true)
{
	const bool sampled = count && ((prim + draw * 7u) & 15u) == 0u;   // v31: the reasons, counted for one triangle in 16 (v46: the rays' only)
	alpha_stat(sampled, STAT_ALPHA_CALLS);
	if (g_alpha_on == 0 || draw >= Table.Load(0))
	{
		alpha_stat(sampled, STAT_ALPHA_FALLBACK);
		return CUTOUT_TRANSMISSION;
	}
	const uint entry = TABLE_DRAWS + draw * TABLE_STRIDE;
	const uint2 a = Table.Load2(entry + TABLE_UV);
	const uint3 e = Table.Load3(entry);                     // vertices at e.x, e.y of them, counter at e.z
	if (a.x == NO_UV || a.y == NO_UV)
	{
		alpha_stat(sampled, STAT_ALPHA_FALLBACK);
		return CUTOUT_TRANSMISSION;
	}
	if ((contribution & 1u) == 0u)                          // v32: cs_convert found the coordinates incomplete
	{
		alpha_stat(sampled, STAT_ALPHA_FALLBACK);
		return CUTOUT_TRANSMISSION;
	}
	const uint slot = Geo.Load(GEO_MASK_SLOT + draw * 4u); // v37: this frame's slot for the mask's texture (cs_mask)
	if (slot == STALE_MASK)
		return 1.0;
	const uint u = a.x + prim * 24;
	const float2 t0 = asfloat(Geo.Load2(u)), t1 = asfloat(Geo.Load2(u + 8)), t2 = asfloat(Geo.Load2(u + 16));
	const float2 uv = t0 + (t1 - t0) * bc.x + (t2 - t0) * bc.y;
	// the mip: the mask's texels per metre on this triangle times the footprint at the hit (~2 mrad, at least 4 mm)
	const uint v = e.x + prim * 3 * VERTEX_BYTES;
	const float3 p0 = asfloat(Geo.Load3(v)), p1 = asfloat(Geo.Load3(v + VERTEX_BYTES)), p2 = asfloat(Geo.Load3(v + 2 * VERTEX_BYTES));
	const float area_w = length(cross(p1 - p0, p2 - p0));
	const float2 d1 = t1 - t0, d2 = t2 - t0;
	const float area_uv = abs(d1.x * d2.y - d1.y * d2.x);
	uint w, h, levels;
	HeapTextures[NonUniformResourceIndex(slot)].GetDimensions(0, w, h, levels);
	if (w == 0 || h == 0 || !(area_w > 1e-12) || !all(abs(uv) < 1e6))
	{
		alpha_stat(sampled, STAT_ALPHA_FALLBACK);
		return CUTOUT_TRANSMISSION;
	}
	const float texels = sqrt(area_uv * float(w) * float(h) / area_w) * max(dist * 0.002, 0.004);
	const uint mip = uint(clamp(floor(log2(max(texels, 1.0))), 0.0, float(max(levels, 1u) - 1u)));
	const uint2 size = uint2(max(w >> mip, 1u), max(h >> mip, 1u));
	const int2 texel = int2(min(uint2(frac(uv) * float2(size)), size - 1u));
	const float alpha = HeapTextures[NonUniformResourceIndex(slot)].Load(int3(texel, int(mip))).x;
	alpha_stat(sampled && alpha >= ALPHA_CUTOFF, STAT_ALPHA_SOLID);
	return alpha >= ALPHA_CUTOFF ? 0.0 : 1.0;
}

// v55 (user: black shapes beside litter and on clothing - floors under a card blocked where the card is not drawn): the
// mask model above (first texture, channel x, the stream-out coordinates, one cutoff) is the alpha test of the simple depth
// shaders only; 252 of the game's alpha-tested depth shaders blend up to 14 samples with constant-buffer coordinate maths.
// cs_motion's camera rays check each card against the depth buffer this frame; a draw whose mask disagrees too often lets
// the light through (a wrong mask blocked it in a shifted or foreign pattern: black copies of the papers beside them)
bool mask_distrusted(uint draw)
{
	const uint2 v = Geo.Load2(GEO_MASK_VOTES + draw * 8u);
	const float agree = float(v.x), disagree = float(v.y & ~MASK_NO_VOTE);
	return agree + disagree >= MASK_VOTES_MIN && disagree > MASK_DISAGREE * (agree + disagree);
}
float cutout_transmission(uint draw, uint prim, float2 bc, float dist, uint contribution, bool count = true)
{
	if (draw < Table.Load(0) && mask_distrusted(draw))
		return 1.0;
	return cutout_mask(draw, prim, bc, dist, contribution, count);
}

// 1 when nothing blocks the ray up to tmax. v30: cut-out cards (grass, leaves, fences) block where their mask is solid
// (cutout_transmission); v22-v29 these rays saw solid geometry only
float trace_visible(float3 origin, float3 dir, float tmax)
{
	RayDesc ray;
	ray.Origin = origin + c[9].xyz;
	ray.TMin = 0.0;
	ray.Direction = dir;
	ray.TMax = tmax;
	RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
	q.TraceRayInline(Scene, RAY_FLAG_NONE, MASK_SCENE | MASK_CUTOUT | MASK_KEPT, ray);   // v94: + the kept geometry
	float t = 1.0;
	while (q.Proceed())
		if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
		{
			t *= cutout_transmission(SCENE_DRAW_CANDIDATE(q), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics(),
				length(origin + dir * q.CandidateTriangleRayT()), q.CandidateInstanceContributionToHitGroupIndex());
			if (t < 0.05)
				q.CommitNonOpaqueTriangleHit();
		}
	return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : t;
}

// The closest hit with the cut-out cards solid where their mask is (DD2's bounce ray [5a, 5b]: alpha > 0.5), hair left out.
// v69: both renderers' bounce rays and the path reuse's visibility (v52-v68 dd2.hlsl dd2_trace, the DD2 renderer only - the
// RE9 renderer's rays went through leaves, fences and grass: trace_scene sees opaque geometry only)
scene_hit trace_cutouts(float3 origin, float3 dir, float tmax)
{
	RayDesc ray;
	ray.Origin = origin + c[9].xyz;
	ray.TMin = 0.0;
	ray.Direction = dir;
	ray.TMax = tmax;
	RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
	q.TraceRayInline(Scene, RAY_FLAG_NONE, MASK_SCENE | MASK_CUTOUT, ray);
	while (q.Proceed())
		if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
			cutout_transmission(SCENE_DRAW_CANDIDATE(q), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics(),
				length(origin + dir * q.CandidateTriangleRayT()), q.CandidateInstanceContributionToHitGroupIndex(), false) < 0.75)
			q.CommitNonOpaqueTriangleHit();
	scene_hit h;
	h.hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
	h.t = h.hit ? q.CommittedRayT() : tmax;
	h.draw = h.hit ? SCENE_DRAW_COMMITTED(q) : 0u;
	h.primitive = h.hit ? q.CommittedPrimitiveIndex() : 0u;
	return h;
}

// v53: last frame's view depth and normal at full-resolution pixel px - the renderers' RE9_GEO plane (re9.hlsl re9_offset:
// planes 6 / 7 by frame parity after pt_offset(2); cs_re9_accum writes it for both renderers). v51 removed cs_temporal,
// PLANE_GEO's only writer, and seen_light found nothing in v51-v52
uint pt_offset(uint parity);
float4 previous_geo(uint2 px)
{
	const uint2 size = full_size();
	return asfloat(Data.Load4(align_up(pt_offset(2), 256) + (6u + ((g_frame & 1u) ^ 1u)) * align_up(size.x * size.y * 16, 256) +
		(px.y * size.x + px.x) * 16));
}

// What the camera saw at a hit point last frame (the lit result kept by cs_history), validated against that frame's
// depth and normal: 1 = seen (radiance), 2 = the back of a surface (no light toward the ray), 0 = not seen
uint seen_light(float3 hit, float3 dir, out float3 radiance)
{
	radiance = 0.0;
	const float3 cam_prev = asfloat(Data.Load4(FRAME_CAM_PREV)).xyz;
	const float4 clip = mul(float4(hit + c[9].xyz - cam_prev, 1.0), load_matrix(FRAME_VP_PREV));
	if (clip.w <= 0.0 || Data.Load4(FRAME_INFO).x <= 2)
		return 0;
	const float2 ndc = clip.xy / clip.w;
	if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0)
		return 0;
	const float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
	const uint2 size = full_size();
	const uint2 fp = min(uint2(uv * float2(size)), size - 1);
	const float4 geo = previous_geo(fp);
	if (!all(isfinite(geo)) || geo.x <= 0.0 || abs(geo.x - clip.w) >= max(0.1, clip.w * 0.03) || !any(geo.yzw != 0.0))
		return 0;                                           // (hair keeps a zero normal there: not taken)
	if (dot(geo.yzw, dir) >= 0.0)
		return 2;
	radiance = unpack_r11g11b10(Data.Load(history_offset() + (fp.y * size.x + fp.x) * 4));
	return 1;
}

// ---- motion (v20) ----

// Where the surface under full-resolution pixel px was last frame, in the previous frame's camera: (ndc.x, ndc.y, view
// depth w); w <= 0: not in front of that camera. p: its camera-relative position now. With cs_motion's plane this
// follows the triangle (a moving character), else the camera only (a static scene).
float3 previous_clip(uint2 px, float3 p)
{
	if (g_motion_on != 0)
		return asfloat(Data.Load3(motion_offset() + (px.y * full_size().x + px.x) * MOTION_BYTES));
	const float3 cam = asfloat(Data.Load4(FRAME_CAM_CUR)).xyz;
	const float3 cam_prev = asfloat(Data.Load4(FRAME_CAM_PREV)).xyz;
	const float4 clip = mul(float4(p + cam - cam_prev, 1.0), load_matrix(FRAME_VP_PREV));
	return clip.w > 0.0 ? float3(clip.xy / clip.w, clip.w) : float3(0.0, 0.0, 0.0);
}

// v21: what cs_motion's camera ray found at full-resolution pixel px (PIXEL_*; 0 before cs_motion ran this frame)
uint pixel_flags(uint2 px)
{
	return g_motion_on != 0 ? Data.Load(motion_offset() + (px.y * full_size().x + px.x) * MOTION_BYTES + 12) : 0u;
}

// Per full-resolution pixel, the surface's position last frame (previous_clip's plane): a camera ray finds the triangle
// under the pixel (within 1% of the depth buffer's distance; cut-outs and hair cards in front of it are stepped through),
// the previous frame region holds the same BLAS entry's vertices as they were then (table +28), and the hit's
// barycentrics place the point on that triangle then. A pixel without such a triangle (a draw new this frame, a surface
// the scene lacks) or whose triangle moved more than MOTION_MAX (the entry was another object's) keeps the camera's
// reprojection.
[numthreads(8, 8, 1)]
void cs_motion(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const float d = Depth[id.xy];
	float3 result = 0.0;
	uint kind = 3;                                          // 0 moving, 1 static, 2 camera only, 3 sky
	uint flags = 0;                                         // PIXEL_* (v21)
	uint2 votes = 0;                                        // v55: this pixel's mask votes (cast, disagreeing)
	if (d < 1.0)
	{
		const float3 p = camera_relative_position(float2(id.xy), d);
		const float dist = length(p);
		float3 motion = 0.0;
		kind = 2;
		float tmin = 0.0;
		const bool voter = ((id.x + id.y + g_frame) & 3u) == 0u;   // v55: one pixel in four votes
		for (uint step = 0; step < 4; ++step)
		{
			RayDesc ray;
			ray.Origin = c[9].xyz;
			ray.TMin = tmin;
			ray.Direction = p / dist;
			ray.TMax = dist * 1.02 + 0.05;
			RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
			q.TraceRayInline(Scene, RAY_FLAG_NONE, MASK_DRAWN, ray);
			q.Proceed();
			if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
				break;
			const float t = q.CommittedRayT();
			// v55: a cut-out card (not hair) on the camera ray votes on its mask (mask_distrusted): at the depth buffer's surface
			// the game drew it there - solid; clearly in front of it the game discarded it there - a hole; in between, no vote
			// (a card lying on a surface). Late and retained draws are not in this frame's depth buffer (cs_mask: no votes)
			if (voter)
			{
				const uint vdraw = SCENE_DRAW_COMMITTED(q);
				if (vdraw < Table.Load(0) && (Table.Load(TABLE_DRAWS + vdraw * TABLE_STRIDE + 12) & 5u) == 4u &&
					(Geo.Load(GEO_MASK_VOTES + vdraw * 8u + 4u) & MASK_NO_VOTE) == 0u)
				{
					const float tight = 0.001 + 0.0005 * dist;
					const bool solid = abs(t - dist) <= tight, hole = t < dist - (3.0 * tight + 0.002);
					if (solid || hole)
					{
						const float m = cutout_mask(vdraw, q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), t,
							q.CommittedInstanceContributionToHitGroupIndex(), false);
						if (m == 0.0 || m == 1.0)                   // a texel read (not a fallback)
						{
							const bool agree = (m == 0.0) == solid;
							Geo.InterlockedAdd(GEO_MASK_VOTES + vdraw * 8u + (agree ? 0u : 4u), 1u);
							votes += uint2(1u, agree ? 0u : 1u);
						}
					}
				}
			}
			if (abs(t - dist) > 0.01 * dist + 0.01)
			{
				tmin = t + 0.001;                               // a cut-out in front: look behind it
				flags |= PIXEL_CUTOUT_IN_FRONT;
				continue;
			}
			const uint draw = SCENE_DRAW_COMMITTED(q);
			const uint entry = TABLE_DRAWS + draw * TABLE_STRIDE;
			const bool listed = draw < Table.Load(0);
			const uint kinds = listed ? Table.Load(entry + 12) : 0u;
			// v46: a hair or cut-out card at the depth buffer's distance is the surface only where its mask is solid (as the shadow
			// rays test it): eyelash and strand cards lie within 1% of the skin under them, and their whole quads (this ray is
			// opaque) flagged the skin and the eyes as hair - rectangles over the eyes and patches by the ear in the game's lighting
			if ((kinds & 5u) != 0u && Geo.Load(GEO_MASK_SLOT + draw * 4u) != STALE_MASK &&
				cutout_mask(draw, q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), t, q.CommittedInstanceContributionToHitGroupIndex(), false) >= 1.0)
			{
				tmin = t + 0.001;
				flags |= PIXEL_CUTOUT_IN_FRONT;
				continue;
			}
			flags |= PIXEL_TRIANGLE | ((kinds & 1u) != 0 ? PIXEL_HAIR : 0u);
			const uint previous = listed ? Table.Load(entry + 28) : NO_PREVIOUS;
			if (previous != NO_PREVIOUS)
			{
				const uint offset = q.CommittedPrimitiveIndex() * 3 * VERTEX_BYTES;
				const uint a = Table.Load(entry) + offset, b = previous + offset;
				const float2 bc = q.CommittedTriangleBarycentrics();
				const float3 w = float3(1.0 - bc.x - bc.y, bc.x, bc.y);
				const float3 now = w.x * asfloat(Geo.Load3(a)) + w.y * asfloat(Geo.Load3(a + VERTEX_BYTES)) + w.z * asfloat(Geo.Load3(a + 2 * VERTEX_BYTES));
				const float3 then = w.x * asfloat(Geo.Load3(b)) + w.y * asfloat(Geo.Load3(b + VERTEX_BYTES)) + w.z * asfloat(Geo.Load3(b + 2 * VERTEX_BYTES));
				const float3 delta = then - now;
				if (finite_and_near(delta) && dot(delta, delta) < MOTION_MAX * MOTION_MAX)
				{
					motion = delta;
					kind = length(delta) > 0.001 + 0.001 * dist ? 0u : 1u;   // above the clip -> world rounding (~1e-4 of the distance)
				}
			}
			break;
		}
		const float3 cam = asfloat(Data.Load4(FRAME_CAM_CUR)).xyz;
		const float3 cam_prev = asfloat(Data.Load4(FRAME_CAM_PREV)).xyz;
		const float4 clip = mul(float4(p + cam - cam_prev + motion, 1.0), load_matrix(FRAME_VP_PREV));
		result = clip.w > 0.0 ? float3(clip.xy / clip.w, clip.w) : float3(0.0, 0.0, 0.0);
	}
	Data.Store4(motion_offset() + (id.y * size.x + id.x) * MOTION_BYTES, uint4(asuint(result), flags));
	const uint3 stat = WaveActiveSum(uint3(kind == 0 ? 1u : 0u, kind == 1 ? 1u : 0u, kind == 2 ? 1u : 0u));
	const uint2 vote_stat = WaveActiveSum(votes);           // v55
	if (WaveIsFirstLane() && (any(stat != 0) || any(vote_stat != 0)))
	{
		Data.InterlockedAdd(STATS + STAT_MOTION_MOVING * 4, stat.x);
		Data.InterlockedAdd(STATS + STAT_MOTION_STATIC * 4, stat.y);
		Data.InterlockedAdd(STATS + STAT_MOTION_CAMERA * 4, stat.z);
		Data.InterlockedAdd(STATS + STAT_MASK_VOTES * 4, vote_stat.x);
		Data.InterlockedAdd(STATS + STAT_MASK_DISAGREE * 4, vote_stat.y);
	}
}

// Temporal accumulation of (B, V), S, the sun visibility and the reflections, with the second moments of V, S and the sun
// (for the filter's variance): reprojection to where the surface was last frame (v20: cs_motion, the triangle's own
// motion; before, the camera only), up to 32 frames, rejected where the surface changed (depth 5%, normal 0.8).
// Reflections keep fewer frames on smooth surfaces, whose reflected image moves with the view (2 at roughness 0, 32 from
// 0.25 up).
[numthreads(8, 8, 1)]
void cs_temporal(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= g_half_w || id.y >= g_half_h)
		return;
	const uint cur = g_frame & 1, prev = cur ^ 1;
	const uint2 px = sample_pixel(id.xy);
	const float d = Depth[px];
	if (d >= 1.0)
	{
		store_px(PLANE_ACCUM + cur, id.xy, float4(0.0, 0.0, 0.0, 1.0));
		store_px(PLANE_MOMENTS + cur, id.xy, float4(1.0, 1.0, 1.0, 0.0));
		store_px(PLANE_SUN + cur, id.xy, float4(1.0, 1.0, 0.0, 0.0));
		store_px(PLANE_ACCUM_R + cur, id.xy, 0.0);
		store_px(PLANE_ACCUM_E + cur, id.xy, 0.0);
		store_px(PLANE_GEO + cur, id.xy, 0.0);
		return;
	}
	const uint4 g0 = GBuffer0[px];
	const float3 n = decode_normal(g0);
	const float3 p = camera_relative_position(float2(px), d);
	const float4 ray = load_px(PLANE_TRACE, id.xy);
	const float4 st = load_px(PLANE_TRACE_S, id.xy);
	const float4 refl = load_px(PLANE_TRACE_R, id.xy);
	const float4 env = load_px(PLANE_TRACE_E, id.xy);
	const float3 ray_moments = float3(st.x, ray.w * ray.w, st.x * st.x);
	const float2 sun_moments = float2(st.y, st.y * st.y);

	const float3 clip = previous_clip(px, p);              // (ndc.xy, view depth) last frame
	float4 history = 0.0, history_moments = 0.0, history_sun = 0.0, history_refl = 0.0, history_env = 0.0;
	float weight = 0.0;
	if (clip.z > 0.0 && Data.Load4(FRAME_INFO).x > 1)
	{
		const float2 ndc = clip.xy;
		const float2 hpos = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * float2(g_half_w, g_half_h) - 0.5;
		const int2 base = int2(floor(hpos));
		const float2 f = hpos - base;
		[unroll]
		for (int k = 0; k < 4; ++k)
		{
			const int2 q = base + int2(k & 1, k >> 1);
			if (q.x < 0 || q.y < 0 || q.x >= int(g_half_w) || q.y >= int(g_half_h))
				continue;
			const float4 geo = load_px(PLANE_GEO + prev, uint2(q));
			if (geo.x <= 0.0 || abs(geo.x - clip.z) > 0.05 * clip.z + 0.05 || dot(geo.yzw, n) < 0.8)
				continue;                                   // a different surface there last frame
			const float w = ((k & 1) ? f.x : 1.0 - f.x) * ((k >> 1) ? f.y : 1.0 - f.y);
			history += load_px(PLANE_ACCUM + prev, uint2(q)) * w;
			history_moments += load_px(PLANE_MOMENTS + prev, uint2(q)) * w;
			history_sun += load_px(PLANE_SUN + prev, uint2(q)) * w;
			history_refl += load_px(PLANE_ACCUM_R + prev, uint2(q)) * w;
			history_env += load_px(PLANE_ACCUM_E + prev, uint2(q)) * w;
			weight += w;
		}
	}
	float4 result = ray, moments = float4(ray_moments, 1.0), sun_result = float4(sun_moments, 0.0, 0.0), refl_result = refl, env_result = env;
	if (weight > 0.01)
	{
		const float n_samples = min(history_moments.w / weight + 1.0, 32.0);
		const float a = 1.0 / n_samples;
		result = lerp(history / weight, ray, a);
		env_result = lerp(history_env / weight, env, a);
		moments = float4(lerp(history_moments.xyz / weight, ray_moments, a), n_samples);
		sun_result.xy = lerp(history_sun.xy / weight, sun_moments, a);
		const float roughness = (g0.w & 0xFFu) / 255.0;
		refl_result = lerp(history_refl / weight, refl, max(a, lerp(0.5, 1.0 / 32.0, saturate(roughness * 4.0))));
	}
	store_px(PLANE_ACCUM + cur, id.xy, result);
	store_px(PLANE_MOMENTS + cur, id.xy, moments);
	store_px(PLANE_SUN + cur, id.xy, sun_result);
	store_px(PLANE_ACCUM_R + cur, id.xy, refl_result);
	store_px(PLANE_ACCUM_E + cur, id.xy, env_result);
	store_px(PLANE_GEO + cur, id.xy, float4(view_z(d), n));
}

// Variance of the accumulated estimates at q (first filter pass): the per-frame variance (temporal second moment minus
// the squared mean, at least the binomial variance of that frame's rays) over the sample count, plus a short-history
// prior (a pixel just revealed has one sample and no measured variance, yet is the noisiest: it gets the widest filter,
// as SVGF's spatial estimate for short histories does). The sigma is 4.5-7x the estimate's real spread (svgf_sigma_sim.py).
// V and S here, the sun (one ray per frame) below.
float2 accumulated_variance(uint2 q, float v, float s)
{
	const float4 m = load_px(PLANE_MOMENTS + (g_frame & 1), q);
	const float count = max(m.w, 1.0);
	const float2 rays = float2(HEMI_RAYS, SHADOW_RAYS);
	const float2 per_frame = max(float2(m.y - v * v, m.z - s * s), float2(v * (1.0 - v), s * (1.0 - s)) / rays);
	return (max(per_frame, 0.0) + 0.25 / (rays * count)) / count;
}
float sun_variance(uint2 q, float sun, float m2)
{
	const float count = max(load_px(PLANE_MOMENTS + (g_frame & 1), q).w, 1.0);
	return (max(max(m2 - sun * sun, sun * (1.0 - sun)), 0.0) + 0.25 / count) / count;
}

// SVGF-style edge-aware filter: ATROUS_PASSES a-trous passes (B3 kernel, steps 1-16) with depth and normal weights plus
// visibility weights scaled by the estimate's standard deviation (sigma 4, as SVGF's luminance weight; the variance is
// filtered along with the weights squared), so noise is smoothed while converged shadow edges stay sharp. (B, V) follows
// the V weights, S and the sun their own. Reflections get the first two passes only (steps 1, 2), weighted down on
// smooth surfaces (whose reflected image is sharp); later passes carry them through.
[numthreads(8, 8, 1)]
void cs_atrous(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= g_half_w || id.y >= g_half_h)
		return;
	const uint cur = g_frame & 1;
	const bool first = g_pass == 0;
	const uint src = first ? PLANE_ACCUM + cur : PLANE_DENOISE + ((g_pass - 1) & 1);
	const uint src_s = first ? PLANE_MOMENTS + cur : PLANE_DENOISE_S + ((g_pass - 1) & 1);
	const uint src_sun = first ? PLANE_SUN + cur : PLANE_DENOISE_SUN + ((g_pass - 1) & 1);
	const uint src_r = first ? PLANE_ACCUM_R + cur : PLANE_DENOISE_R + ((g_pass - 1) & 1);
	const uint src_e = first ? PLANE_ACCUM_E + cur : PLANE_DENOISE_E + ((g_pass - 1) & 1);
	const uint dst = PLANE_DENOISE + (g_pass & 1);
	const uint dst_s = PLANE_DENOISE_S + (g_pass & 1);
	const uint dst_sun = PLANE_DENOISE_SUN + (g_pass & 1);
	const uint dst_r = PLANE_DENOISE_R + (g_pass & 1);
	const uint dst_e = PLANE_DENOISE_E + (g_pass & 1);
	const float4 geo = load_px(PLANE_GEO + cur, id.xy);
	if (geo.x <= 0.0)
	{
		store_px(dst, id.xy, float4(0.0, 0.0, 0.0, 1.0));
		store_px(dst_s, id.xy, float4(1.0, 0.0, 0.0, 0.0));
		store_px(dst_sun, id.xy, float4(1.0, 0.0, 0.0, 0.0));
		store_px(dst_r, id.xy, 0.0);
		store_px(dst_e, id.xy, 0.0);
		return;
	}
	const float4 center = load_px(src, id.xy);
	const float4 center_s = load_px(src_s, id.xy);
	const float4 center_sun = load_px(src_sun, id.xy);
	const float2 center_var = first ? accumulated_variance(id.xy, center.w, center_s.x) : center_s.yz;
	const float center_var_sun = first ? sun_variance(id.xy, center_sun.x, center_sun.y) : center_sun.y;
	const float3 sigma = 4.0 * sqrt(float3(center_var, center_var_sun)) + 0.02;
	const bool filter_r = g_pass < 2;
	const float roughness = filter_r ? (GBuffer0[sample_pixel(id.xy)].w & 0xFFu) / 255.0 : 0.0;
	const float r_weight = saturate(roughness * 5.0);      // neighbours count for little on smooth surfaces
	const int step = int(1u << g_pass);
	const float kernel[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };  // B3 spline
	float4 sum = 0.0, sum_r = 0.0, sum_e = 0.0;
	float sum_s = 0.0, sum_sun = 0.0, sum_wr = 0.0;
	float3 sum_w = 0.0, sum_var = 0.0;
	[unroll]
	for (int y = -2; y <= 2; ++y)
	{
		[unroll]
		for (int x = -2; x <= 2; ++x)
		{
			const int2 q = int2(id.xy) + int2(x, y) * step;
			if (q.x < 0 || q.y < 0 || q.x >= int(g_half_w) || q.y >= int(g_half_h))
				continue;
			const float4 gq = load_px(PLANE_GEO + cur, uint2(q));
			if (gq.x <= 0.0)
				continue;
			const float4 vq = load_px(src, uint2(q));
			const float4 sq = load_px(src_s, uint2(q));
			const float4 sunq = load_px(src_sun, uint2(q));
			const float2 var_q = first ? accumulated_variance(uint2(q), vq.w, sq.x) : sq.yz;
			const float var_sun_q = first ? sun_variance(uint2(q), sunq.x, sunq.y) : sunq.y;
			const float w_geo = kernel[abs(x)] * kernel[abs(y)] *
				exp(-abs(gq.x - geo.x) / (0.02 * geo.x * step + 0.01)) *
				pow(saturate(dot(gq.yzw, geo.yzw)), 16.0);
			const float3 w = w_geo * exp(-abs(float3(vq.w, sq.x, sunq.x) - float3(center.w, center_s.x, center_sun.x)) / sigma);
			sum += vq * w.x;
			sum_e += load_px(src_e, uint2(q)) * w.x;
			sum_s += sq.x * w.y;
			sum_sun += sunq.x * w.z;
			sum_w += w;
			sum_var += w * w * float3(var_q, var_sun_q);
			if (filter_r)
			{
				const float wr = w_geo * (x == 0 && y == 0 ? 1.0 : r_weight);
				sum_r += load_px(src_r, uint2(q)) * wr;
				sum_wr += wr;
			}
		}
	}
	const float3 inv = 1.0 / max(sum_w, 1e-6);
	store_px(dst, id.xy, sum * inv.x);
	store_px(dst_e, id.xy, sum_e * inv.x);
	store_px(dst_s, id.xy, float4(sum_s * inv.y, sum_var.xy * inv.xy * inv.xy, 0.0));
	store_px(dst_sun, id.xy, float4(sum_sun * inv.z, sum_var.z * inv.z * inv.z, 0.0, 0.0));
	store_px(dst_r, id.xy, filter_r ? sum_r / max(sum_wr, 1e-6) : load_px(src_r, id.xy));
}

// Depth/normal-aware upsampling of the filtered half-resolution results to a full-resolution pixel: (B, V), S, the sun
// visibility and the reflections (rgb, hit fraction)
void upsample(uint2 px, float z, float3 n, out float4 bv, out float s, out float sun, out float4 refl, out float3 env)
{
	const uint cur = g_frame & 1;
	const float2 hpos = (float2(px) + 0.5) * 0.5 - 0.5;
	const int2 base = int2(floor(hpos));
	const float2 f = hpos - base;
	float4 sum = 0.0, sum_r = 0.0, sum_e = 0.0;
	float sum_s = 0.0, sum_sun = 0.0, wsum = 0.0;
	[unroll]
	for (int k = 0; k < 4; ++k)
	{
		const uint2 q = uint2(clamp(base + int2(k & 1, k >> 1), int2(0, 0), int2(g_half_w, g_half_h) - 1));
		const float4 geo = load_px(PLANE_GEO + cur, q);
		const float w = ((k & 1) ? f.x : 1.0 - f.x) * ((k >> 1) ? f.y : 1.0 - f.y) *
			(geo.x > 0.0 ? exp(-abs(geo.x - z) / (0.02 * z + 0.01)) * pow(saturate(dot(geo.yzw, n)), 8.0) : 0.0) + 1e-4;
		sum += load_px(PLANE_DENOISE, q) * w;
		sum_s += load_px(PLANE_DENOISE_S, q).x * w;
		sum_sun += load_px(PLANE_DENOISE_SUN, q).x * w;
		sum_r += load_px(PLANE_DENOISE_R, q) * w;
		sum_e += load_px(PLANE_DENOISE_E, q) * w;
		wsum += w;
	}
	bv = sum / wsum;
	s = sum_s / wsum;
	sun = sum_sun / wsum;
	refl = sum_r / wsum;
	env = sum_e.rgb / wsum;
}

// Before the lighting: its inputs with the path-traced light, packed for the copies into the game's textures:
// t4' = t4 * V' + B * bounce, t5' = t5 * S' (G-buffer targets 4 / 5), t10.x' = t10.x * sun' (the sun shadow mask: the game's
// shadow maps and cloud shadows stay, the rays add what they block), t18' = the reflection rays' light where they hit
// something, the game's reflection where they did not (V', S', sun' = lerp(1, visibility, occlusion))
[numthreads(8, 8, 1)]
void cs_ambient(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	float3 a4 = InT4[id.xy].rgb, a5 = InT5[id.xy].rgb, r18 = InT18[id.xy].rgb;
	float4 m10 = InT10[id.xy];
	const float d = Depth[id.xy];
	if (d < 1.0 && (pixel_flags(id.xy) & PIXEL_HAIR) == 0)  // v21: hair keeps the game's own inputs (its strands are
	{                                                       // dithered: no stable path-traced estimate there)
		float4 bv, refl;
		float3 env;
		float s, sun;
		upsample(id.xy, view_z(d), decode_normal(GBuffer0[id.xy]), bv, s, sun, refl, env);
		a4 = a4 * (saturate(bv.w) * g_sky) + max(bv.rgb, 0.0) * g_bounce;   // (v97: the visibility fully, Occlusion's 1)
		a5 = a5 * (saturate(s) * g_sky);
		m10.x *= saturate(sun);
		r18 = r18 * (1.0 - saturate(refl.w) * g_reflect) + max(refl.rgb, 0.0) * g_reflect;
	}
	const uint at = id.y * copy_pitch() + id.x * 4;
	const uint4 m = uint4(saturate(m10) * 255.0 + 0.5);
	Data.Store(copy_offset(0) + at, pack_r11g11b10(a4));
	Data.Store(copy_offset(1) + at, pack_r11g11b10(a5));
	Data.Store(copy_offset(2) + at, m.x | (m.y << 8) | (m.z << 16) | (m.w << 24));
	Data.Store(copy_offset(3) + at, pack_r11g11b10(r18));
}

// (cs_history, which keeps the lit result for the next frame's hits, follows cs_full: v21 takes out exactly what cs_full
// put on the bounce)

// Material cache: a camera ray to the depth buffer's surface at the half-resolution pixel's sample; where it hits within
// 1% the triangle under the pixel is known (TLAS instance = the draw, BLAS primitive = the triangle, instancing included)
// and the G-buffer's albedo and roughness are blended into its slot (a quarter per sample: texture detail averages out
// over the frames). Cut-outs the camera sees through (the ray stops on the card in front) teach nothing.
uint blend_material(uint old, float3 albedo, float roughness)
{
	if (old == 0)
		return pack_material(albedo, roughness);
	const float4 m = unpack_material(old);
	return pack_material(lerp(m.rgb, albedo, 0.25), lerp(m.w, roughness, 0.25));
}

[numthreads(8, 8, 1)]
void cs_material(uint3 id : SV_DispatchThreadID)
{
	uint4 sample = 0;                                       // albedo x 1024 (rgb), 1 when written
	if (id.x < g_half_w && id.y < g_half_h)
	{
		const uint2 px = cycle_pixel(id.xy);
		const float d = Depth[px];
		const float3 p = camera_relative_position(float2(px), d);
		const float dist = length(p);
		if (d < 1.0 && dist > 0.5)
		{
			RayDesc ray;
			ray.Origin = c[9].xyz;                          // the camera (the scene is in absolute world space)
			ray.TMin = 0.3;                                 // past the near plane (0.21 m)
			ray.Direction = p / dist;
			ray.TMax = dist * 1.02 + 0.05;
			RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
			q.TraceRayInline(Scene, RAY_FLAG_NONE, MASK_DRAWN, ray);   // hair included: it learns its own material
			q.Proceed();
			if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT && abs(q.CommittedRayT() - dist) < dist * 0.01 + 0.02)
			{
				const uint base = Table.Load(TABLE_DRAWS + SCENE_DRAW_COMMITTED(q) * TABLE_STRIDE + 24);
				if (base != NO_MATERIAL)
				{
					// v82: from the game adapter's surface of this frame (cs_adapt ran first): the game's values or the auto-PBR
					// layer's - the same for every game; with the metalness and specular reflectance into the second slot
					const surface_record r = load_surface_record(g_frame & 1u, px);
					const float3 albedo = r.albedo;
					const uint at = GEO_MATERIAL + (base + q.CommittedPrimitiveIndex()) * 4;
					Geo.Store(at, blend_material(Geo.Load(at), albedo, r.roughness));
					const uint at_pbr = GEO_MATERIAL_PBR + (base + q.CommittedPrimitiveIndex()) * 4;
					Geo.Store(at_pbr, blend_material_pbr(Geo.Load(at_pbr), r.metallic, r.spec));
					sample = uint4(uint3(saturate(albedo) * 1024.0 + 0.5), 1u);
				}
			}
		}
	}
	sample = WaveActiveSum(sample);
	if (WaveIsFirstLane() && sample.w != 0)
	{
		Data.InterlockedAdd(STATS + STAT_ALBEDO_SUM * 4, sample.x);
		Data.InterlockedAdd(STATS + (STAT_ALBEDO_SUM + 1) * 4, sample.y);
		Data.InterlockedAdd(STATS + (STAT_ALBEDO_SUM + 2) * 4, sample.z);
		Data.InterlockedAdd(STATS + STAT_ALBEDO_COUNT * 4, sample.w);
		Data.InterlockedAdd(STATS + STAT_MATERIAL_WRITES * 4, sample.w);
	}
}

// Live check of the BLAS: rays from the camera to the depth buffer's surface on a grid of every 4th pixel
[numthreads(8, 8, 1)]
void cs_check(uint3 id : SV_DispatchThreadID)
{
	const uint2 px = id.xy * 4 + 2;
	const uint2 size = full_size();
	if (px.x >= size.x || px.y >= size.y)
		return;
	const float d = Depth[px];
	RayDesc ray;
	ray.Origin = c[9].xyz;                                  // the camera (the scene is in absolute world space)
	ray.TMin = 0.3;                                         // past the near plane (0.21 m)
	RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
	uint slot;
	if (d >= 1.0)
		return;                                             // v36: sky pixels not checked (v7-v35: never a hit)
	{
		const float3 p = camera_relative_position(float2(px), d);
		const float dist = length(p);
		if (dist < 0.5)
			return;
		ray.Direction = p / dist;
		ray.TMax = dist * 1.05 + 0.1;
		q.TraceRayInline(Scene, RAY_FLAG_NONE, MASK_DRAWN, ray);
		q.Proceed();
		if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
			slot = STAT_MISSED;
		else
		{
			const float rel = (q.CommittedRayT() - dist) / dist;
			slot = abs(rel) < 0.01 ? STAT_ON_SURFACE : rel < 0.0 ? STAT_IN_FRONT : STAT_BEHIND;
		}
	}
	Data.InterlockedAdd(STATS + slot * 4, 1);
}

// ---- the game's lights, ray traced (v18) ----

// The surface at a pixel as CS_DeferredLighting reads it from the G-buffer (v76: from the adapter's plane, which
// game_surface fills with it)
struct surface
{
	float3 p;               // camera-relative position
	float3 n;
	float3 ng;              // v76: the geometric normal (the depth buffer's; the path passes take the re-traced triangle's)
	float3 v;               // toward the camera
	float3 albedo;          // linear
	float roughness;        // the byte the split-sum LUT is indexed with (perceptual)
	float metallic;
	float spec;             // specular byte (F0 = spec * 0.08 / 255 for non-metals)
	uint stencil;
	bool character;         // G-buffer 1 .w bit 0: characters (the stencil's 32 bit in the captures; lights can keep to or off them)
	uint2 px;               // v22: the pixel, its view depth (the shadow maps' distance fade, filter tiling, screen shadow) ...
	float view_depth;
	float cavity;           // ... the material's occlusion (G-buffer 1 .w bits 2-7; the shadow maps' cavity term) and the
	float light_occlusion;  //     lights' occlusion without a shadow map: min(G-buffer 1 .x bits 8-13, capsule t16.x)
};

surface load_surface(uint2 px, float d)
{
	surface s;
	const uint4 g0 = GBuffer0[px], g1 = GBuffer1[px];
	s.px = px;
	s.view_depth = view_z(d);
	s.cavity = ((g1.w >> 2) & 63u) / 63.0;
	s.light_occlusion = min(((g1.x >> 8) & 63u) / 63.0, InT16.SampleLevel(SamplerS3, (float2(px) + 0.5) * c[8].zw, 0.0).x);
	s.p = camera_relative_position(float2(px), d);
	s.n = decode_normal(g0);
	s.v = -normalize(s.p);
	s.albedo = decode_albedo(g0);
	s.roughness = (g0.w & 0xFFu) / 255.0;
	s.metallic = (g1.y & 0x2000u) != 0 ? ((g1.y >> 8) & 31u) / 31.0 : 0.0;
	s.spec = float((g0.y >> 8) & 0xFFu);
	s.stencil = InT23[px].x;
	s.character = (g1.w & 1u) != 0;
	// v82: "Force auto-PBR (debug)" - the material as the game adapter's auto-PBR layer estimated it (adapter.hlsl cs_adapt
	// stored it for this frame), what another game would give the passes
	if (auto_pbr_forced())
	{
		const surface_record r = load_surface_record(g_frame & 1u, px);
		s.n = r.n;
		s.roughness = r.roughness;
		s.metallic = r.metallic;
		s.spec = r.spec;
	}
	return s;
}

// The 16x16-pixel light tile of a pixel (the lighting's numbering: rows rounded up to 4 tiles)
// v46: hair cards are two-sided - a strand seen from behind is lit on the viewer's side (the RE9 renderer's hair)
void face_viewer(inout surface s)
{
	if (dot(s.n, s.v) < 0.0)
		s.n = -s.n;
}

// v58 (user: "black splotches are flickering and still a big issue, ellies clothing and face seems to be broken too" - black
// patches in the textures' shapes: cloth folds, dirt and wet decals): the normal maps tilt the shading normal past the
// viewer or below the triangle; the response then gave no light (N.V <= 0), the bounce hemisphere turned into the surface
// and the contact ray took the light away. The shading normal is lifted onto the triangle's side (ng) and toward the viewer.
// v69: both renderers and ReSTIR DI's candidates (v58-v68 dd2.hlsl dd2_visible_normal, the DD2 renderer only)
float3 visible_normal(float3 n, float3 v, float3 ng)
{
	float k = dot(n, ng);
	if (k < 0.05)
		n = normalize(n + ng * (0.05 - k));
	k = dot(n, v);
	if (k < 0.05)
		n = normalize(n + v * (0.05 - k));
	return n;
}

// v47: the skin's subsurface amount as CS_DeferredLighting reads it: material flags (t3.x) bit 3 -> G-buffer 1 .x bits 0-7.
// The lighting fades its own diffuse light by (1 - w) there and adds CS_SkinDiffuse's after the SSSS blur (t7)
// v73: unused - the 2026-10-13 capture has that byte at 255 on every character's clothes and 0 on skin and hair (re9.hlsl)
float subsurface_weight(uint2 px)
{
	return (InT3[px].x & 8u) != 0u ? (GBuffer1[px].x & 0xFFu) / 255.0 : 0.0;
}

uint light_tiles_per_row()
{
	return 4u * ((uint(c[8].x) + 63u) >> 6);
}
uint light_tile(uint2 px)
{
	return (px.y >> 4) * light_tiles_per_row() + (px.x >> 4);
}
uint light_tile_count()
{
	return min(light_tiles_per_row() * ((full_size().y + 15u) >> 4), g_light_tiles);
}

struct light_sample
{
	float3 diffuse;         // radiance toward the camera, in the light buffer's units (before the lighting's final factor)
	float3 specular;
	float3 shadow_dir;      // toward the light's centre (planar box lights: against their axis)
	float shadow_dist;      // to the centre (planar: to the light's plane)
	float radius;
};

// The game's specular for a sphere light (its lights, and the sun as a sphere at distance 10): GGX with the representative
// point (the point of the sphere closest to the reflected ray), its energy normalisation and horizon fade, Smith
// height-correlated visibility, Schlick Fresnel. tl: to the light's centre, r: the roughness for this light, ndl: N.L
float3 game_specular(surface s, float3 tl, float dist, float radius, float r, float spec_scale, float ndl)
{
	const float3 refl = 2.0 * dot(s.n, s.v) * s.n - s.v;
	const float3 ray = refl * dot(tl, refl) - tl;
	const float3 lr = normalize(tl + ray * saturate(radius / max(length(ray), 1e-7)));
	const float size = radius / max(dist * 2.0, 1e-7);
	const float ac = max(r, 0.06) * max(r, 0.06);
	const float norm = ac / max(saturate(ac + size), 1e-7);
	const float horizon = saturate(ndl / max(size, 1e-7) * 0.5 + 0.5);
	const float3 h = normalize(lr + s.v);
	const float nol = saturate(dot(s.n, lr));
	const float nov = max(1e-7, dot(s.n, s.v));
	const float voh = saturate(dot(h, lr));
	const float noh = saturate(dot(h, s.n));
	const float alpha = r * r;
	const float dd = alpha / max(noh * noh * (alpha * alpha - 1.0) + 1.0, 1e-7);
	const float k2 = (r * 0.5 + 0.5) * (r * 0.5 + 0.5);
	const float a2 = k2 * k2;
	const float vis = 0.5 / (nov * sqrt(nol * (nol - nol * a2) + a2) + nol * sqrt(nov * (nov - nov * a2) + a2));
	const float3 f0 = lerp(s.spec * (0.08 / 255.0), s.albedo, s.metallic);
	return (dd * dd * vis * norm * norm * spec_scale * horizon * nol) * (f0 + (1.0 - f0) * pow(1.0 - voh, 5.0));
}

// ---- the game's own shadows of its lights (v22, decoded from CS_DeferredLighting's DXIL) ----
Texture2D<float> ShadowMaps[21] : register(t0, space2);    // the lights' shadow maps (D16), index LightDesc.shadow_map
Texture3D<float4> PcfTaps : register(t15);                  // the filter's tap offsets (xy): 4x4 tiles by pixel, 64 deep (RG32F)
Texture2D<float4> InT21 : register(t21);                    // a screen-space shadow (lights with flags bit 17 take its min)
SamplerComparisonState SamplerS1 : register(s1);           // the lighting's shadow comparison sampler

// A light's shadow map at surface s exactly as the lighting reads it. Light-space position: point lights one of four
// tetrahedral faces (fixed projections, the face whose axis is closest), spot / box lights the light's matrix (planar box
// maps linear). Reference depth (depth_a / w + depth_b) * depth_scale + bias. One comparison tap, or with a filter
// radius (pcf_radius) a blocker search over c[39].y taps from t15 (tiled by pixel) giving the penumbra, then the mean of
// as many comparison taps over the radius (flags bit 12: the radius grows from pcf_min with the penumbra squared).
// Spot / box maps add a contrast curve; then the minimum shadow, the distance fade, the cavity term on non-characters
// (against the light's axis) and the screen-space shadow (flags bit 17).
float game_shadow_map(LightDesc L, surface s, bool screen)
{
	const uint map = min(uint(L.shadow_map), 20u);
	const uint type = L.flags & 15u;
	const bool planar = (L.flags & 0x4Fu) == 0x41u;
	float2 xy;
	float lin, depth;                                       // the receiver's distance and its stored-depth form
	if (type == 0)
	{
		const float3 d = s.p + c[9].xyz - L.position;      // light -> surface
		const float f0 = dot(d.yz, float2(-0.57735026, 0.81649661)), f1 = dot(d.yz, float2(-0.57735026, -0.81649661));
		const float f2 = dot(d.xy, float2(-0.81649661, 0.57735026)), f3 = dot(d.xy, float2(0.81649661, 0.57735026));
		const float m = max(max(f0, max(f1, f2)), f3);
		float3 h;
		if (m == f0)
			h = float3(-0.30593601 * d.x, 0.44617999 * d.y - 0.33205599 * d.z, -0.45970100 * d.y + 0.88807398 * d.z);
		else if (m == f1)
			h = float3(-0.44617999 * d.y - 0.33205599 * d.z, 0.30593601 * d.x, -0.45970100 * d.y - 0.88807398 * d.z);
		else if (m == f2)
			h = float3(-0.30593601 * d.z, 0.44617999 * d.y - 0.33205599 * d.x, 0.45970100 * d.y - 0.88807398 * d.x);
		else
			h = float3(-0.44617999 * d.y - 0.33205599 * d.x, 0.30593601 * d.z, 0.45970100 * d.y + 0.88807398 * d.x);
		xy = h.xy / h.z;
		lin = h.z;
		depth = L.depth_a / h.z + L.depth_b;
	}
	else
	{
		const float4 pr = float4(s.p, 1.0);
		const float3 h = float3(dot(L.proj[0], pr), dot(L.proj[1], pr), dot(L.proj[2], pr));
		xy = planar ? h.xy : h.xy / h.z;
		lin = planar ? h.z * L.depth_b + L.depth_a : h.z;
		depth = planar ? h.z : L.depth_a / h.z + L.depth_b;
	}
	const float2 uv = float2(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
	const float ref = depth * L.depth_scale + L.shadow_fade.w;
	float sh = 1.0;
	if (L.pcf_radius == 0.0 || !screen)                     // v23: bounce hits take one tap
		sh = ShadowMaps[NonUniformResourceIndex(map)].SampleCmpLevelZero(SamplerS1, uv, ref);
	else
	{
		const uint4 c39 = asuint(c[39]);                    // x: tile mask, y: taps, zw: tile offset
		const uint2 tile = uint2((s.px.x + c39.z) & c39.x, (s.px.y + c39.w) & c39.x);
		uint mw, mh;
		ShadowMaps[NonUniformResourceIndex(map)].GetDimensions(mw, mh);
		const float texel = 1.0 / float(mw);
		float far = 0.0;
		for (uint k = 0; k < c39.y; ++k)
		{
			const float2 o = PcfTaps.Load(int4(tile, k, 0)).xy;
			const float b = ShadowMaps[NonUniformResourceIndex(map)].SampleLevel(SamplerS0, uv + o * (texel * L.pcf_radius), 0.0);
			far = max(far, lin - (planar ? b * L.depth_b + L.depth_a : L.depth_a / (b - L.depth_b)));
		}
		const float penumbra = saturate(far / L.pcf_range);
		const float radius = (L.flags & 0x1000u) != 0 ? L.pcf_min + penumbra * penumbra * (L.pcf_radius - L.pcf_min) : L.pcf_radius;
		float lit = 0.0;
		for (uint j = 0; j < c39.y; ++j)
			lit += ShadowMaps[NonUniformResourceIndex(map)].SampleCmpLevelZero(SamplerS1, uv + PcfTaps.Load(int4(tile, j, 0)).xy * (radius * texel), ref);
		sh = c39.y != 0 ? lit / c39.y : 1.0;
	}
	if (type != 0)
	{
		const float e = (L.shadow_curve & 0xFFFFu) / 256.0, k = (L.shadow_curve >> 16) / 256.0;
		sh = saturate(k * (1.001 - exp2(e * log2(1.001 - sh))));
	}
	sh = max(sh, L.shadow_fade.z);
	sh += saturate(L.shadow_fade.x * s.view_depth + L.shadow_fade.y) * (1.0 - sh);
	if (!s.character)
		sh *= saturate(2.0 * s.cavity * s.cavity - 1.0 + abs(dot(-L.axis, s.n)));
	if (screen && (L.flags & 0x20000u) != 0)
		sh = min(sh, InT21.SampleLevel(SamplerS2, (float2(s.px) + 0.5) * c[8].zw, 0.0).x);
	return sh;
}

// A light's shadow as the lighting applies it: its shadow map, and for lights without one the G-buffer's occlusion
// (points c[77].x, spots c[77].y; box lights none); spots with a polar cookie take that occlusion (c[77].z) on top
float game_light_shadow(LightDesc L, surface s)
{
	const uint type = L.flags & 15u, kind = (L.flags >> 9) & 7u;
	const float sh = L.shadow_map != -1 ? game_shadow_map(L, s, true) : 1.0;
	if (type == 2 && kind > 3)
		return sh * saturate((s.light_occlusion - 1.0) * c[77].z + 1.0);
	if (L.shadow_map == -1 && type != 1)
		return saturate((s.light_occlusion - 1.0) * (type == 0 ? c[77].x : c[77].y) + 1.0);
	return sh;
}

// One of the game's lights at a surface as CS_DeferredLighting computes it, v22 with its own shadow (the shadow map, or
// the G-buffer's occlusion for lights without one; the ray-traced visibility of those comes on top): false when the light
// does not reach the surface
// A game's light's distance curve as the lighting has it: rational, windowed to 0 at its range by falloff.w (TLOU's lights:
// 1.11 / (1 + d) - 0.11, 0 at 9 m)
float light_curve(LightDesc L, float dist, float dist2)
{
#ifdef UE_GAME
	if (L.unused272[0].x != 0.0)
		return ue_light_curve(L, dist, dist2);          // UE RT v19: an Unreal light's (ue_game.hlsli)
#endif
	return saturate(L.falloff.z / (1.0 + dist * L.falloff.x + (dist * L.falloff_cubic + L.falloff.y) * dist2) + L.falloff.w);
}
// v99: the RTXPT renderer's Physical light falloff - a game's light as a sphere light whose intensity keeps the light's
// energy within its authored range: I = colour x (1 / R) x the integral over [0, R] of d^2 x its curve (the mean flux
// through the spheres around it out to its cull radius R; 8-point Gauss-Legendre, exact to 4 digits on TLOU's curves: 1.17
// for 1.11 / (1 + d) - 0.11 over 9 m - 2.6x the game's light at 1 m, 0.8x at 3 m, and light past 9 m) ...
float light_intensity(LightDesc L)
{
	static const float2 k_gl[4] = { float2(0.1834346425, 0.3626837834), float2(0.5255324099, 0.3137066459),
		float2(0.7966664774, 0.2223810345), float2(0.9602898565, 0.1012285363) };
	const float R = max(L.cull_radius, 0.01);
	float sum = 0.0;
	[unroll]
	for (uint i = 0; i < 4; ++i)
	{
		const float a = 0.5 * R * (1.0 - k_gl[i].x), b = 0.5 * R * (1.0 + k_gl[i].x);
		sum += k_gl[i].y * (a * a * light_curve(L, a, a * a) + b * b * light_curve(L, b, b * b));
	}
	return 0.5 * sum;
}
// ... and the sphere's light at normal incidence per unit intensity, its solid angle / (pi r^2): 2 / (d^2 (1 + sqrt(1 -
// r^2 / d^2))) - inverse-square away from it (r: the light's radius, at least the Light size slider's), no range
float sphere_light_falloff(float dist, float radius)
{
	const float d = max(dist, max(radius, 1e-3));
	return 2.0 / (d * d * (1.0 + sqrt(1.0 - saturate(radius * radius / (d * d)))));
}

// The light arriving from light L at camera-relative point p_rel as the lighting computes it before the surface's response
// and the shadow (v23, shared with the bounce hits' lamps): the clip plane of kind-2 cookies, the light's frustum (box
// lights, cookies, spots with a shadow map), falloff and near fade, the spot cones / box window, the cookie, point lights'
// angular mask (lamp shades) - its colour, the lighting's L toward it (not unit for planar box lights) and the distance;
// false when nothing arrives. v99: physical - the distance curve the sphere light's (light_intensity x
// sphere_light_falloff; planar box lights, a beam, keep theirs), the rest the lighting's (light_arrival: the game's curve)
bool light_arrival_ex(LightDesc L, float3 p_rel, bool physical, out float3 color, out float3 l, out float dist)
{
	color = 0.0;
	l = 0.0;
	dist = 0.0;
	const uint flags = L.flags, type = flags & 15u, kind = (flags >> 9) & 7u;
	const float3 pabs = p_rel + c[9].xyz;
	if ((flags & 0xE00u) == 0x400u && dot(float4(pabs, 1.0), c[38]) <= 0.0)
		return false;
	// distance: to the centre, or along the axis for planar box lights
	const float3 to_light = L.position - pabs;
	const bool planar = (flags & 0x4Fu) == 0x41u;
	float dist2;
	if (planar)
	{
		const float a = dot(to_light, L.axis);
		dist = -a;
		dist2 = a * a;
	}
	else
	{
		dist2 = dot(to_light, to_light);
		dist = sqrt(dist2);
	}
	l = to_light / dist;                                    // the lighting's L (not unit for planar box lights)
	// the light's frustum (box lights, cookies, spots with a shadow map): nothing outside it
	const bool projected = L.shadow_map != -1 ? type != 0 : (type == 1 || kind != 0);
	float2 q = 0.0;
	if (projected)
	{
		const float4 pr = float4(p_rel, 1.0);
		const float3 h = float3(dot(L.proj[0], pr), dot(L.proj[1], pr), dot(L.proj[2], pr));
		q = planar ? h.xy : h.xy / h.z;
		if (!(abs(q.x) <= 1.0 && abs(q.y) <= 1.0))
			return false;
	}
	const float along = type == 2 ? -dot(to_light, L.axis) : dist;
	float strength = (physical && !planar ? light_intensity(L) * sphere_light_falloff(dist, max(L.radius, g_light_radius)) : light_curve(L, dist, dist2)) *
		(L.near_fade > 0.0 ? saturate((along - L.near_dist) / L.near_fade) : (along > L.near_dist ? 1.0 : 0.0));
	if (type != 0)
	{
		float shape = 1.0;
		if (type == 1 && L.box_soft > 0.0)
			shape = saturate((L.box_x - abs(q.y) * L.box_x) / L.box_soft) * saturate((L.box_y - abs(q.x) * L.box_y) / L.box_soft);
		else if (kind != 0)
			shape = dot(-l, L.axis) > 0.0 ? 1.0 : 0.0;     // cookie lights: in front of the light (the cookie is the shape)
		else if (type == 2)
		{
			const float cl = dot(l, L.axis);
			shape = saturate(cl * L.cone.y + L.cone.x);
			if ((flags & 0x20000000u) != 0)
				shape *= saturate(cl * L.cone2.y + L.cone2.x);
#ifdef UE_GAME
			if (L.unused272[0].x != 0.0)
				shape *= shape;                             // UE RT v19: UE's SpotAttenuation squares its cone
#endif
		}
		strength *= shape;
	}
	if (type == 0)
	{
		// point lights: an angular mask around mask_axis (lamp shades)
		const float t = saturate((dot(normalize(to_light), L.mask_axis) - L.mask_lo) / (L.mask_hi - L.mask_lo));
		strength *= 1.0 - t * t * (3.0 - 2.0 * t);
	}
	if (!(strength > 0.0))
		return false;
	color = L.color * strength;
	if (kind != 0)
	{
		if (L.cookie >= 16u)
			return false;                                   // no cookie texture: the lighting gets black
		const uint cookie = L.cookie;
		if (kind > 3)
		{
			// polar cookie: angle around the light's axis, distance terms
			float u = atan2(q.y, q.x) * 0.15915494 + 0.5;
			if (q.x == 0.0)
				u = q.y < 0.0 ? 0.25 : 0.75;
			u = saturate(u);
			const float r = saturate(length(q));
			const float t = min(saturate(L.cookie_b.x / dist + L.cookie_b.y), L.cookie_a.x);
			const float lod = sqrt(saturate((dist - L.cookie_b.z) * L.cookie_a.y)) * L.cookie_b.w;
			float3 k = Cookies[NonUniformResourceIndex(cookie)].SampleLevel(SamplerS4, float3(u, (t + r) / (t + 1.0), 0.0), lod).rgb;
			const float v2 = t - r;
			if (v2 == saturate(v2))
				k += Cookies[NonUniformResourceIndex(cookie)].SampleLevel(SamplerS4, float3(u, v2, 0.0), 0.0).rgb;
			color *= k;
		}
		else
		{
			// projected cookie (kind 2 mirrored, kind 3 one channel)
			const float2 uv = kind == 2 ? 0.5 - 0.5 * q : 0.5 + 0.5 * q;
			const float3 k = Cookies[NonUniformResourceIndex(cookie)].SampleLevel(SamplerS2, float3(uv * L.cookie_b.xy + L.cookie_a.xy, L.cookie_a.z), 0.0).rgb;
			color *= kind == 3 ? k.xxx : k;
		}
	}
	return any(color > 0.0);
}
bool light_arrival(LightDesc L, float3 p_rel, out float3 color, out float3 l, out float dist)
{
	return light_arrival_ex(L, p_rel, false, color, l, dist);
}

bool eval_light(LightDesc L, surface s, out light_sample o)
{
	o.diffuse = 0.0;
	o.specular = 0.0;
	o.shadow_dir = 0.0;
	o.shadow_dist = 0.0;
	o.radius = L.radius;
	const uint flags = L.flags, kind = (flags >> 9) & 7u;
	// which pixels a light takes (stencil, characters)
	if ((flags & ((s.stencil & 32u) != 0 ? 16u : 32u)) != 0 || (flags & 0x4002000u) != 0)
		return false;
	if (((flags & 0x4000u) != 0 && !s.character) || ((flags & 0x10000000u) != 0 && s.character))
		return false;
	if ((flags & 0x8000u) != 0 && (s.character || (s.stencil & 32u) == 0))
		return false;
	if ((kind == 5 && (s.stencil & 192u) == 64u) || ((flags & 0x40000u) != 0 && (s.stencil & 64u) == 0))
		return false;
	float3 color, l;
	float dist;
	if (!light_arrival(L, s.p, color, l, dist))
		return false;
	const float ndl = dot(s.n, l);
	if ((flags & 128u) != 0)
		o.diffuse = color * (saturate(ndl) * (1.0 - s.metallic)) * s.albedo;
	if ((flags & 256u) != 0)
		o.specular = color * game_specular(s, l * dist, dist, L.radius, lerp(L.min_roughness, 1.0, s.roughness), L.spec_scale, ndl);
	o.shadow_dir = (flags & 0x4Fu) == 0x41u ? -L.axis : l;
	o.shadow_dist = dist;
	if (!any(o.diffuse + o.specular > 0.0))
		return false;
	const float sh = game_light_shadow(L, s);
	o.diffuse *= sh;
	o.specular *= sh;
	return sh > 0.0;
}

float random1(uint seed)
{
	return (pcg(seed) >> 8) / 16777216.0;
}

// The direction and distance of a shadow ray toward light L from absolute position pabs, as eval_light measures them
// (planar box lights: against their axis, the distance to their plane)
void light_ray(LightDesc L, float3 pabs, out float3 dir, out float dist)
{
	const float3 to_light = L.position - pabs;
	if ((L.flags & 0x4Fu) == 0x41u)
	{
		dist = -dot(to_light, L.axis);
		dir = -L.axis;
	}
	else
	{
		dist = length(to_light);
		dir = to_light / max(dist, 1e-6);
	}
}

// v21: the near plane of a light's shadow map (the depth along its axis below which the game's map holds no occluders):
// the stored depth depth_a / w + depth_b spans 0..1 between the near and the far plane (either way round); 0 = none
float shadow_near(LightDesc L)
{
	if (L.shadow_map == -1 || (L.flags & 0x4Fu) == 0x41u)
		return 0.0;                                         // no shadow map, or a planar (linear) one
	const float w0 = L.depth_b != 0.0 ? -L.depth_a / L.depth_b : -1.0;          // depth 0
	const float w1 = L.depth_b != 1.0 ? L.depth_a / (1.0 - L.depth_b) : -1.0;   // depth 1
	const float lo = min(w0 > 0.0 ? w0 : 1e30, w1 > 0.0 ? w1 : 1e30);
	return lo < 1e30 ? lo : 0.0;
}

// One shadow ray from surface s (leaving at origin, off its true surface) toward a point of light L's sphere (its radius,
// at least the Light size slider's; xi picks the point within the sphere's cone): 1 when nothing blocks it. The part of
// the ray next to the light is left out: LIGHT_SKIP, 1.5 radii or 5% of the distance (the lamp's own fixture), and v25
// for lights with a shadow map the zone before its near plane (shadow_near: where the game itself leaves the carrier out,
// e.g. Ellie's body in her flashlight beam). The rays see solid geometry only (MASK_SCENE: no cut-out cards, no hair).
// v101: the ray toward a point of a sphere of radius cone about the light (the RTXPT renderer's Light softness scales it;
// the fixture's skip keeps the light's own radius)
float light_visibility_cone(surface s, float3 origin, LightDesc L, float2 xi, float cone)
{
	float3 dir;
	float dist;
	light_ray(L, s.p + c[9].xyz, dir, dist);
	const float radius = max(L.radius, g_light_radius);
	if (cone > 0.0 && dist > cone)
	{
		const float sin_max = cone / dist;
		dir = cone_direction(dir, sqrt(1.0 - sin_max * sin_max), xi);
	}
	const float tmax = dist - max(max(max(LIGHT_SKIP, radius * 1.5), dist * 0.05), shadow_near(L));
	return tmax > 0.0 ? trace_visible(origin, dir, tmax) : 1.0;
}
float light_visibility(surface s, float3 origin, LightDesc L, float2 xi)
{
	return light_visibility_cone(s, origin, L, xi, max(L.radius, g_light_radius));
}

// Per full-resolution pixel: every light of its tile with the game's own shadow (v22: its shadow map, or for lights
// without one the G-buffer's occlusion), summed (LIGHT_SUM), and the share of that sum the rays leave lit, sum(f V) /
// sum(f) with luminance weights (LIGHT_RAW). v22-v24 traced only lights without a shadow map. v25 (user: GBRT's local
// lights cast every object's shadow): every light is traced, f already carrying the game's own shadow, so a light with a
// map gets map x rays - the rays add the casters its map lacks (characters, props, detail), the map keeps the casters my
// scene lacks (culled off screen) and the cut-out shapes; light_visibility skips the zone before its near plane (the
// player in her own flashlight beam). Except on characters (stencil 32): their skin and clothes keep the game's maps for
// the lights that have one (v19 traced them there and the faces lost their light). v20: with LIGHT_RAYS or fewer traced lights every
// one gets LIGHT_RAYS / count rays (points on its sphere), weighted by its contribution (no selection noise); with more,
// LIGHT_RAYS rays to traced lights picked in proportion to their contributions, one per stratum of the running sum
// (systematic resampling; v18-v19: one ray to one light picked by RIS). Light shadows 0: no rays.
// (v20_light_sim.py: unbiased; per-frame noise 2-3x lower than v19's.)
[numthreads(8, 8, 1)]
void cs_lights(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const float d = Depth[id.xy];
	float3 sum = 0.0;
	float vis = -1.0;                                       // -1: no light here
	uint4 stat = 0;                                         // lit pixel, lights evaluated, mostly blocked, character
	if (d < 1.0 && g_lights_on != 0)
	{
		const uint tile = light_tile(id.xy);
		const uint count = tile < light_tile_count() ? min(LightCounts.Load(tile * 4), 16u) : 0u;
		if (count != 0)
		{
			const surface s = load_surface(id.xy, d);
			float weight[16];                               // traced lights' weights (0: not traced)
			float total = 0.0, traced = 0.0;
			uint lit = 0;
			for (uint i = 0; i < count; ++i)
			{
				weight[i] = 0.0;
				const LightDesc L = Lights[LightIndex[tile * 16 + i]];
				light_sample ls;
				if (!eval_light(L, s, ls))
					continue;
				const float3 f = ls.diffuse + ls.specular;
				const float w = dot(f, float3(0.2126, 0.7152, 0.0722));
				sum += f;
				total += w;
				if ((L.shadow_map == -1 || !s.character) && w > 0.0)
				{
					weight[i] = w;
					traced += w;
					++lit;
				}
			}
			if (total > 0.0)
			{
				vis = 1.0;
				if (g_light_shadows > 0.0 && traced > 0.0)
				{
					float seen = 0.0;                       // estimate of sum(f V) over the traced lights
					const float3 origin = s.p + geometric_normal(id.xy, s.p, s.n) * (0.01 + length(s.p) * 0.002);   // v25: off the true surface
					if (lit <= LIGHT_RAYS)
					{
						const uint per_light = LIGHT_RAYS / lit;    // 1 light: 4 points on its sphere, 2: 2 each, 3-4: 1
						for (uint k = 0; k < count; ++k)
							if (weight[k] > 0.0)
							{
								const LightDesc L = Lights[LightIndex[tile * 16 + k]];
								float v = 0.0;
								for (uint j = 0; j < per_light; ++j)
									v += light_visibility(s, origin, L, sequence(g_frame * LIGHT_RAYS + j, random3(id.xy, 5 + k).xy));
								seen += weight[k] * v / per_light;
							}
					}
					else
					{
						// stratum r takes the traced light whose share of the running sum covers (r + u) / LIGHT_RAYS of it
						const float u = random1(pcg(id.x + pcg(id.y + pcg(g_frame + 0x5EEDu))));
						uint k = 0;
						float cdf = weight[0];
						float v = 0.0;
						for (uint r = 0; r < LIGHT_RAYS; ++r)
						{
							const float target = (float(r) + u) * (traced / LIGHT_RAYS);
							while (cdf <= target && k + 1 < count)
							{
								++k;
								cdf += weight[k];
							}
							v += light_visibility(s, origin, Lights[LightIndex[tile * 16 + k]], sequence(g_frame, random3(id.xy, 5 + r).xy));
						}
						seen = traced * v / LIGHT_RAYS;
					}
					vis = saturate((total - traced + seen) / total);
				}
				stat = uint4(1u, count, vis < 0.5 ? 1u : 0u, s.character ? 1u : 0u);
			}
		}
	}
	const uint i = id.y * size.x + id.x;
	Data.Store(light_dword(LIGHT_SUM) + i * 4, pack_r11g11b10(sum));
	Data.Store(light_dword(LIGHT_RAW) + i * 4, asuint(vis));
	stat = WaveActiveSum(stat);
	if (WaveIsFirstLane() && stat.x != 0)
	{
		Data.InterlockedAdd(STATS + STAT_LIT_PIXELS * 4, stat.x);
		Data.InterlockedAdd(STATS + STAT_LIGHT_EVALS * 4, stat.y);
		Data.InterlockedAdd(STATS + STAT_LIGHT_BLOCKED * 4, stat.z);
		Data.InterlockedAdd(STATS + STAT_LIGHT_CHARACTER * 4, stat.w);
	}
}

uint pack_normal(float3 n)
{
	n /= abs(n.x) + abs(n.y) + abs(n.z);
	const float2 o = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
	const uint2 u = uint2(saturate(o * 0.5 + 0.5) * 65535.0 + 0.5);
	return u.x | (u.y << 16);
}
float3 unpack_normal(uint v)
{
	const float2 o = float2(v & 0xFFFFu, v >> 16) * (2.0 / 65535.0) - 1.0;
	float3 n = float3(o, 1.0 - abs(o.x) - abs(o.y));
	if (n.z < 0.0)
		n.xy = (1.0 - abs(o.yx)) * float2(o.x >= 0.0 ? 1.0 : -1.0, o.y >= 0.0 ? 1.0 : -1.0);
	return normalize(n);
}

// Temporal accumulation of the visibility (up to LIGHT_HISTORY frames; reprojection to where the surface was last frame -
// v20: with its triangle's motion - rejected where the surface changed: depth 5%, normal 0.8). The history is clamped to this frame's samples around the pixel (5x5:
// their mean +- 1.5 standard deviations), so a shadow that moves (the player's flashlight) does not trail behind.
[numthreads(8, 8, 1)]
void cs_light_temporal(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint cur = g_frame & 1, prev = cur ^ 1;
	const uint i = id.y * size.x + id.x;
	const float raw = asfloat(Data.Load(light_dword(LIGHT_RAW) + i * 4));
	const float d = Depth[id.xy];
	if (raw < 0.0 || d >= 1.0)
	{
		Data.Store4(light_quad(LIGHT_ACC + cur) + i * 16, asuint(float4(1.0, 1.0, 0.0, 0.0)));
		return;
	}
	float m1 = 0.0, m2 = 0.0, taps = 0.0;
	[unroll]
	for (int y = -2; y <= 2; ++y)
	{
		[unroll]
		for (int x = -2; x <= 2; ++x)
		{
			const int2 q = clamp(int2(id.xy) + int2(x, y), int2(0, 0), int2(size) - 1);
			const float v = asfloat(Data.Load(light_dword(LIGHT_RAW) + (q.y * size.x + q.x) * 4));
			if (v >= 0.0)
			{
				m1 += v;
				m2 += v * v;
				taps += 1.0;
			}
		}
	}
	m1 /= taps;                                             // at least the pixel itself
	const float spread = 1.5 * sqrt(max(m2 / taps - m1 * m1, 0.0)) + 0.05;
	const float3 n = decode_normal(GBuffer0[id.xy]);
	const float3 p = camera_relative_position(float2(id.xy), d);
	const float3 clip = previous_clip(id.xy, p);           // (ndc.xy, view depth) last frame (v20: the triangle's motion)
	float4 history = 0.0;
	float weight = 0.0;
	if (clip.z > 0.0 && Data.Load4(FRAME_INFO).x > 1)
	{
		const float2 ndc = clip.xy;
		const float2 fpos = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * float2(size) - 0.5;
		const int2 base = int2(floor(fpos));
		const float2 f = fpos - base;
		[unroll]
		for (int k = 0; k < 4; ++k)
		{
			const int2 q = base + int2(k & 1, k >> 1);
			if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y))
				continue;
			const uint j = q.y * size.x + q.x;
			const float4 h = asfloat(Data.Load4(light_quad(LIGHT_ACC + prev) + j * 16));
			if (h.w <= 0.0 || abs(h.w - clip.z) > 0.05 * clip.z + 0.05 || dot(unpack_normal(Data.Load(light_dword(LIGHT_NRM + prev) + j * 4)), n) < 0.8)
				continue;                                   // a different surface (or no light) there last frame
			const float w = ((k & 1) ? f.x : 1.0 - f.x) * ((k >> 1) ? f.y : 1.0 - f.y);
			history += h * w;
			weight += w;
		}
	}
	float4 acc = float4(raw, raw * raw, 1.0, view_z(d));
	if (weight > 0.01)
	{
		history /= weight;
		const float frames = min(history.z + 1.0, LIGHT_HISTORY);
		const float a = 1.0 / frames;
		acc.x = lerp(clamp(history.x, m1 - spread, m1 + spread), raw, a);
		acc.y = lerp(history.y, raw * raw, a);
		acc.z = frames;
	}
	Data.Store4(light_quad(LIGHT_ACC + cur) + i * 16, asuint(acc));
	Data.Store(light_dword(LIGHT_NRM + cur) + i * 4, pack_normal(n));
}

// Variance of the accumulated visibility (LIGHT_RAYS rays per frame: at least their binomial variance), as sun_variance
float light_variance(float4 acc)
{
	const float frames = max(acc.z, 1.0);
	return (max(max(acc.y - acc.x * acc.x, acc.x * (1.0 - acc.x) / LIGHT_RAYS), 0.0) + 0.25 / (LIGHT_RAYS * frames)) / frames;
}

// Edge-aware filter of the visibility: LIGHT_FILTER_PASSES a-trous passes (B3 kernel, steps 1, 2, 4) with depth and normal
// weights and a visibility weight scaled by the estimate's standard deviation (SVGF-style, as cs_atrous): noise in the
// penumbrae is smoothed, converged shadow edges stay sharp
[numthreads(8, 8, 1)]
void cs_light_filter(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint cur = g_frame & 1, src = (g_pass - 1) & 1, dst = g_pass & 1;
	const bool first = g_pass == 0;
	const uint i = id.y * size.x + id.x;
	const float4 acc = asfloat(Data.Load4(light_quad(LIGHT_ACC + cur) + i * 16));
	if (acc.w <= 0.0)
	{
		Data.Store(light_dword(LIGHT_FS + dst) + i * 4, asuint(1.0));
		Data.Store(light_dword(LIGHT_FV + dst) + i * 4, 0u);
		return;
	}
	const float s0 = first ? acc.x : asfloat(Data.Load(light_dword(LIGHT_FS + src) + i * 4));
	const float var0 = first ? light_variance(acc) : asfloat(Data.Load(light_dword(LIGHT_FV + src) + i * 4));
	const float sigma = 4.0 * sqrt(var0) + 0.02;
	const float3 n = unpack_normal(Data.Load(light_dword(LIGHT_NRM + cur) + i * 4));
	const int step = int(1u << g_pass);
	const float kernel[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };
	float sum = 0.0, sum_w = 0.0, sum_var = 0.0;
	[unroll]
	for (int y = -2; y <= 2; ++y)
	{
		[unroll]
		for (int x = -2; x <= 2; ++x)
		{
			const int2 q = int2(id.xy) + int2(x, y) * step;
			if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y))
				continue;
			const uint j = q.y * size.x + q.x;
			const float4 aq = asfloat(Data.Load4(light_quad(LIGHT_ACC + cur) + j * 16));
			if (aq.w <= 0.0)
				continue;
			const float sq = first ? aq.x : asfloat(Data.Load(light_dword(LIGHT_FS + src) + j * 4));
			const float vq = first ? light_variance(aq) : asfloat(Data.Load(light_dword(LIGHT_FV + src) + j * 4));
			const float w = kernel[abs(x)] * kernel[abs(y)] * exp(-abs(aq.w - acc.w) / (0.02 * acc.w * step + 0.01)) *
				pow(saturate(dot(unpack_normal(Data.Load(light_dword(LIGHT_NRM + cur) + j * 4)), n)), 16.0) * exp(-abs(sq - s0) / sigma);
			sum += sq * w;
			sum_var += vq * w * w;
			sum_w += w;
		}
	}
	const float inv = 1.0 / max(sum_w, 1e-6);
	Data.Store(light_dword(LIGHT_FS + dst) + i * 4, asuint(sum * inv));
	Data.Store(light_dword(LIGHT_FV + dst) + i * 4, asuint(sum_var * inv * inv));
}

// The lighting's final factor on its whole result (lights included): an occlusion boost, a stencil class tint, a distance
// lift (c[49], c[74], c[75..76], read as CS_DeferredLighting does)
float3 lighting_factor(uint2 px, float d)
{
	float3 factor = 1.0;
	if (c[49].z > 0.0)
	{
		const float a = min(InT16.SampleLevel(SamplerS3, (float2(px) + 0.5) * c[8].zw, 0.0).y, ((GBuffer1[px].w >> 2) & 63u) / 63.0);
		factor *= 1.0 + (1.0 - a) * a;
	}
	if ((InT23[px].x & 192u) == 128u)
		factor *= 1.0 + c[74].x * float3(2.0, 1.4, 1.0);
	if (c[76].y > 0.0)
	{
		const float h = saturate((length(camera_relative_position(float2(px), d)) - c[75].z) * c[75].w);
		if (h > 0.0)
		{
			const float k = c[76].x != 0.0 ? (dot(InT4[px].rgb, float3(0.3, 0.59, 0.11)) - c[76].x) / -c[76].x : 0.0;
			factor *= lerp(1.0, exp2(max(0.0, k) * c[76].y), h);
		}
	}
	return factor;
}

// After the game's lighting: the lights into the light buffer, shadowed by the filtered visibility, times the lighting's
// final factor
[numthreads(8, 8, 1)]
void cs_light_add(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint i = id.y * size.x + id.x;
	if (asfloat(Data.Load(light_dword(LIGHT_RAW) + i * 4)) < 0.0)
		return;                                             // no light here
	const float3 sum = unpack_r11g11b10(Data.Load(light_dword(LIGHT_SUM) + i * 4));
	const float vis = saturate(asfloat(Data.Load(light_dword(LIGHT_FS) + i * 4)));
	Light[id.xy] = float4(Light[id.xy].rgb + sum * lerp(1.0, vis, g_light_shadows) * lighting_factor(id.xy, Depth[id.xy]), 1.0);
}

// ---- full path tracing (v19) ----

// The sun's colour at a point as the lighting blends it: zones around c[46] by distance (toward c[45], then c[44]) and by
// height (back to c[7]); c[49].x on characters, c[48].w on the stencil 32 class
float3 sun_colour(float3 pabs, bool character, uint stencil)
{
	float3 col = c[7].xyz;
	if (c[46].w > 0.0)
	{
		const float r1 = c[81].z * c[46].w, r2 = c[81].y * c[46].w;
		const float dist = length(pabs - c[46].xyz);
		if (r1 > 0.0 && dist >= r1)
		{
			const float t = saturate((dist - r1) / c[82].x);
			col = lerp(col, c[45].xyz, t * t * t);
		}
		if (r2 > 0.0 && dist >= r2)
		{
			const float t = saturate((dist - r2) / c[82].x);
			col = lerp(col, c[44].xyz, t * t * t);
		}
		if (c[81].w > 0.0 && pabs.y >= c[81].w)
		{
			const float t = saturate((pabs.y - c[81].w) / c[82].x);
			col = lerp(col, c[7].xyz, t * t * t);
		}
	}
	return character ? col * c[49].x : (stencil & 32u) != 0 ? col * c[48].w : col;
}

// The lighting's sun factor apart from its shadow (t10.x): the material's sun byte (G-buffer 1 .y low byte), a sun mask
// (t11) close to the camera, a cavity term on non-characters, a height term (c[77].w)
float sun_factor(surface s, uint4 g1, uint2 px, float2 uv)
{
	const float receive = (g1.y & 0xFFu) / 255.0;
	if (receive == 0.0)
		return 0.0;
	const float mask = max(InT11[px].x - 1.0 / 255.0, 0.0) * (255.0 / 254.0);
	float f = saturate(receive - 0.004 + saturate(length(s.p) * c[48].y + c[48].z) * (mask - 1.0));
	if (!s.character)
	{
		const float a = ((g1.w >> 2) & 63u) / 63.0;
		f *= saturate(2.0 * a * a - 1.0 + abs(dot(c[6].xyz, s.n)));
	}
	if (asuint(c[77].w) != 0 && s.p.y + c[9].y < c[78].x)
	{
		const float e = 1.0 - saturate(c[6].y);
		f *= InT10.SampleLevel(SamplerS0, uv, 0.0).z * 12.5 * (0.95 - 0.95 * e * e * e * e * e);
	}
	return f;
}

// ---- v23: the path tracer's light at its hits - NEE of the sun and the game's lamps at every bounce (Remix-style), SHaRC
// for what the camera did not see, MIS for the sky ----

// SHaRC: a spatially hashed radiance cache (the idea of NVIDIA's SHaRC, own implementation). World-space voxels - their
// size grows with the distance to the camera (SHARC_SCALE of it, powers of two of SHARC_VOXEL), the normal's dominant axis
// and sign keep the two faces of a wall apart - in a hash table: non-zero 32-bit tags in buckets of SHARC_BUCKET entries,
// a fixed-point accumulation per frame (atomic adds) and the resolved light (blended over up to SHARC_MAX_FRAMES frames;
// an entry without samples for SHARC_STALE frames is freed). It holds the light LEAVING a surface, as if diffuse. Written
// by the camera's own view (cs_history: the lit result of a sixteenth of the pixels each frame - every light, the sun, the
// indirect light) and by the rays (the estimate at a hit the camera did not see); read at such hits - off screen, behind the
// camera, and at the second bounce, so the cache carries the bounces beyond.
#define SHARC_CAPACITY (1u << 20)   // entries (tlou_rt.cpp k_sharc_capacity)
#define SHARC_BUCKET 8
#define SHARC_FIXED 16384.0         // fixed point of the accumulation (per unit of radiance) ...
#define SHARC_RADIANCE_MAX 128.0    // ... and the largest radiance a sample adds (no overflow below 2000 samples a frame)
#define SHARC_MAX_FRAMES 32u
#define SHARC_STALE 120u
#define SHARC_VOXEL 0.05            // metres at level 0
#define SHARC_SCALE 0.02            // the voxel size aimed at, per metre of distance to the camera
#define SHARC_NONE 0xFFFFFFFFu

uint sharc_offset()                                         // keys, then the accumulation (16 B), then the resolved light (8 B)
{
	return align_up(motion_offset() + full_size().x * full_size().y * MOTION_BYTES, 256);
}

void sharc_key(float3 pabs, float3 n, out uint bucket, out uint tag)
{
	const float level = clamp(round(log2(max(length(pabs - c[9].xyz) * (SHARC_SCALE / SHARC_VOXEL), 1e-6))), 0.0, 15.0);
	const int3 g = int3(floor(pabs / (SHARC_VOXEL * exp2(level))));
	const float3 a = abs(n);
	const uint axis = a.x >= a.y ? (a.x >= a.z ? 0u : 2u) : (a.y >= a.z ? 1u : 2u);
	const uint face = axis * 2u + (n[axis] < 0.0 ? 1u : 0u);
	const uint3 u = uint3(g) & 0x7FFFFu;                    // 19 bits per axis: 64-bit key x | y | z | level | face
	const uint lo = u.x | (u.y << 19);
	const uint hi = (u.y >> 13) | (u.z << 6) | (uint(level) << 25) | (face << 29);
	bucket = (pcg(lo ^ pcg(hi)) % (SHARC_CAPACITY / SHARC_BUCKET)) * SHARC_BUCKET;
	tag = pcg(hi + pcg(lo + 0x9E3779B9u)) | 1u;
}

// The entry of a voxel (inserted on request); SHARC_NONE: not there, or its bucket is full
uint sharc_find(float3 pabs, float3 n, bool insert)
{
	uint bucket, tag;
	sharc_key(pabs, n, bucket, tag);
	const uint keys = sharc_offset();
	for (uint i = 0; i < SHARC_BUCKET; ++i)
		if (Data.Load(keys + (bucket + i) * 4) == tag)
			return bucket + i;
	if (insert)
		for (uint j = 0; j < SHARC_BUCKET; ++j)
		{
			uint original;
			Data.InterlockedCompareExchange(keys + (bucket + j) * 4, 0u, tag, original);
			if (original == 0u || original == tag)
				return bucket + j;
		}
	return SHARC_NONE;
}

void sharc_add(float3 pabs, float3 n, float3 radiance)
{
	const uint e = sharc_find(pabs, n, true);
	if (e == SHARC_NONE)
	{
		Data.InterlockedAdd(STATS + STAT_SHARC_FULL * 4, 1);
		return;
	}
	const uint3 q = uint3(clamp(radiance, 0.0, SHARC_RADIANCE_MAX) * SHARC_FIXED + 0.5);
	const uint at = sharc_offset() + SHARC_CAPACITY * 4 + e * 16;
	Data.InterlockedAdd(at, q.x);
	Data.InterlockedAdd(at + 4, q.y);
	Data.InterlockedAdd(at + 8, q.z);
	Data.InterlockedAdd(at + 12, 1u);
}

bool sharc_get(float3 pabs, float3 n, out float3 radiance)
{
	radiance = 0.0;
	const uint e = sharc_find(pabs, n, false);
	if (e == SHARC_NONE)
		return false;
	const uint2 r = Data.Load2(sharc_offset() + SHARC_CAPACITY * 20 + e * 8);
	radiance = unpack_r11g11b10(r.x);
	return (r.y & 0xFFu) != 0u;
}

// The sun the path tracing uses: the game's (c[7] and its zones; v97: always - the Sun slider's own sun is gone)
float3 pt_sun_colour(float3 pabs, bool character, uint stencil)
{
	return sun_colour(pabs, character, stencil);
}

// A ray toward the add-on's own sun: 1 when nothing blocks it. Cut-out cards (leaves, grass, fences, curtains) and hair
// (non-opaque instances, cs_convert): v30 their masks (cutout_transmission: leaf-shaped shadows); a card without one lets
// CUTOUT_TRANSMISSION through (v24-v29 every card: a fixed fraction - v23 let a random half through, drawn again every
// frame, and the light behind curtains and leaves moved about).
float sun_visible(float3 origin, float3 dir, float tmax)
{
	RayDesc ray;
	ray.Origin = origin + c[9].xyz;
	ray.TMin = 0.0;
	ray.Direction = dir;
	ray.TMax = tmax;
	RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
	q.TraceRayInline(Scene, RAY_FLAG_NONE, MASK_SCENE | MASK_CUTOUT | MASK_HAIR | MASK_KEPT, ray);   // v94: + the kept geometry
	float t = 1.0;
	while (q.Proceed())
		if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
		{
			t *= cutout_transmission(SCENE_DRAW_CANDIDATE(q), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics(),
				length(origin + dir * q.CandidateTriangleRayT()), q.CandidateInstanceContributionToHitGroupIndex());
			if (t < 0.05)
				q.CommitNonOpaqueTriangleHit();             // a solid texel, or several cards deep: blocked
		}
	return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : t;
}

// A surface for the shadow maps at a hit off the G-buffer (no screen terms, no cavity)
surface hit_surface(float3 p_rel, float3 n)
{
	surface s;
	s.p = p_rel;
	s.n = n;
	s.v = -normalize(p_rel);
	s.albedo = 0.0;
	s.roughness = 1.0;
	s.metallic = 0.0;
	s.spec = 0.0;
	s.stencil = 0u;
	s.character = false;
	s.px = uint2(0, 0);
	s.view_depth = length(p_rel);
	s.cavity = 1.0;
	s.light_occlusion = 1.0;
	return s;
}

// The game's lamps at a bounce hit (NEE, as Remix does at every path vertex): RIS over 4 candidates drawn uniformly from
// the lights the tiles name this frame (cs_light_census' list), weighted by their unshadowed diffuse light, and one shadow
// for the pick - the game's map (one tap) for lights with one, else a ray. Returns the incoming light x saturate(N.L)
// (the hit's albedo is applied outside); RIS keeps it unbiased (v23_hit_sim.py)
float3 lamps_at(float3 hit_rel, float3 n, uint seed)
{
	const uint count = min(Data.Load(LIGHT_LIST_COUNT), LIGHT_LIST_MAX);
	if (g_lights_on == 0 || count == 0)
		return 0.0;
	float total = 0.0, picked = 0.0;
	float3 f_pick = 0.0, l_pick = 0.0;
	float d_pick = 0.0;
	uint pick = 0;
	[loop]
	for (uint k = 0; k < 4; ++k)
	{
		const uint index = Data.Load(LIGHT_LIST + (pcg(seed + k * 0x68E31DA4u) % count) * 4);
		const LightDesc L = Lights[index];
		// diffuse on; not off (13 / 26), not for characters only (14, 15, stencil 32 / 64 gated: 5, 18)
		if ((L.flags & 0x80u) == 0 || (L.flags & 0x404E020u) != 0)
			continue;
		float3 color, l;
		float dist;
		if (!light_arrival(L, hit_rel, color, l, dist))
			continue;
		const float3 f = color * saturate(dot(n, l));
		const float w = dot(f, float3(0.2126, 0.7152, 0.0722));
		total += w;
		if (w > 0.0 && random1(seed + k * 0x9E3779B9u + 17u) * total < w)
		{
			pick = index;
			f_pick = f;
			picked = w;
			l_pick = (L.flags & 0x4Fu) == 0x41u ? -L.axis : l;
			d_pick = dist;
		}
	}
	if (!(picked > 0.0))
		return 0.0;
	const LightDesc L = Lights[pick];
	float vis = 1.0;
	if (L.shadow_map != -1)
		vis = game_shadow_map(L, hit_surface(hit_rel, n), false);
	else
	{
		const float tmax = d_pick - max(LIGHT_SKIP, d_pick * 0.05);
		vis = tmax > 0.0 ? trace_visible(hit_rel, l_pick, tmax) : 1.0;
	}
	return f_pick * (vis * float(count) * total / (4.0 * picked));
}

// Light leaving a hit toward the ray (v23, Remix-style). What the camera saw there last frame (the game's full lighting);
// else SHaRC's light at that voxel (three times in four; the fourth refreshes it); else - from the scene data - the
// triangle's cached material lit by the sun (NEE: a shadow ray), the game's lamps (NEE: lamps_at) and one continuation ray
// (the open sky as the shading point's baked ambient sees it; a hit the camera saw, or SHaRC's light there; else nothing
// more). That estimate goes into SHaRC, which carries it over the frames and so the bounces beyond.
// stat: SHaRC lookups, found, estimates written
float3 hit_light(uint2 px, float3 origin, float3 dir, scene_hit h, float3 a4, float3 a5, float3 axis, float2 xi, uint seed, inout uint3 stat)
{
	const float3 hit = origin + dir * h.t;
	float3 seen;
	const uint state = seen_light(hit, dir, seen);
	if (state == 1)
		return seen;
	if (state == 2)
		return 0.0;
	float3 n = hit_normal(h.draw, h.primitive);
	if (dot(n, dir) > 0.0)
		n = -n;                                             // toward the ray's origin
	const float3 habs = hit + c[9].xyz;
	float3 cached;
	const bool known = sharc_get(habs, n, cached);
	stat.xy += uint2(1u, known ? 1u : 0u);
	if (known && (pcg(seed) & 3u) != 0u)
		return cached;
	const float3 from = hit + n * (0.01 + length(hit) * 0.001);
	float3 e = 0.0;                                         // irradiance / pi at the hit
	const float ndl = dot(n, c[6].xyz);
	const float3 sun = pt_sun_colour(habs, false, 0u);
	if (ndl > 0.0 && any(sun > 0.0))
		e += sun * (ndl * trace_visible(from, c[6].xyz, HIT_SUN_RAY_LENGTH));
	e += lamps_at(from, n, seed + 0x51u);
	const float3 d2 = cosine_direction(n, xi);
	const scene_hit h2 = trace_scene(from, d2, g_ray_length);
	if (!h2.hit)
		e += escaped_light(px, from, d2, a4, a5, axis);
	else
	{
		const float3 hit2 = from + d2 * h2.t;
		float3 l2;
		if (seen_light(hit2, d2, l2) == 1)
			e += l2;
		else
		{
			float3 n2 = hit_normal(h2.draw, h2.primitive);
			if (dot(n2, d2) > 0.0)
				n2 = -n2;
			float3 c2;
			if (sharc_get(hit2 + c[9].xyz, n2, c2))
				e += c2;                                    // the cache: the bounces beyond
		}
	}
	const float3 radiance = hit_material(h.draw, h.primitive).rgb * e;
	sharc_add(habs, n, radiance);
	stat.z += 1u;
	return known ? cached : radiance;
}

// GGX visible-normal sampling (Heitz 2018): a microfacet normal for the view direction v (toward the camera) around n
float3 sample_ggx_vndf(float3 v, float3 n, float alpha, float2 xi)
{
	const float3 t = normalize(abs(n.y) < 0.999 ? cross(n, float3(0.0, 1.0, 0.0)) : cross(n, float3(1.0, 0.0, 0.0)));
	const float3 b = cross(n, t);
	const float3 vl = float3(dot(v, t), dot(v, b), dot(v, n));
	const float3 vh = normalize(float3(alpha * vl.x, alpha * vl.y, max(vl.z, 1e-4)));
	const float lensq = vh.x * vh.x + vh.y * vh.y;
	const float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
	const float3 t2 = cross(vh, t1);
	const float r = sqrt(xi.x);
	const float phi = 6.2831853 * xi.y;
	const float p1 = r * cos(phi);
	const float s = 0.5 * (1.0 + vh.z);
	const float p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * (r * sin(phi));
	const float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
	const float3 ne = normalize(float3(alpha * nh.x, alpha * nh.y, max(nh.z, 0.0)));
	return normalize(t * ne.x + b * ne.y + n * ne.z);
}

// ---- v38: ReSTIR GI (Ouyang et al. 2021) on the cosine rays' hits ----
// Each half-resolution pixel keeps a reservoir: one path vertex (a hit of its cosine rays: position, normal, the light it
// sends toward the pixel), its RIS weight W and sample count M. cs_trace_rt resamples the frame's hits (target: luminance x
// cosine; the cosine pdf makes each candidate's weight pi x luminance) and merges last frame's reservoir at the
// reprojected pixel (temporal reuse, history capped); cs_restir merges RESTIR_NEIGHBOURS neighbours' (spatial reuse; a
// neighbour's sample has to be visible from the pixel - a ray - else the pixel keeps its own) and writes the bounce light
// B = L cos / pi x W. Moving a sample to another visible point takes the solid-angle Jacobian of the reconnection.
#define RESTIR_BYTES 48
#define RESTIR_M_CAP 16.0       // temporal history cap (the light at the samples changes: the feedback loop, moving lights)
#define RESTIR_NEIGHBOURS 4
#define RESTIR_RADIUS 10.0      // half-resolution pixels
#define RESTIR_MAX 32.0         // the largest bounce light (luminance) a pixel's estimate may take: no fireflies

uint restir_offset(uint parity)                             // after SHaRC, two reservoir planes (ping-pong by frame parity)
{
	return align_up(sharc_offset() + SHARC_CAPACITY * 28u, 256) + parity * align_up(g_half_w * g_half_h * RESTIR_BYTES, 256);
}
struct reservoir
{
	float3 xs;                  // the sample: a hit point (world) ...
	uint ns;                    // ... its normal (pack_normal)
	float3 L;                   // ... the light it sends toward the visible point
	float W;                    // RIS weight
	float M;                    // samples behind it (0 = empty)
	float3 xv;                  // the visible point the sample was found from (world, the rays' origin) ...
	uint nv;                    // ... and its normal
};
void store_reservoir(uint parity, uint2 hp, reservoir r)
{
	const uint at = restir_offset(parity) + (hp.y * g_half_w + hp.x) * RESTIR_BYTES;
	Data.Store4(at, uint4(asuint(r.xs), r.ns));
	Data.Store4(at + 16, uint4(f32tof16(r.L.x) | (f32tof16(r.L.y) << 16), f32tof16(r.L.z), asuint(r.W), asuint(r.M)));
	Data.Store4(at + 32, uint4(asuint(r.xv), r.nv));
}
reservoir load_reservoir(uint parity, uint2 hp)
{
	const uint at = restir_offset(parity) + (hp.y * g_half_w + hp.x) * RESTIR_BYTES;
	const uint4 a = Data.Load4(at), b = Data.Load4(at + 16), e = Data.Load4(at + 32);
	reservoir r;
	r.xs = asfloat(a.xyz);
	r.ns = a.w;
	r.L = float3(f16tof32(b.x), f16tof32(b.x >> 16), f16tof32(b.y));
	r.W = asfloat(b.z);
	r.M = asfloat(b.w);
	r.xv = asfloat(e.xyz);
	r.nv = e.w;
	// never initialised memory or a broken value: empty
	if (!(r.M > 0.0 && r.M <= 64.0) || !(r.W >= 0.0 && r.W < 1e6) || !all(isfinite(r.L)) || any(r.L < 0.0) || !all(isfinite(r.xs)) || !all(isfinite(r.xv)))
		r.M = 0.0;
	return r;
}
float restir_lum(float3 c)
{
	return dot(c, float3(0.2126, 0.7152, 0.0722));
}
// The target at visible point x (camera-relative) with normal n: luminance x cosine toward the sample
float restir_target(float3 L, float3 xs, float3 x, float3 n)
{
	const float3 d = xs - x;
	return restir_lum(L) * max(dot(n, d) * rsqrt(max(dot(d, d), 1e-8)), 0.0);
}
// A sample found from visible point "from", reconnected to "to" (camera-relative): the solid-angle Jacobian
float restir_jacobian(float3 xs, float3 ns, float3 from, float3 to)
{
	const float3 a = from - xs, b = to - xs;
	const float la = max(dot(a, a), 1e-8), lb = max(dot(b, b), 1e-8);
	const float ca = abs(dot(ns, a)) * rsqrt(la), cb = abs(dot(ns, b)) * rsqrt(lb);
	if (ca < 1e-3)
		return 0.0;
	return clamp((cb / ca) * (la / lb), 0.0, 10.0);
}

// cs_trace_rt's part: the frame's candidates (one picked, sum of their weights) merged with last frame's reservoir
void restir_initial_temporal(uint2 hp, uint2 px, float3 p, float3 origin, float3 n, float sum, float3 L, float3 dir, float t, uint draw, uint prim, uint seed)
{
	const uint cur = g_frame & 1u, prev = cur ^ 1u;
	float wsum = sum;
	float M = HEMI_RAYS;
	float3 xs = origin, ns = n, Ls = 0.0;
	if (sum > 0.0)
	{
		xs = origin + dir * t;
		ns = hit_normal(draw, prim);
		if (dot(ns, dir) > 0.0)
			ns = -ns;
		Ls = L;
	}
	const float3 clip = previous_clip(px, p);              // (ndc.xy, view depth) last frame
	if (clip.z > 0.0 && Data.Load4(FRAME_INFO).x > 2)
	{
		const int2 q = int2(floor(float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float2(g_half_w, g_half_h)));
		if (all(q >= 0) && q.x < int(g_half_w) && q.y < int(g_half_h))
		{
			const float4 geo = load_px(PLANE_GEO + prev, uint2(q));
			if (geo.x > 0.0 && abs(geo.x - clip.z) <= 0.05 * clip.z + 0.05 && dot(geo.yzw, n) >= 0.8)
			{
				const reservoir rp = load_reservoir(prev, uint2(q));
				if (rp.M > 0.0)
				{
					const float3 xsp = rp.xs - c[9].xyz;
					const float3 nsp = unpack_normal(rp.ns);
					const float m = min(rp.M, RESTIR_M_CAP);
					const float w = restir_target(rp.L, xsp, origin, n) * rp.W * m * restir_jacobian(xsp, nsp, rp.xv - c[9].xyz, origin);
					M += m;
					if (w > 0.0)
					{
						wsum += w;
						if (random1(seed + 0x7E3Au) * wsum < w)
						{
							xs = xsp;
							ns = nsp;
							Ls = rp.L;
						}
					}
				}
			}
		}
	}
	const float target = restir_target(Ls, xs, origin, n);
	reservoir r;
	r.xs = xs + c[9].xyz;
	r.ns = pack_normal(ns);
	r.L = Ls;
	r.W = target > 0.0 ? wsum / (M * target) : 0.0;
	r.M = M;
	r.xv = origin + c[9].xyz;
	r.nv = pack_normal(n);
	store_reservoir(cur, hp, r);
}

// ---- v39: the game's emissive surfaces as lights ----
// The light buffer holds the emissive surfaces' own light before CS_DeferredLighting adds the lighting (lamps, screens,
// fires: written while it was a render target). cs_emit keeps it per pixel (EMIT plane) before the lighting runs; cs_emit_list
// picks the brightest emissive pixel of each cell of a 32 x 32 screen grid as a point light (world position, normal, mean
// radiance of the cell's emissive samples, the area they cover); cs_full lights every surface from that list (RIS over
// EMIT_CANDIDATES, one shadow ray: lamps cast ray-traced shadows) and puts the emitters' own light back (full path tracing
// replaced it with the rest of the game's result). The camera's history leaves the emitters' own light out, so the bounce
// rays do not count it twice. Gate: when half the screen or more holds light before the lighting, the buffer is not the
// emitters' (an old frame) and all of this stays off.
// v64 took the light buffer before the lighting for baked light + the lamps' glow and dropped the gate. WRONG (v65): none of
// the 44 CS_DeferredLighting variants reads its u0 (each overwrites it), and the frame map shows nothing writes the buffer
// before them - the game alternates two light buffers, so before the lighting one holds the frame from two frames back,
// the add-on's own composite included. v64 kept that as the pixels' glow and as emitters: its own output came back every
// two frames and grew without bound (user's v64 shot: the bar blinding red). v65 ran cs_emit / cs_emit_list after the game's
// lighting and took a pixel whose light was 12x the frame's mean as emissive - lit faces and the hot spots beside the lamps'
// game lights passed too (user's v65 shot: hard-edged bright patches on Ellie's face and clothes, the room over-lit).
// v66: the game's OWN emission. Nine of the 44 CS_DeferredLighting variants (the tiles holding emissive pixels) add
// hsv(hue = x & 255, saturation = x >> 8 & 255) x f16(z) of G-buffer render target 6 (t2: zero in other materials; .y is
// another material parameter) to the lighting's sum, before its final factor (lighting_factor) - the bar's globes: about
// (8.5, 6.5, 2.2) in the light buffer, their ambient inputs ordinary (capture inputs-20261012-185504). cs_emit decodes it
// per pixel (game_emission x lighting_factor: light-buffer units, emission_at) - the composites add it as the game does,
// the gather leaves it out (it is sampled as a light) - and cs_emit_list scans every pixel of each of the EMIT_GRID x
// EMIT_GRID cells (a group each; v39-v63: 8 random samples - small lamps came and went) and lists the cell's emissive area
// as one emitter at its brightest pixel (mean radiance, total area). No threshold, no frame mean.
// v67 (user: "once a light leaves the camera it stops lighting"): those emitters were this frame's screen cells only. Now a
// cell's emissive pixels within LAMP_VOXEL of its brightest one are an observation (cs_emit_list: area-weighted centre,
// normal, mean radiance, area), and cs_lamp_merge keeps the lamps in world space: LAMP_SLOTS voxels of LAMP_VOXEL in a hash
// table (buckets of LAMP_BUCKET, one thread per bucket - no atomics), each frame's observations of a voxel summed and
// blended into what it held (radiance and place a quarter; the area at once when larger, slowly when smaller - a lamp half
// off the screen keeps its size). A voxel not seen this frame keeps its lamp - off the screen or behind something it still
// lights - unless its centre is in view with nothing in front of it and no emission at its pixel: the lamp went out or
// away, its area fades. A camera jump of LAMP_JUMP (a load, a cut) forgets them all. The emitter list the paths sample
// (emitter_irradiance, ReSTIR DI) holds the live lamps within LAMP_RANGE.
#define EMIT_MAX 1024u
#define EMIT_GRID 32u
#define EMIT_MIN 1e-4
#define EMIT_BYTES 32u
#define EMIT_CANDIDATES 8u      // v67: was 4 (the remembered lamps join the list)
#define EMIT_MAX_IRRADIANCE 64.0
#define EMIT_OBS_MAX 1024u      // v67: this frame's observations (at most one per cell)
#define LAMP_SLOTS 4096u        // v67: the lamps' memory - voxels ...
#define LAMP_BUCKET 8u          //   ... in buckets of 8 (cs_lamp_merge: a thread each) ...
#define LAMP_BYTES 96u          //   ... of 96 bytes: tag, frame last seen, centre (absolute), normal, radiance, area, 2 spare,
                                //   then this frame's sums (radiance x area, area | centre x area | normal x area)
#define LAMP_VOXEL 0.4          // metres
#define LAMP_RANGE 40.0         // metres from the camera within which a remembered lamp is listed
#define LAMP_JUMP 25.0          // metres the camera moves in one frame that forget every lamp
#define LAMP_DARK_FRAMES 15u    // v102: frames in a row a lamp's place must show in plain view without emission before it fades
#define LAMP_DARK_REACH 2       // v102: ... within this many pixels of its centre
#define LAMP_HIDDEN 0x80000000u // v105: the record's + 44 bit 31 - the camera's ray to the lamp was blocked this frame
#define LAMP_HIDDEN_WEIGHT 0.1  // v105: ... its weight in RTXDI's power PDF (never 0: its light still comes, unbiased)
float emit_skip(float area, float len);

uint emit_offset()                                          // full-resolution plane (packed R11G11B10) after the reservoirs
{
	return restir_offset(2);
}
// v76: two emitter lists, this frame's and last frame's (by frame parity) - RTXDI's temporal reuse looks lamps up in last
// frame's list (rtxdi_bridge.hlsli). An entry's last word is its lamp's slot in the lamps' memory + 1 (0: none)
#define EMIT_LIST_BYTES (16u + EMIT_MAX * EMIT_BYTES)
uint emit_list_base()
{
	return emit_offset() + align_up(full_size().x * full_size().y * 4, 256);
}
uint emit_list_offset()                                     // header (listed, -, on, surface pixels), entries (v67: the remembered lamps)
{
	return emit_list_base() + (g_frame & 1u) * EMIT_LIST_BYTES;
}
uint emit_list_prev_offset()                                // v76: last frame's
{
	return emit_list_base() + ((g_frame & 1u) ^ 1u) * EMIT_LIST_BYTES;
}
uint emit_obs_offset()                                      // v67: this frame's observations (count, entries as the list's)
{
	return emit_list_base() + 2u * EMIT_LIST_BYTES;
}
uint lamp_offset()                                          // v67: the lamps' memory (tlou_rt.cpp layout_for)
{
	return align_up(emit_obs_offset() + 16u + EMIT_OBS_MAX * EMIT_BYTES, 256);
}
// v76: per lamp slot, where this frame's list put its lamp (index + 1; 0: not listed) and LAMP_MAP_NEW when the slot took
// a new lamp this frame - two maps by frame parity, after the lamps' memory (RTXDI's light index translation)
#define LAMP_MAP_NEW 0x80000000u
uint lamp_map_offset(uint parity)
{
	return lamp_offset() + LAMP_SLOTS * LAMP_BYTES + parity * LAMP_SLOTS * 4u;
}
// v66: the game's emission at a pixel, raw (before the lighting's factor): G-buffer render target 6 as CS_DeferredLighting's
// emissive variants decode it - hue (x bits 0-7, x 6 / 255) and saturation (x bits 8-15) to RGB, times the f16 in z
float3 game_emission(uint2 px)
{
#ifdef UE_GAME
	return ue_game_emission(px);                        // UE RT v24: the base pass's scene colour (ue_game.hlsli)
#else
	const uint4 g = InT2[px];
	const float v = f16tof32(g.z & 0xFFFFu);
	if (!(v > 0.0) || !isfinite(v))
		return 0.0;
	const float h = float(g.x & 255u) * (6.0 / 255.0), s = float((g.x >> 8) & 255u) * (1.0 / 255.0);
	const float3 hue = saturate(float3(abs(h - 3.0) - 1.0, 2.0 - abs(h - 2.0), 2.0 - abs(h - 4.0)));
	return ((hue - 1.0) * s + 1.0) * v;
#endif
}
// v64: a pixel's own glow - (v66) the game's emission there, in the light buffer's units (cs_emit; 0 where none)
float3 emission_at(uint2 px)
{
	return unpack_r11g11b10(Data.Load(emit_offset() + (px.y * full_size().x + px.x) * 4));
}

[numthreads(8, 8, 1)]
void cs_emit(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (all(id.xy == 0u))
	{
		Data.Store(emit_list_offset(), 0u);                 // v64: this frame's emitters (v67: cs_lamp_merge lists them) ...
		Data.Store(emit_obs_offset(), 0u);                  // ... v67: and observations (cs_emit_list)
	}
	if (id.x >= size.x || id.y >= size.y)
		return;
	// v66: the game's emission x the lighting's factor, as the light buffer holds it (the sky: not an emitter); the game's result
	// holds it whole where its lighting added it - more than that result: render target 6 means something else there
	const float d = Depth[id.xy];
	const float3 e = d < 1.0 ? game_emission(id.xy) * lighting_factor(id.xy, d) : 0.0;
	const bool on = all(isfinite(e)) && restir_lum(e) > EMIT_MIN && restir_lum(max(Light[id.xy].rgb, 0.0)) >= 0.9 * restir_lum(e);
	Data.Store(emit_offset() + (id.y * size.x + id.x) * 4, on ? pack_r11g11b10(min(e, 60000.0)) : 0u);
}

groupshared float4 gs_emit_sum[64];                         // v64: radiance x area (rgb), area
groupshared float4 gs_emit_pos[64];                         // v67: camera-relative position x area
groupshared float4 gs_emit_nrm[64];                         // v67: normal x area
groupshared float2 gs_emit_best[64];                        // brightest luminance, its pixel (asfloat)
groupshared uint gs_emit_pixels[64];
[numthreads(64, 1, 1)]
void cs_emit_list(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	const uint2 size = full_size();
	if (gid.x == 0 && gi == 0)
		Data.Store2(emit_list_offset() + 8, uint2(g_emit > 0.0 ? 1u : 0u, size.x * size.y));   // on, pixels (v66: no threshold)
	const uint2 cell = uint2(gid.x % EMIT_GRID, gid.x / EMIT_GRID);
	const uint2 lo = cell * size / EMIT_GRID, hi = (cell + 1u) * size / EMIT_GRID;
	const uint2 ext = max(hi - lo, uint2(1u, 1u));
	// the cell's brightest emissive pixel ...
	float best = 0.0;
	uint best_px = 0u;
	for (uint i = gi; i < ext.x * ext.y; i += 64u)
	{
		const uint2 q = lo + uint2(i % ext.x, i / ext.x);
		const float l = restir_lum(emission_at(q));
		if (l > EMIT_MIN && l > best && Depth[q] < 1.0)
		{
			best = l;
			best_px = q.x | (q.y << 16);
		}
	}
	gs_emit_best[gi] = float2(best, asfloat(best_px));
	GroupMemoryBarrierWithGroupSync();
	[unroll]
	for (uint stride = 32u; stride > 0u; stride >>= 1)
	{
		if (gi < stride && gs_emit_best[gi + stride].x > gs_emit_best[gi].x)
			gs_emit_best[gi] = gs_emit_best[gi + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	if (!(gs_emit_best[0].x > 0.0))
		return;                                             // no emission in the cell (the whole group returns)
	const uint anchor_px = asuint(gs_emit_best[0].y);
	const uint2 aq = uint2(anchor_px & 0xFFFFu, anchor_px >> 16);
	const float3 anchor = camera_relative_position(float2(aq), Depth[aq]);
	// ... v67: and the emissive pixels within LAMP_VOXEL of it - one lamp (another lamp in the cell: another cell's brightest,
	// or seen another frame): radiance x area, position x area, normal x area, area
	float4 sum = 0.0, pos = 0.0, nrm = 0.0;
	uint pixels = 0u;
	for (uint j = gi; j < ext.x * ext.y; j += 64u)
	{
		const uint2 q = lo + uint2(j % ext.x, j / ext.x);
		const float3 b = emission_at(q);
		if (!(restir_lum(b) > EMIT_MIN))
			continue;
		const float d = Depth[q];
		if (d >= 1.0)
			continue;
		++pixels;                                           // (every emissive pixel, for the status line)
		const float3 p = camera_relative_position(float2(q), d);
		if (length(p - anchor) > LAMP_VOXEL)
			continue;
		const float3 n = decode_normal(GBuffer0[q]);
		const float3 px1 = camera_relative_position(float2(q) + float2(1.0, 0.0), d), py1 = camera_relative_position(float2(q) + float2(0.0, 1.0), d);
		const float a = length(px1 - p) * length(py1 - p) / max(abs(dot(n, normalize(p))), 0.2);
		if (!(a > 0.0) || !isfinite(a))
			continue;
		sum += float4(b * a, a);
		pos.xyz += p * a;
		nrm.xyz += n * a;
	}
	gs_emit_sum[gi] = sum;
	gs_emit_pos[gi] = pos;
	gs_emit_nrm[gi] = nrm;
	gs_emit_pixels[gi] = pixels;
	GroupMemoryBarrierWithGroupSync();
	[unroll]
	for (uint half_stride = 32u; half_stride > 0u; half_stride >>= 1)
	{
		if (gi < half_stride)
		{
			gs_emit_sum[gi] += gs_emit_sum[gi + half_stride];
			gs_emit_pos[gi] += gs_emit_pos[gi + half_stride];
			gs_emit_nrm[gi] += gs_emit_nrm[gi + half_stride];
			gs_emit_pixels[gi] += gs_emit_pixels[gi + half_stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (gi != 0)
		return;
	Data.InterlockedAdd(STATS + STAT_EMIT_PIXELS * 4, gs_emit_pixels[0]);
	const float area = gs_emit_sum[0].w;
	if (!(area > 0.0))
		return;
	const float3 e = gs_emit_sum[0].rgb / area;
	const float3 centre = gs_emit_pos[0].xyz / area + c[9].xyz;
	const float3 nsum = gs_emit_nrm[0].xyz;
	const float3 n = dot(nsum, nsum) > 1e-20 ? normalize(nsum) : -normalize(anchor);
	const uint base = emit_obs_offset();
	uint slot;
	Data.InterlockedAdd(base, 1u, slot);
	if (slot < EMIT_OBS_MAX)
	{
		const uint at = base + 16 + slot * EMIT_BYTES;
		Data.Store4(at, uint4(asuint(centre), pack_normal(n)));
		Data.Store4(at + 16, uint4(f32tof16(e.x) | (f32tof16(e.y) << 16), f32tof16(e.z), asuint(area), 0u));
	}
}

// v67: a lamp's voxel at absolute position p - its tag (never 0) and bucket
uint lamp_tag(float3 pabs, out uint bucket)
{
	const int3 g = int3(floor(pabs / LAMP_VOXEL));
	const uint h = pcg(uint(g.x) + pcg(uint(g.y) + pcg(uint(g.z) + 0x1A3Bu)));
	bucket = h % (LAMP_SLOTS / LAMP_BUCKET);
	return max(pcg(h ^ 0x5EED5EEDu), 1u);
}

// v67: the lamps' memory (see above): one thread per bucket - this frame's observations of its voxels summed into them, then
// each voxel updated (seen: blended; unseen: kept, or faded where its place is in plain view without emission) and the live
// ones near enough listed for the paths. STAT_EMIT_LIST: listed | seen this frame << 16.
[numthreads(64, 1, 1)]
void cs_lamp_merge(uint3 id : SV_DispatchThreadID)
{
	const uint bucket = id.x;
	if (bucket >= LAMP_SLOTS / LAMP_BUCKET)
		return;
	const uint base = lamp_offset() + bucket * LAMP_BUCKET * LAMP_BYTES;
	const float3 cam = asfloat(Data.Load4(FRAME_CAM_CUR)).xyz, cam_prev = asfloat(Data.Load4(FRAME_CAM_PREV)).xyz;
	if (length(cam - cam_prev) > LAMP_JUMP)
		for (uint f = 0; f < LAMP_BUCKET; ++f)
			Data.Store(base + f * LAMP_BYTES, 0u);              // a load or a cut: another place
	const uint obs = emit_obs_offset();
	const uint count = min(Data.Load(obs), EMIT_OBS_MAX);
	for (uint k = 0; k < count; ++k)
	{
		const uint at = obs + 16 + k * EMIT_BYTES;
		const uint4 a = Data.Load4(at), q = Data.Load4(at + 16);
		const float3 p = asfloat(a.xyz);
		uint b;
		const uint tag = lamp_tag(p, b);
		if (b != bucket)
			continue;
		// its voxel's slot, else a free one, else the one unseen longest
		uint slot = LAMP_BUCKET, victim = 0u, victim_age = 0u;
		for (uint i = 0; i < LAMP_BUCKET; ++i)
		{
			const uint2 h = Data.Load2(base + i * LAMP_BYTES);
			if (h.x == tag)
			{
				slot = i;
				break;
			}
			const uint age = h.x == 0u ? 0xFFFFFFFFu : g_frame - h.y;
			if (age >= victim_age)
			{
				victim_age = age;
				victim = i;
			}
		}
		const uint s_at = base + (slot < LAMP_BUCKET ? slot : victim) * LAMP_BYTES;
		if (slot == LAMP_BUCKET)
		{
			Data.Store4(s_at, uint4(tag, g_frame, 0u, 0u));      // new: no lamp yet (area 0), no sums
			Data.Store4(s_at + 16, uint4(0u, 0u, 0u, 0u));
			Data.Store4(s_at + 32, uint4(0u, 0u, g_frame + 1u, 0u));   // v76: + 40 the frame it took this lamp (+ 1)
			Data.Store4(s_at + 48, uint4(0u, 0u, 0u, 0u));
			Data.Store4(s_at + 64, uint4(0u, 0u, 0u, 0u));
			Data.Store4(s_at + 80, uint4(0u, 0u, 0u, 0u));
		}
		const float A = asfloat(q.z);
		const float3 Le = float3(f16tof32(q.x), f16tof32(q.x >> 16), f16tof32(q.y));
		if (!(A > 0.0) || !all(isfinite(Le)) || !all(isfinite(p)))
			continue;
		Data.Store4(s_at + 48, asuint(asfloat(Data.Load4(s_at + 48)) + float4(Le * A, A)));
		Data.Store4(s_at + 64, asuint(asfloat(Data.Load4(s_at + 64)) + float4(p * A, 0.0)));
		Data.Store4(s_at + 80, asuint(asfloat(Data.Load4(s_at + 80)) + float4(unpack_normal(a.w) * A, 0.0)));
	}
	const float4x4 vp = load_matrix(FRAME_VP_CUR);
	const uint2 size = full_size();
	uint listed = 0u, seen = 0u;
	for (uint i = 0; i < LAMP_BUCKET; ++i)
	{
		const uint s_at = base + i * LAMP_BYTES;
		const uint map_at = lamp_map_offset(g_frame & 1u) + (bucket * LAMP_BUCKET + i) * 4u;
		Data.Store(map_at, 0u);                                 // v76: not listed (unless below)
		const uint4 h = Data.Load4(s_at);
		if (h.x == 0u)
			continue;
		float3 centre = asfloat(uint3(h.z, h.w, Data.Load(s_at + 16)));
		uint nrm = Data.Load(s_at + 20);
		float3 Le = asfloat(Data.Load3(s_at + 24));
		float area = asfloat(Data.Load(s_at + 36));
		uint last = h.y;
		uint dark = 0u;                                         // v102: frames in a row of plain view without emission
		const float4 s0 = asfloat(Data.Load4(s_at + 48));
		if (s0.w > 0.0)
		{
			// seen this frame
			const float3 oLe = s0.rgb / s0.w, oc = asfloat(Data.Load3(s_at + 64)) / s0.w, on = asfloat(Data.Load3(s_at + 80));
			const float3 onn = dot(on, on) > 1e-20 ? normalize(on) : float3(0.0, 1.0, 0.0);
			if (!(area > 0.0))
			{
				centre = oc;
				Le = oLe;
				area = s0.w;
				nrm = pack_normal(onn);
			}
			else
			{
				centre = lerp(centre, oc, 0.25);
				Le = lerp(Le, oLe, 0.25);
				area = s0.w >= area ? s0.w : lerp(area, s0.w, 0.05);
				const float3 nb = lerp(unpack_normal(nrm), onn, 0.25);
				nrm = pack_normal(dot(nb, nb) > 1e-12 ? normalize(nb) : onn);
			}
			last = g_frame;
			Data.Store4(s_at + 48, uint4(0u, 0u, 0u, 0u));
			Data.Store4(s_at + 64, uint4(0u, 0u, 0u, 0u));
			Data.Store4(s_at + 80, uint4(0u, 0u, 0u, 0u));
			++seen;                                             // (v102: not dark - dark stays 0)
		}
		else if (area > 0.0)
		{
			// not seen: off the screen or behind something, it is kept; in view with nothing in front of it and no emission at
			// its pixel, the lamp went out or away - it fades. v102 (user, video 2026-10-15 08-13-38: the greenhouse's hanging
			// bulbs - "lamps seem to be constantly on an on/off state, once detected, the lamps should stay on"): only after
			// LAMP_DARK_FRAMES frames in a row of that, with no emission within LAMP_DARK_REACH pixels of its centre (the count
			// at + 44). v67-v101 faded it the first frame its centre pixel alone showed none: a bulb a few pixels wide that DLSS's
			// jitter moved off that pixel, or not observed because a brighter lamp of its screen cell was, lost 30 % of its light
			// a frame and came back whole at the next observation - the light going out and on
			const float4 clip = mul(float4(centre - c[9].xyz, 1.0), vp);
			if (clip.w > 0.05)
			{
				const float2 ndc = clip.xy / clip.w;
				if (all(abs(ndc) < 0.98))
				{
					const uint2 px = min(uint2(float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * float2(size)), size - 1u);
					const float d = Depth[px];
					const bool clear = d >= 1.0 || view_z(d) > clip.w - max(0.25, clip.w * 0.05);
					bool lit = false;
					[loop]
					for (int dy = -LAMP_DARK_REACH; dy <= LAMP_DARK_REACH && !lit; ++dy)
						[loop]
						for (int dx = -LAMP_DARK_REACH; dx <= LAMP_DARK_REACH && !lit; ++dx)
							lit = restir_lum(emission_at(uint2(clamp(int2(px) + int2(dx, dy), int2(0, 0), int2(size) - 1)))) > EMIT_MIN;
					if (clear && !lit)
					{
						dark = (Data.Load(s_at + 44) & ~LAMP_HIDDEN) + 1u;
						if (dark > LAMP_DARK_FRAMES)
							area *= 0.7;
					}
				}
			}
		}
		if (!(area > 1e-6) || !all(isfinite(Le)) || !all(isfinite(centre)))
		{
			Data.Store(s_at, 0u);                                // forgotten (or a slot claimed without a valid observation)
			continue;
		}
		Data.Store4(s_at, uint4(h.x, last, asuint(centre.xy)));
		Data.Store4(s_at + 16, uint4(asuint(centre.z), nrm, asuint(Le.xy)));
		Data.Store2(s_at + 32, uint2(asuint(Le.z), asuint(area)));
		// v105 (user, v104's captures rt-recon\inputs-20261015-131156 / -131201: "Direct light only" 20x "Indirect light only"
		// on the hall's walls and as noisy - 40-70 % per pixel, 10 % of the wall black; of 742 listed lamps 737 not seen for
		// 1600+ frames: the streets' and other rooms' lamps behind the hall's walls took most of the lamps' samples, their
		// shadow rays blocked): a listed lamp the camera's ray does not reach (to short of its own surface) is hidden - RTXDI's
		// power PDF weighs it by LAMP_HIDDEN_WEIGHT (rtxdi_di.hlsl cs_rtxdi_pdf), so the lamps' samples go to the lamps in view
		// or open to it; never 0 - one around a corner or through a door still lights what it lights (RTXPT's NEE-AT learns
		// which lights contribute from its shadow rays the same way)
		bool hidden = false;
		const bool listed_lamp = g_emit > 0.0 && any(Le > 0.0) && length(centre - cam) < LAMP_RANGE;
		if (listed_lamp)
		{
			const float3 rel = centre - c[9].xyz;
			const float len = length(rel);
			const float tmax = len - emit_skip(area, len);
			hidden = tmax > 0.0 && !(trace_visible(0.0, rel / len, tmax) > 0.0);
		}
		Data.Store(s_at + 44, dark | (hidden ? LAMP_HIDDEN : 0u));
		if (listed_lamp)
		{
			uint slot;
			Data.InterlockedAdd(emit_list_offset(), 1u, slot);
			if (slot < EMIT_MAX)
			{
				const uint at = emit_list_offset() + 16 + slot * EMIT_BYTES;
				const uint lamp = bucket * LAMP_BUCKET + i;
				Data.Store4(at, uint4(asuint(centre), nrm));
				Data.Store4(at + 16, uint4(f32tof16(Le.x) | (f32tof16(Le.y) << 16), f32tof16(Le.z), asuint(area), lamp + 1u));
				Data.Store(map_at, (slot + 1u) | (Data.Load(s_at + 40) == g_frame + 1u ? LAMP_MAP_NEW : 0u));   // v76
			}
			++listed;
		}
	}
	if (listed + seen != 0u)
		Data.InterlockedAdd(STATS + STAT_EMIT_LIST * 4, listed | (seen << 16));
}

// v65: an emitter's area x its falloff - a disc of that area seen on its axis: A / (d^2 + A / pi) (v39-v64: A / max(d^2, 0.04),
// a point: a cell's whole emissive area at its brightest pixel gave a surface beside it far more than pi x its radiance,
// the most a lamp of any size can give)
float emit_falloff(float area, float len)
{
	return area / (len * len + area * 0.31830989);
}
// v67: how far short of an emitter's centre its shadow ray stops - its own surface lies around the centre (a globe's glass
// up to about its radius: sqrt(area / pi) of its visible area), v39-v66 0.1 m or 5% of the distance
float emit_skip(float area, float len)
{
	return max(max(0.1, len * 0.05), sqrt(max(area, 0.0) * 0.31830989) + 0.05);
}

// Irradiance at p (camera-relative, normal n, geometric normal ng) from the listed emitters: RIS over EMIT_CANDIDATES
// uniform picks (target: the unshadowed irradiance's luminance), one shadow ray for the pick. Emitters radiate as a
// cardioid around their normal (the camera sees bulbs from one side; their light reaches the walls beside them).
float3 emitter_irradiance(float3 p, float3 n, float3 ng, uint seed)
{
	const uint base = emit_list_offset();
	const uint4 h = Data.Load4(base);
	const uint count = min(h.x, EMIT_MAX);
	if (h.z == 0u || count == 0u || g_emit <= 0.0)
		return 0.0;
	float total = 0.0, picked = 0.0, d_pick = 0.0, a_pick = 0.0;
	float3 f_pick = 0.0, l_pick = 0.0;
	[loop]
	for (uint k = 0; k < EMIT_CANDIDATES; ++k)
	{
		const uint at = base + 16 + (pcg(seed + k * 0x68E31DA4u) % count) * EMIT_BYTES;
		const uint4 a = Data.Load4(at), b = Data.Load4(at + 16);
		const float3 d = asfloat(a.xyz) - c[9].xyz - p;
		const float len = length(d);
		if (!(len > 0.05))
			continue;                                       // the emitter itself
		const float3 l = d / len;
		const float cs = dot(n, l);
		if (cs <= 0.0)
			continue;
		const float ce = 0.5 + 0.5 * dot(unpack_normal(a.w), -l);
		const float3 Le = float3(f16tof32(b.x), f16tof32(b.x >> 16), f16tof32(b.y));
		const float3 f = Le * (ce * cs * emit_falloff(asfloat(b.z), len));
		const float w = restir_lum(f);
		if (!(w > 0.0) || !isfinite(w))
			continue;
		total += w;
		if (random1(seed + k * 0x9E3779B9u + 23u) * total < w)
		{
			f_pick = f;
			picked = w;
			l_pick = l;
			d_pick = len;
			a_pick = asfloat(b.z);
		}
	}
	if (!(picked > 0.0))
		return 0.0;
	const float3 o = p + ng * (0.01 + length(p) * 0.002);
	const float tmax = d_pick - emit_skip(a_pick, d_pick);
	const float vis = tmax > 0.0 ? trace_visible(o, l_pick, tmax) : 1.0;
	float3 e = f_pick * (vis * float(count) * total / (float(EMIT_CANDIDATES) * picked));
	const float le = restir_lum(e);
	if (le > EMIT_MAX_IRRADIANCE)
		e *= EMIT_MAX_IRRADIANCE / le;
	return e;
}

// ---- v42: ReSTIR DI (Bitterli et al. 2020) for the sun, the game's lights and the emitters ----
// Full path tracing's direct light. Each full-resolution pixel keeps a reservoir holding one light sample: the sun (a
// direction in the Sun softness cone), a game light of the pixel's tile (its index, checked against its position the
// next frame) or an emitter (its point). cs_di draws candidates (the sun, DI_LIGHT_CANDIDATES tile lights and
// DI_EMITTER_CANDIDATES emitters, RIS with the target = the sample's unshadowed light at the pixel, the game's BRDF and,
// for its lights, its own shadow maps) and merges last frame's reservoir at the reprojected pixel; cs_full merges
// DI_NEIGHBOURS neighbours' (spatial reuse), traces one shadow ray for the pick and adds f x V x W - in place of the sun
// term, the tile lights' sum (cs_lights) and the emitters' RIS of v39. The sample format is the light vertex ReSTIR PT
// reconnects to.
#define DI_BYTES 48
#define DI_M_CAP 20.0
#define DI_NEIGHBOURS 3
#define DI_RADIUS 12.0
#define DI_LIGHT_CANDIDATES 2u
#define DI_EMITTER_CANDIDATES 4u        // v67: was 2 (the remembered lamps join the list)
#define DI_MAX 256.0            // the largest direct light (luminance) a pixel's estimate may take
#define DI_NONE 0u
#define DI_SUN 1u
#define DI_LIGHT 2u
#define DI_EMITTER 3u

uint di_offset(uint parity)                                 // after the lamps' memory (v67; the emitters' list before), two planes
{
	const uint2 size = full_size();
	return align_up(lamp_map_offset(2), 256) + parity * align_up(size.x * size.y * DI_BYTES, 256);   // v76: after the lamps' maps
}
struct di_reservoir
{
	uint kind;                  // DI_*
	uint index;                 // a game light's index in Lights
	float3 a;                   // sun: direction; game light: its position (world); emitter: its position (world)
	uint n;                     // emitter: normal
	float3 Le;                  // emitter: radiance
	float area;                 // emitter: area
	float W;
	float M;
	float vz;                   // the pixel's view depth and normal (reuse checks)
	uint nv;
};
void store_di(uint parity, uint2 px, di_reservoir r)
{
	const uint at = di_offset(parity) + (px.y * full_size().x + px.x) * DI_BYTES;
	Data.Store4(at, uint4((r.kind << 30) | (r.index & 0x3FFFFFFFu), asuint(r.a)));
	Data.Store4(at + 16, uint4(r.n, f32tof16(r.Le.x) | (f32tof16(r.Le.y) << 16), f32tof16(r.Le.z), asuint(r.area)));
	Data.Store4(at + 32, uint4(asuint(r.W), asuint(r.M), asuint(r.vz), r.nv));
}
di_reservoir load_di(uint parity, uint2 px)
{
	const uint at = di_offset(parity) + (px.y * full_size().x + px.x) * DI_BYTES;
	const uint4 q0 = Data.Load4(at), q1 = Data.Load4(at + 16), q2 = Data.Load4(at + 32);
	di_reservoir r;
	r.kind = q0.x >> 30;
	r.index = q0.x & 0x3FFFFFFFu;
	r.a = asfloat(q0.yzw);
	r.n = q1.x;
	r.Le = float3(f16tof32(q1.y), f16tof32(q1.y >> 16), f16tof32(q1.z));
	r.area = asfloat(q1.w);
	r.W = asfloat(q2.x);
	r.M = asfloat(q2.y);
	r.vz = asfloat(q2.z);
	r.nv = q2.w;
	if (!(r.M > 0.0 && r.M <= 64.0) || !(r.W >= 0.0 && r.W < 1e8) || !all(isfinite(r.a)) || !all(isfinite(r.Le)) ||
		!(r.area >= 0.0 && r.area < 1e6) || !isfinite(r.vz))
		r.kind = DI_NONE;                                   // never written, or broken
	return r;
}

// The sample's light at surface s, unshadowed by the rays (game lights keep their own maps / occlusion, as eval_light),
// in the light buffer's units before the lighting's factor (emitters, which skip that factor, divided by post); v47: diff
// its diffuse part (the RE9 renderer's skin blur)
float3 di_eval(di_reservoir r, surface s, float3 post, out float3 diff)
{
	diff = 0.0;
	if (r.kind == DI_SUN)
		return 0.0;                                         // v70: the sun is the renderers' own term (DD2's, re9.hlsl cs_re9_path) -
	                                                        //   a sun sample still in last frame's reservoirs brings nothing
	if (r.kind == DI_LIGHT)
		return 0.0;                                         // v71: the game's lights are DD2's every-light term (re9.hlsl
	                                                        //   cs_re9_path) - a light sample still in last frame's reservoirs
	                                                        //   brings nothing; ReSTIR DI draws the lamps (emitters) alone
	if (r.kind == DI_EMITTER)
	{
		if (g_emit <= 0.0 || Data.Load(emit_list_offset() + 8) == 0u)
			return 0.0;
		const float3 d = r.a - c[9].xyz - s.p;
		const float len = length(d);
		if (!(len > 0.05))
			return 0.0;
		const float3 l = d / len;
		const float cs = dot(s.n, l);
		if (cs <= 0.0)
			return 0.0;
		const float ce = 0.5 + 0.5 * dot(unpack_normal(r.n), -l);
		diff = s.albedo * ((1.0 - s.metallic) * 0.31830989 * g_emit) * r.Le * (ce * cs * emit_falloff(r.area, len)) / post;   // v65
		return diff;
	}
	return 0.0;
}
float3 di_eval(di_reservoir r, surface s, float3 post)
{
	float3 diff;
	return di_eval(r, s, post, diff);
}

// Streaming RIS: candidate x with weight w (target p) replaces the pick with probability w / sum
void di_offer(inout di_reservoir y, inout float wsum, inout float target, di_reservoir x, float w, float p, float u)
{
	if (!(w > 0.0) || !isfinite(w))
		return;
	wsum += w;
	if (u * wsum < w)
	{
		y = x;
		target = p;
	}
}

// v75: the share of the shadow ray from o (camera-relative, off the surface) that reaches the emitter sample r - di_shade's
// ray (1 when it would end before it starts: inside the emitter's own surface); cs_di's checks take the same ray
float di_visibility(float3 o, di_reservoir r)
{
	const float3 dd = r.a - c[9].xyz - o;
	const float len = length(dd);
	const float tmax = len - emit_skip(r.area, len);       // v67
	return tmax > 0.0 ? trace_visible(o, dd / len, tmax) : 1.0;
}

[numthreads(8, 8, 1)]
void cs_di(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const uint cur = g_frame & 1u, prev = cur ^ 1u;
	di_reservoir y = (di_reservoir)0;
	const float d = Depth[id.xy];
	if (d >= 1.0)
	{
		store_di(cur, id.xy, y);                            // the sky: none (v46: hair has reservoirs too, for the RE9 renderer)
		return;
	}
	surface s = load_surface(id.xy, d);
	const bool hair = (pixel_flags(id.xy) & PIXEL_HAIR) != 0;
	if (hair)
		face_viewer(s);
	s.n = visible_normal(s.n, s.v, hair ? s.n : geometric_normal(id.xy, s.p, s.n));   // v69: as the RE9 renderer shades them
	const float3 post = max(lighting_factor(id.xy, d) * g_brightness, 1e-4);
	const uint seed = pcg(id.x + pcg(id.y + pcg(g_frame + 0xD1D1u)));
	float wsum = 0.0, target = 0.0;
	// (v42-v69: with the Sun slider above 0 the add-on's sun was a candidate here - v70: the RE9 renderer lights with DD2's
	// sun term instead, every frame; the user found this one "extremely broken and flickery". v42-v70: DI_LIGHT_CANDIDATES of
	// the tile's game lights - v71: DD2's every-light term lights them; the lamps (emitters) remain ReSTIR DI's)
	{
		const uint base = emit_list_offset();
		const uint4 h = Data.Load4(base);
		const uint count = h.z != 0u ? min(h.x, EMIT_MAX) : 0u;
		for (uint k = 0; k < DI_EMITTER_CANDIDATES && count != 0 && g_emit > 0.0; ++k)
		{
			const uint at = base + 16 + (pcg(seed + 0x9E37u + k * 0x2C1B3C6Du) % count) * EMIT_BYTES;
			const uint4 a = Data.Load4(at), b = Data.Load4(at + 16);
			di_reservoir x = (di_reservoir)0;
			x.kind = DI_EMITTER;
			x.a = asfloat(a.xyz);
			x.n = a.w;
			x.Le = float3(f16tof32(b.x), f16tof32(b.x >> 16), f16tof32(b.y));
			x.area = asfloat(b.z);
			const float p = restir_lum(di_eval(x, s, post));
			di_offer(y, wsum, target, x, p * float(count) / float(DI_EMITTER_CANDIDATES), p, random1(seed + 0x200u + k));
		}
	}
	// v72 (user: "RE9's reuse toggle (not the restir pt one) improves Path tracing visuals by ALOT, Much more accurate shadowing,
	// etc, but it comes with ALOT of noise aswell, near impossible to see anything type of noise"): visibility reuse (Bitterli
	// et al. 2020) - with "Reuse light samples (RE9)" on, the lamp the candidates picked takes one shadow ray (di_shade's)
	// before it enters the reservoir, and a blocked one does not (this frame's candidates still count: M 1). v42-v71 kept and
	// passed on the lamp brightest unshadowed: in its shadow the pixels went on drawing it (black - the final ray blocked)
	// and where it was seen it came back weighted by the many (white). A reservoir left without one takes last frame's below
	const float3 o = s.p + geometric_normal(id.xy, s.p, s.n) * (0.01 + length(s.p) * 0.002);   // di_shade's ray origin
	if (y.kind == DI_EMITTER && (g_pass & 1u) == 0u)
	{
		if (!(di_visibility(o, y) > 0.0))
		{
			y = (di_reservoir)0;
			wsum = 0.0;
			target = 0.0;
		}
	}
	float M = 1.0;
	// temporal reuse: last frame's reservoir where this surface was (the motion plane), its target re-evaluated here (v67: not
	// with g_pass bit 0 - "Reuse samples (RE9)" off: fresh candidates every frame, DD2's kind of noise)
	// v69 (user: "noise pattern downwards" - the RE9 renderer's since v66): one of the four pixels around the reprojected
	// point, picked by the bilinear weights (as cs_re9_accum weighs them). v42-v68 took the pixel containing it (floor): with
	// the camera still, the point moves only by the TAA jitter's difference between the frames, whose vertical Halton (base 3)
	// steps are +1/3 px and back by -5/9 px or more - the floor never crossed to the pixel below but took the one above at
	// every jump back, a third of the frames: the samples, and their noise, flowed down the screen ~0.3 px a frame (scratchpad
	// jitter_drift.py; the v67 clip measured the image drifting down 0.1 px a frame). The bilinear pick's expectation is the
	// point itself: no drift
	const float3 clip = previous_clip(id.xy, s.p);
	if (clip.z > 0.0 && Data.Load4(FRAME_INFO).x > 2 && (g_pass & 1u) == 0u)
	{
		const float2 fpos = float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float2(size) - 0.5;
		const int2 base = int2(floor(fpos));
		const float2 f = fpos - float2(base);
		const float2 u = random3(id.xy, g_frame + 0xD17u).xy;
		const int2 q = base + int2(u.x < f.x ? 1 : 0, u.y < f.y ? 1 : 0);
		if (all(q >= 0) && q.x < int(size.x) && q.y < int(size.y))
		{
			const di_reservoir rp = load_di(prev, uint2(q));
			if (rp.kind == DI_EMITTER && abs(rp.vz - clip.z) <= 0.05 * clip.z + 0.05 && dot(unpack_normal(rp.nv), s.n) >= 0.8)
			{
				// v75 (the 2026-10-13 debug-view captures: RE9's lamps alone 0 on 74 % of Ellie's jacket under her chin, where
				// her head shadows a lamp above her - DD2's fresh picks fill it from the other lamps): last frame's lamp takes
				// the shadow ray here too, and one this pixel does not see is not reused - the history goes, its M with it (the
				// lamps it stood for are unknown here). v72 checked this frame's pick only: the brightest lamp came back through
				// the history and stayed (it outweighs the fresh picks), its final ray blocked frame after frame. The fresh
				// pick's lamp, just checked, takes no second ray. (Since v71 emitters are the only kind cs_di stores)
				bool seen = y.kind == DI_EMITTER && all(rp.a == y.a);
				if (!seen)
					seen = di_visibility(o, rp) > 0.0;
				if (seen)
				{
					const float m = min(rp.M, DI_M_CAP);
					const float p = restir_lum(di_eval(rp, s, post));
					di_offer(y, wsum, target, rp, p * rp.W * m, p, random1(seed + 0x300u));
					M += m;
				}
			}
		}
	}
	y.M = M;
	y.W = target > 0.0 ? wsum / (M * target) : 0.0;
	if (!(target > 0.0))
		y.kind = DI_NONE;
	y.vz = view_z(d);
	y.nv = pack_normal(s.n);
	store_di(cur, id.xy, y);
}

// cs_full's part: spatial reuse over this frame's reservoirs, one shadow ray, the direct light f V W (before the factor)
float3 di_shade(uint2 px, surface s, float d, float3 post, out float3 diffuse)   // v47: + its diffuse part
{
	diffuse = 0.0;
	const uint cur = g_frame & 1u;
	const uint2 size = full_size();
	di_reservoir y = load_di(cur, px);
	const di_reservoir own = y;                             // v75: the pixel's own sample (cs_di checked it from here)
	float target = y.kind != DI_NONE ? restir_lum(di_eval(y, s, post)) : 0.0;
	float wsum = target * y.W * y.M;
	float M = y.kind != DI_NONE ? y.M : 0.0;
	const float vz = view_z(d);
	const uint seed = pcg(px.x + pcg(px.y + pcg(g_frame + 0xD15Au)));
	const uint neighbours = (g_pass & 1u) != 0u ? 0u : DI_NEIGHBOURS;   // v67: none with "Reuse samples (RE9)" off (the caller's g_pass)
	[loop]
	for (uint k = 0; k < neighbours; ++k)
	{
		const float rad = DI_RADIUS * sqrt(random1(seed + k * 2u)), ang = 6.2831853 * random1(seed + k * 2u + 1u);
		const int2 q = int2(px) + int2(round(float2(cos(ang), sin(ang)) * rad));
		if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y) || all(q == int2(px)))
			continue;
		const di_reservoir rk = load_di(cur, uint2(q));
		if (rk.kind == DI_NONE || abs(rk.vz - vz) > 0.1 * vz || dot(unpack_normal(rk.nv), s.n) < 0.9)
			continue;
		const float p = restir_lum(di_eval(rk, s, post));
		M += rk.M;
		di_offer(y, wsum, target, rk, p * rk.W * rk.M, p, random1(seed + 0x40u + k));
	}
	if (!(target > 0.0) || !(M > 0.0))
		return 0.0;
	float W = wsum / (M * target);
	if (!(W < 1e30))
		return 0.0;                                         // v73: a target near 0 made W infinite (and 0 x inf, NaN, when blocked)
	float3 fd;
	float3 f = di_eval(y, s, post, fd);
	const float3 ng = geometric_normal(px, s.p, s.n);
	const float3 origin = s.p + ng * (0.01 + length(s.p) * 0.002);
	float vis = 1.0;
	if (y.kind == DI_SUN)
		vis = dot(ng, y.a) > 0.0 ? sun_visible(origin, y.a, SUN_RAY_LENGTH) : 0.0;
	else if (y.kind == DI_LIGHT)
	{
		const LightDesc L = Lights[y.index];
		if (!(L.shadow_map != -1 && s.character))           // characters keep the maps of lights that have one (cs_lights)
			vis = lerp(1.0, light_visibility(s, origin, L, random3(px, g_frame + 0x1D1u).xy), g_light_shadows);
	}
	else
	{
		vis = di_visibility(origin, y);
		// v75: a neighbour's lamp this pixel does not see - the pixel's own sample instead (cs_di checked it from here), with
		// its own weight: the light of the lamps the pixel sees. v42-v74 took 0 that frame, and the pixels around drew the
		// same neighbour's lamp: a black splotch for a frame (faces: the nose's and brows' shadows)
		if (!(vis > 0.0) && own.kind == DI_EMITTER && any(own.a != y.a))
		{
			y = own;
			W = own.W;
			f = di_eval(own, s, post, fd);
			vis = di_visibility(origin, own);
		}
	}
	float3 e = f * (vis * W);
	diffuse = fd * (vis * W);
	if (!all(isfinite(e)) || !all(isfinite(diffuse)))       // v73: nothing that is not a finite number leaves here
	{
		diffuse = 0.0;
		return 0.0;
	}
	const float le = restir_lum(e);
	if (le > DI_MAX)
	{
		e *= DI_MAX / le;
		diffuse *= DI_MAX / le;
	}
	return e;
}
float3 di_shade(uint2 px, surface s, float d, float3 post)
{
	float3 diffuse;
	return di_shade(px, s, d, post, diffuse);
}

// ---- v43: ReSTIR PT (Lin et al. 2022: generalized resampled importance sampling over paths) ----
// The half-resolution cosine rays' paths - through a hit (the light it sends back: the camera's view of it, SHaRC, or
// its own bounce, hit_light) or out to the sky (escaped_light) - are resampled into one path reservoir per pixel. Paths
// move between pixels by a hybrid shift: a hit far enough from both primaries is reconnected (the solid-angle Jacobian,
// a visibility ray), a hit closer than pt_rc_min is replayed (the same random numbers traced from the other pixel; the
// cosine mapping keeps the pdf, Jacobian 1) and a sky path keeps its direction (a visibility ray). Temporal reuse (last
// frame's reservoir at the reprojected pixel) and spatial reuse (PT_NEIGHBOURS neighbours) are weighted by pairwise MIS
// with confidence weights (unbiased, unlike v38's count weighting). cs_pt writes the bounce light B (paths through a hit)
// and the sky light E (paths to the sky) from the result: f / pi x cos x W.
#define PT_BYTES 64
#define PT_M_CAP 20.0
#define PT_NEIGHBOURS 3
#define PT_RADIUS 10.0
#define PT_NONE 0u
#define PT_RC 1u
#define PT_REPLAY 2u
#define PT_ENV 3u

uint pt_offset(uint parity)                                 // after ReSTIR DI's planes, two half-resolution planes
{
	return align_up(di_offset(2), 256) + parity * align_up(g_half_w * g_half_h * PT_BYTES, 256);
}
struct pt_sample
{
	uint kind;                  // PT_*
	float3 a;                   // hit position (world) / sky direction
	uint n;                     // hit normal (toward the primary)
	float3 L;                   // the light arriving at the primary along the path
	float2 xi;                  // the direction's random numbers (replay)
	float W;
	float M;
	float3 x0;                  // the primary the path starts at (world, off the surface) ...
	uint n0;                    // ... and its normal
};
void store_pt(uint parity, uint2 hp, pt_sample r)
{
	const uint at = pt_offset(parity) + (hp.y * g_half_w + hp.x) * PT_BYTES;
	Data.Store4(at, uint4(asuint(r.a), r.n));
	Data.Store4(at + 16, uint4(f32tof16(r.L.x) | (f32tof16(r.L.y) << 16), f32tof16(r.L.z) | (r.kind << 16), asuint(r.W), asuint(r.M)));
	Data.Store4(at + 32, uint4(asuint(r.x0), r.n0));
	const uint2 q = min(uint2(saturate(r.xi) * 65536.0), uint2(65535u, 65535u));
	Data.Store4(at + 48, uint4(q.x | (q.y << 16), 0u, 0u, 0u));
}
pt_sample load_pt(uint parity, uint2 hp)
{
	const uint at = pt_offset(parity) + (hp.y * g_half_w + hp.x) * PT_BYTES;
	const uint4 q0 = Data.Load4(at), q1 = Data.Load4(at + 16), q2 = Data.Load4(at + 32), q3 = Data.Load4(at + 48);
	pt_sample r;
	r.a = asfloat(q0.xyz);
	r.n = q0.w;
	r.L = float3(f16tof32(q1.x), f16tof32(q1.x >> 16), f16tof32(q1.y));
	r.kind = (q1.y >> 16) & 3u;
	r.W = asfloat(q1.z);
	r.M = asfloat(q1.w);
	r.x0 = asfloat(q2.xyz);
	r.n0 = q2.w;
	r.xi = (float2(q3.x & 0xFFFFu, q3.x >> 16) + 0.5) / 65536.0;
	if (!(r.M > 0.0 && r.M <= 64.0) || !(r.W >= 0.0 && r.W < 1e6) || !all(isfinite(r.L)) || any(r.L < 0.0) || !all(isfinite(r.a)) || !all(isfinite(r.x0)))
	{
		r.M = 0.0;                                          // never written, or broken
		r.kind = PT_NONE;
	}
	return r;
}
float pt_weight(uint kind)                                  // the target follows the sliders' weights of B and E
{
	return kind == PT_ENV ? max(g_sky, 0.05) : max(g_bounce, 0.05);
}
float pt_rc_min(float3 x0)                                  // closer hits are replayed, not reconnected (Jacobian blow-up)
{
	return 0.1 + 0.01 * length(x0);
}
float3 pt_direction(pt_sample s)
{
	return s.kind == PT_ENV ? s.a : normalize(s.a - s.x0);
}
float pt_target(pt_sample s)                                // in the path's own domain: luminance x cosine x the slider weight
{
	if (s.kind == PT_NONE)
		return 0.0;
	return restir_lum(s.L) * max(dot(unpack_normal(s.n0), pt_direction(s)), 0.0) * pt_weight(s.kind);
}
void pt_offer(inout pt_sample y, inout float wsum, inout float target, pt_sample x, float w, float t, float u)
{
	if (!(w > 0.0) || !isfinite(w))
		return;
	wsum += w;
	if (u * wsum < w)
	{
		y = x;
		target = t;
	}
}
// The path moved to the primary x0d (camera-relative, normal n0d): false when the shift fails (blocked, behind, too close)
bool pt_shift(pt_sample s, float3 x0d, float3 n0d, out pt_sample o, out float jac)
{
	o = s;
	jac = 1.0;
	o.x0 = x0d + c[9].xyz;
	o.n0 = pack_normal(n0d);
	if (s.kind == PT_ENV)
	{
		if (dot(n0d, s.a) <= 0.0)
			return false;
		return !trace_scene(x0d, s.a, g_ray_length).hit;
	}
	if (s.kind == PT_RC)
	{
		const float3 xr = s.a - c[9].xyz, nr = unpack_normal(s.n);
		const float3 dd = xr - x0d, ds = xr - (s.x0 - c[9].xyz);
		const float ld = length(dd), ls = length(ds);
		if (!(ld > pt_rc_min(x0d)) || !(ls > 1e-4))
			return false;
		const float3 dir = dd / ld;
		if (dot(n0d, dir) <= 0.0 || dot(nr, -dir) <= 0.0)
			return false;
		const float cs = abs(dot(nr, ds / ls));
		if (cs < 1e-3)
			return false;
		jac = min(abs(dot(nr, dir)) / cs * (ls * ls) / (ld * ld), 10.0);
		return trace_visible(x0d, dir, max(ld * 0.98 - 0.02, 0.0)) >= 0.5;
	}
	if (s.kind == PT_REPLAY)
	{
		const float3 dir = cosine_direction(n0d, s.xi);
		const scene_hit h = trace_scene(x0d, dir, g_ray_length);
		if (!h.hit)
			return false;
		const float3 hit = x0d + dir * h.t;
		float3 nh = hit_normal(h.draw, h.primitive);
		if (dot(nh, dir) > 0.0)
			nh = -nh;
		float3 L;
		if (seen_light(hit, dir, L) == 0 && !sharc_get(hit + c[9].xyz, nh, L))
			return false;                                   // no light known there without a new bounce
		o.a = hit + c[9].xyz;
		o.n = pack_normal(nh);
		o.L = L;
		return true;
	}
	return false;
}

// cs_trace_rt's part: the frame's candidate paths (one picked, the sum of their weights) and temporal reuse
void pt_initial_temporal(uint2 hp, uint2 px, float3 p, float3 origin, float3 n, pt_sample y, float wsum, float target, uint seed)
{
	const uint cur = g_frame & 1u, prev = cur ^ 1u;
	y.x0 = origin + c[9].xyz;
	y.n0 = pack_normal(n);
	y.M = 1.0;
	y.W = target > 0.0 ? wsum / (float(HEMI_RAYS) * target) : 0.0;
	if (!(target > 0.0))
		y.kind = PT_NONE;
	pt_sample r = y;
	float rt = target, M = 1.0, W = y.W;
	const float3 clip = previous_clip(px, p);
	if (clip.z > 0.0 && Data.Load4(FRAME_INFO).x > 2)
	{
		const int2 q = int2(floor(float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float2(g_half_w, g_half_h)));
		if (all(q >= 0) && q.x < int(g_half_w) && q.y < int(g_half_h))
		{
			const float4 geo = load_px(PLANE_GEO + prev, uint2(q));
			if (geo.x > 0.0 && abs(geo.x - clip.z) <= 0.05 * clip.z + 0.05 && dot(geo.yzw, n) >= 0.8)
			{
				const pt_sample rp = load_pt(prev, uint2(q));
				if (rp.M > 0.0)
				{
					// pairwise MIS of two: last frame's path moved here, this frame's path moved to last frame's primary
					const float Mp = min(rp.M, PT_M_CAP);
					pt_sample ypc = (pt_sample)0, ycp = (pt_sample)0;
					float jp = 1.0, jc = 1.0;
					const float tp_c = rp.kind != PT_NONE && pt_shift(rp, origin, n, ypc, jp) ? pt_target(ypc) * jp : 0.0;
					const float tp_p = pt_target(rp);
					const float tc_p = y.kind != PT_NONE && pt_shift(y, rp.x0 - c[9].xyz, unpack_normal(rp.n0), ycp, jc) ? pt_target(ycp) * jc : 0.0;
					const float mc = target > 0.0 ? target / (target + Mp * tc_p) : 0.0;
					const float mp = tp_p > 0.0 ? Mp * tp_p / (Mp * tp_p + tp_c) : 0.0;
					const float wc = mc * target * y.W;
					const float wp = mp * tp_c * rp.W;
					const float sum = wc + wp;
					if (wp > 0.0 && random1(seed + 0x7E3Bu) * sum < wp)
					{
						r = ypc;
						rt = pt_target(ypc);
					}
					M = 1.0 + Mp;
					W = rt > 0.0 ? sum / rt : 0.0;
				}
			}
		}
	}
	r.M = min(M, PT_M_CAP);
	r.W = W;
	if (!(rt > 0.0))
	{
		r.kind = PT_NONE;
		r.W = 0.0;
	}
	store_pt(cur, hp, r);
}

// Per half-resolution pixel (the front-most of its 2x2 block), paths from the game's G-buffer surface (Remix's primary hit):
// - HEMI_RAYS cosine rays: open sky (V; full path tracing's E) or the light of what they hit (B, hit_light)
// - SHADOW_RAYS rays toward the ambient's dominant direction D in the softness cone (sky shadow S; v23 also full path
//   tracing's light sampling of the sky lobe, combined with the cosine rays by MIS - balance heuristic, v23_mis_sim.py)
// - one ray toward the sun in the sun cone (sun visibility; 1 when the sun is below the horizon or behind the surface; the
//   add-on's own sun lets light through cut-outs at random, sun_visible)
// - one GGX reflection ray (roughness = the G-buffer byte the lighting's split-sum LUT is indexed with, alpha = r^2): the
//   light of what it hits (w = 1), or nothing hit (w = 0: the game's own reflection stays there)
[numthreads(8, 8, 1)]
void cs_trace_rt(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= g_half_w || id.y >= g_half_h)
		return;
	const uint2 px = sample_pixel(id.xy);
	const float d = Depth[px];
	float4 bv = float4(0.0, 0.0, 0.0, 1.0), refl = 0.0;
	float3 env = 0.0;                                       // full path tracing's sky / surroundings E (v19; v23 MIS)
	float3 env_rays = 0.0;                                  // v43: the cosine rays' part of E (ReSTIR PT resamples it)
	float s = 1.0, sun = 1.0;
	uint2 counts = 0;
	uint3 stat = 0;
	uint2 sky_stat = 0;                                     // v28: escaped rays with Baked light < 1, of them with the sky probe
	if (d < 1.0)
	{
		const uint4 g0 = GBuffer0[px];
		const float3 n = decode_normal(g0);
		const float3 axis = decode_ambient_direction(GBuffer1[px]);
		const float3 p = camera_relative_position(float2(px), d);
		const float3 ng = geometric_normal(px, p, n);           // v24: the rays leave the true surface
		const float3 origin = p + ng * (0.01 + length(p) * 0.002);
		const float3 a4 = InT4[px].rgb, a5 = InT5[px].rgb;
		const float2 rotation = random3(id.xy, 0).xy, rotation2 = random3(id.xy, 1).xy;
		const uint seed = pcg(id.x + pcg(id.y + pcg(g_frame + 0xC0FFEEu)));
		// MIS between the cosine rays (pdf cos / pi) and the cone rays toward D (uniform in the cone), for the sky lobe
		const bool cone = dot(n, axis) > 0.0;              // else the lighting gives the lobe no weight here: no cone rays
		const float pdf_cone = 1.0 / (6.2831853 * max(1.0 - g_cone_cos, 1e-5));
		bv = 0.0;
		float rs_sum = 0.0, rs_t = 0.0;                     // v38: ReSTIR GI's pick among the hits
		float3 rs_L = 0.0, rs_dir = 0.0;
		uint rs_draw = 0, rs_prim = 0;
		pt_sample pt_y = (pt_sample)0;                       // v43: ReSTIR PT's pick among the paths
		float pt_sum = 0.0, pt_t = 0.0;
		[loop]
		for (uint k = 0; k < HEMI_RAYS; ++k)
		{
			const float2 xi = sequence(g_frame * HEMI_RAYS + k, rotation);
			const float3 dir = cosine_direction(n, xi);
			const scene_hit h = trace_scene(origin, dir, g_ray_length);
			if (!h.hit)
			{
				bv.w += 1.0;
				const float pc = HEMI_RAYS * saturate(dot(dir, n)) * 0.31830989;
				const float pl = cone && dot(dir, axis) >= g_cone_cos ? SHADOW_RAYS * pdf_cone : 0.0;
				const float3 baked = a4 + 1.5 * a5 * saturate(dot(dir, axis)) * (pc / max(pc + pl, 1e-12));
				float3 esc = baked;                             // v43: this ray's light (its share of E)
				if (g_baked < 1.0)
				{
					// v28: g_baked of the baked estimate (this ray's MIS share; the cone rays' share below is scaled too), the
					// rest the open sky - or, without a sky probe, the baked ambient this ray alone sees (cosine sampling)
					float3 open;
					const bool sky = sky_env(px, dir, open);
					sky_stat += uint2(1u, sky ? 1u : 0u);
					esc = baked * g_baked + (sky ? open : a4 + 1.5 * a5 * saturate(dot(dir, axis))) * (1.0 - g_baked);
				}
				env += esc;
				if ((g_pass & 2u) != 0u)                        // v43: ReSTIR PT candidate - a path to the sky
				{
					pt_sample x = (pt_sample)0;
					x.kind = PT_ENV;
					x.a = dir;
					x.L = esc;
					x.xi = xi;
					pt_offer(pt_y, pt_sum, pt_t, x, 3.14159265 * restir_lum(esc) * pt_weight(PT_ENV), restir_lum(esc) * saturate(dot(dir, n)) * pt_weight(PT_ENV),
						random1(seed + k * 0x2C1B3C6Du + 0x7A11u));
				}
			}
			else
			{
				counts += uint2(material_slot(h.draw, h.primitive) != 0 ? 1u : 0u, 1u);
				const float3 L = hit_light(px, origin, dir, h, a4, a5, axis, sequence(g_frame * HEMI_RAYS + k + 1024u, rotation2), seed + k * 0x1000193u, stat);
				bv.rgb += L;
				const float w = 3.14159265 * restir_lum(L);    // v38: candidate weight (target / cosine pdf)
				rs_sum += w;
				if (w > 0.0 && random1(seed + k * 0x2C1B3C6Du + 0x5E57u) * rs_sum < w)
				{
					rs_L = L;
					rs_dir = dir;
					rs_t = h.t;
					rs_draw = h.draw;
					rs_prim = h.primitive;
				}
				if ((g_pass & 2u) != 0u)                        // v43: ReSTIR PT candidate - a path through this hit
				{
					float3 nh = hit_normal(h.draw, h.primitive);
					if (dot(nh, dir) > 0.0)
						nh = -nh;
					pt_sample x = (pt_sample)0;
					x.kind = h.t > pt_rc_min(origin) ? PT_RC : PT_REPLAY;
					x.a = origin + dir * h.t + c[9].xyz;
					x.n = pack_normal(nh);
					x.L = L;
					x.xi = xi;
					pt_offer(pt_y, pt_sum, pt_t, x, 3.14159265 * restir_lum(L) * pt_weight(PT_RC), restir_lum(L) * saturate(dot(dir, n)) * pt_weight(PT_RC),
						random1(seed + k * 0x2C1B3C6Du + 0x7A12u));
				}
			}
		}
		bv /= HEMI_RAYS;
		env /= HEMI_RAYS;
		env_rays = env;
		if ((g_pass & 2u) != 0u)                            // v43: ReSTIR PT on (cs_pt writes B and E)
			pt_initial_temporal(id.xy, px, p, origin, n, pt_y, pt_sum, pt_t, seed);
		else if ((g_pass & 1u) != 0u)                       // v38: ReSTIR GI on (cs_restir writes B)
			restir_initial_temporal(id.xy, px, p, origin, n, rs_sum, rs_L, rs_dir, rs_t, rs_draw, rs_prim, seed);
		if (cone)
		{
			s = 0.0;
			float3 lobe = 0.0;
			[unroll]
			for (uint j = 0; j < SHADOW_RAYS; ++j)
			{
				const float3 dir = cone_direction(axis, g_cone_cos, sequence(g_frame * SHADOW_RAYS + j + 2048u, rotation.yx));
				const float cosn = dot(dir, n);
				const float v = cosn > 0.0 ? trace_visible(origin, dir, g_ray_length) : 0.0;   // below the surface: its own shadow
				s += v;
				const float pc = HEMI_RAYS * cosn * 0.31830989, pl = SHADOW_RAYS * pdf_cone;
				lobe += v * (1.5 * a5 * saturate(dot(dir, axis))) * (cosn * 0.31830989 / pdf_cone) * (pl / (pc + pl));
			}
			s /= SHADOW_RAYS;
			env += lobe * (g_baked / SHADOW_RAYS);              // v28: the baked lobe's MIS share, as Baked light
		}
		if (c[6].y > 0.0)
		{
			// v24: the side by the true surface (ng): turned away = its own shadow (v19-v23: "lit" by the normal maps' details)
			sun = 0.0;
			if (dot(ng, c[6].xyz) > 0.0)
			{
				const float3 dir = cone_direction(c[6].xyz, g_sun_cos, sequence(g_frame + 3072u, rotation2.yx));
				sun = dot(dir, ng) > 0.0 ? trace_visible(origin, dir, SUN_RAY_LENGTH) : 0.0;
			}
		}
		const float roughness = max((g0.w & 0xFFu) / 255.0, 0.02);
		const float3 v = -normalize(p);
		const float3 r = reflect(-v, sample_ggx_vndf(v, n, roughness * roughness, sequence(g_frame + 3584u, rotation)));
		if (dot(r, n) > 0.0)
		{
			const scene_hit h = trace_scene(origin, r, REFLECTION_RAY_LENGTH);
			if (h.hit)
				refl = float4(hit_light(px, origin, r, h, a4, a5, axis, sequence(g_frame + 3840u, rotation2), seed + 0x7777u, stat), 1.0);
		}
	}
	if (d >= 1.0 && (g_pass & 1u) != 0u)
	{
		const reservoir empty = (reservoir)0;              // v38: the sky: no reservoir
		store_reservoir(g_frame & 1u, id.xy, empty);
	}
	if (d >= 1.0 && (g_pass & 2u) != 0u)
	{
		const pt_sample none = (pt_sample)0;               // v43: the sky: no path reservoir
		store_pt(g_frame & 1u, id.xy, none);
	}
	store_px(PLANE_TRACE, id.xy, bv);
	store_px(PLANE_TRACE_S, id.xy, float4(s, sun, 0.0, 0.0));
	store_px(PLANE_TRACE_R, id.xy, refl);
	store_px(PLANE_TRACE_E, id.xy, float4((g_pass & 2u) != 0u ? env - env_rays : env, 0.0));   // v43 PT: the cone rays' share (cs_pt adds the paths')
	counts = WaveActiveSum(counts);
	stat = WaveActiveSum(stat);
	sky_stat = WaveActiveSum(sky_stat);
	if (WaveIsFirstLane())
	{
		if (counts.y != 0)
		{
			Data.InterlockedAdd(STATS + STAT_MATERIAL_KNOWN * 4, counts.x);
			Data.InterlockedAdd(STATS + STAT_MATERIAL_HITS * 4, counts.y);
		}
		if (sky_stat.x != 0)
		{
			Data.InterlockedAdd(STATS + STAT_SKY_ESCAPED * 4, sky_stat.x);
			Data.InterlockedAdd(STATS + STAT_SKY_FOUND * 4, sky_stat.y);
		}
		if (stat.x != 0)
		{
			Data.InterlockedAdd(STATS + STAT_SHARC_QUERIES * 4, stat.x);
			Data.InterlockedAdd(STATS + STAT_SHARC_FOUND * 4, stat.y);
			Data.InterlockedAdd(STATS + STAT_SHARC_WRITES * 4, stat.z);
		}
	}
}

// v38: ReSTIR GI's spatial reuse and the bounce light B it gives (see restir_initial_temporal); V stays the rays'
[numthreads(8, 8, 1)]
void cs_restir(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= g_half_w || id.y >= g_half_h)
		return;
	const uint cur = g_frame & 1u;
	const reservoir r0 = load_reservoir(cur, id.xy);
	if (!(r0.M > 0.0))
		return;
	const float3 xv = r0.xv - c[9].xyz, nv = unpack_normal(r0.nv);
	const float dist_v = length(xv);
	const uint seed = pcg(id.x + pcg(id.y + pcg(g_frame + 0x2E57u)));
	const float3 xs0 = r0.xs - c[9].xyz;
	const float w0 = restir_target(r0.L, xs0, xv, nv) * r0.W * r0.M;
	float3 xs = xs0, L = r0.L;
	float wsum = w0, M = r0.M;
	bool other = false;
	[loop]
	for (uint k = 0; k < RESTIR_NEIGHBOURS; ++k)
	{
		const float rad = RESTIR_RADIUS * sqrt(random1(seed + k * 2u)), ang = 6.2831853 * random1(seed + k * 2u + 1u);
		const int2 q = int2(id.xy) + int2(round(float2(cos(ang), sin(ang)) * rad));
		if (q.x < 0 || q.y < 0 || q.x >= int(g_half_w) || q.y >= int(g_half_h) || all(q == int2(id.xy)))
			continue;
		const reservoir rk = load_reservoir(cur, uint2(q));
		if (!(rk.M > 0.0))
			continue;
		const float3 xk = rk.xv - c[9].xyz;
		if (abs(length(xk) - dist_v) > 0.1 * dist_v || dot(unpack_normal(rk.nv), nv) < 0.9)
			continue;                                       // another surface
		const float3 xsk = rk.xs - c[9].xyz;
		const float w = restir_target(rk.L, xsk, xv, nv) * rk.W * rk.M * restir_jacobian(xsk, unpack_normal(rk.ns), xk, xv);
		M += rk.M;
		if (w > 0.0)
		{
			wsum += w;
			if (random1(seed + 0x100u + k) * wsum < w)
			{
				xs = xsk;
				L = rk.L;
				other = true;
			}
		}
	}
	if (other)
	{
		// a neighbour's sample must be in view of this pixel, else its light would come through walls: the own reservoir
		const float3 d = xs - xv;
		const float len = length(d);
		if (trace_visible(xv, d / max(len, 1e-6), max(len * 0.98 - 0.02, 0.0)) < 0.5)
		{
			xs = xs0;
			L = r0.L;
			wsum = w0;
			M = r0.M;
		}
	}
	const float3 ds = xs - xv;
	const float cosv = max(dot(nv, ds) * rsqrt(max(dot(ds, ds), 1e-8)), 0.0);
	const float target = restir_lum(L) * cosv;
	float3 B = target > 0.0 ? L * (cosv * 0.31830989 * wsum / (M * target)) : 0.0;
	const float lb = restir_lum(B);
	if (lb > RESTIR_MAX)
		B *= RESTIR_MAX / lb;
	store_px(PLANE_TRACE, id.xy, float4(B, load_px(PLANE_TRACE, id.xy).w));
}

// v43: ReSTIR PT's spatial reuse (pairwise MIS, the pixel's own reservoir canonical) and the light it gives: B for paths
// through a hit, E for paths to the sky (added to the cone rays' share cs_trace_rt left in E)
[numthreads(8, 8, 1)]
void cs_pt(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= g_half_w || id.y >= g_half_h)
		return;
	const uint cur = g_frame & 1u;
	const pt_sample c0 = load_pt(cur, id.xy);
	if (!(c0.M > 0.0))
		return;                                             // the sky: B and E as traced
	const float3 x0 = c0.x0 - c[9].xyz, n0 = unpack_normal(c0.n0);
	const float dist0 = length(x0);
	const uint seed = pcg(id.x + pcg(id.y + pcg(g_frame + 0x9E55u)));
	uint2 qs[PT_NEIGHBOURS];
	uint nk = 0;
	[loop]
	for (uint k = 0; k < PT_NEIGHBOURS; ++k)
	{
		const float rad = PT_RADIUS * sqrt(random1(seed + k * 2u)), ang = 6.2831853 * random1(seed + k * 2u + 1u);
		const int2 q = int2(id.xy) + int2(round(float2(cos(ang), sin(ang)) * rad));
		if (q.x < 0 || q.y < 0 || q.x >= int(g_half_w) || q.y >= int(g_half_h) || all(q == int2(id.xy)))
			continue;
		const pt_sample r = load_pt(cur, uint2(q));
		if (!(r.M > 0.0))
			continue;
		if (abs(length(r.x0 - c[9].xyz) - dist0) > 0.1 * dist0 || dot(unpack_normal(r.n0), n0) < 0.9)
			continue;                                       // another surface
		qs[nk] = uint2(q);
		++nk;
	}
	const float kf = max(float(nk), 1.0);
	const float tc = pt_target(c0);
	const float mcm = c0.M / kf;                            // the canonical's confidence, shared among the pairs
	pt_sample sel = c0;
	float sel_t = tc, wsum = 0.0, mc = nk == 0 ? 1.0 : 0.0;
	[loop]
	for (uint j = 0; j < nk; ++j)
	{
		const pt_sample r = load_pt(cur, qs[j]);
		const float3 xr = r.x0 - c[9].xyz, nr = unpack_normal(r.n0);
		pt_sample t1 = (pt_sample)0, t2 = (pt_sample)0;
		float j1 = 1.0, j2 = 1.0;
		const float tc_r = c0.kind != PT_NONE && pt_shift(c0, xr, nr, t1, j1) ? pt_target(t1) * j1 : 0.0;
		if (tc > 0.0)
			mc += mcm * tc / (mcm * tc + r.M * tc_r);
		const float tr_c = r.kind != PT_NONE && pt_shift(r, x0, n0, t2, j2) ? pt_target(t2) * j2 : 0.0;
		const float tr_r = pt_target(r);
		const float mr = tr_r > 0.0 ? r.M * tr_r / (r.M * tr_r + mcm * tr_c) : 0.0;
		pt_offer(sel, wsum, sel_t, t2, mr / kf * tr_c * r.W, pt_target(t2), random1(seed + 0x100u + j));
	}
	if (nk != 0)
		mc /= kf;
	pt_offer(sel, wsum, sel_t, c0, mc * tc * c0.W, tc, random1(seed + 0x200u));
	const float W = sel_t > 0.0 ? wsum / sel_t : 0.0;
	const float cosv = max(dot(n0, pt_direction(sel)), 0.0);
	float3 f = sel.L * (cosv * 0.31830989 * W);
	const float lf = restir_lum(f);
	if (!(lf <= RESTIR_MAX))
		f = lf > RESTIR_MAX ? f * (RESTIR_MAX / lf) : 0.0;
	const bool sky = sel.kind == PT_ENV;
	const float4 tr = load_px(PLANE_TRACE, id.xy), te = load_px(PLANE_TRACE_E, id.xy);
	store_px(PLANE_TRACE, id.xy, float4(sky ? float3(0.0, 0.0, 0.0) : f, tr.w));
	store_px(PLANE_TRACE_E, id.xy, float4(te.rgb + (sky ? f : float3(0.0, 0.0, 0.0)), te.w));
}

// Full path tracing: every surface pixel lit by the add-on, Remix-style - the sun (the game's sun term with the
// ray-traced visibility instead of its shadow maps), the game's lights (cs_lights), the path-traced indirect light on
// the diffuse albedo (bounce B of the cosine rays that hit, and E of those that leave the near field: the shading point's
// baked ambient in their direction, under the game's large-scale ambient shadow t10.y), with the game's small-scale
// occlusion; the specular indirect light (the reflections, path traced where the rays hit, through the game's split-sum
// LUT), skin's subsurface light as the game adds it, then the lighting's final factor. No shadow maps. Blended over the
// game's result by the Path tracing slider. v21: hair keeps the game's lighting (with the add-on's lights).
[numthreads(8, 8, 1)]
void cs_full(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const float d = Depth[id.xy];
	if (d >= 1.0 || (pixel_flags(id.xy) & PIXEL_HAIR) != 0)
		return;                                             // sky: the game's (its sky dome is drawn later); hair: see above
	const surface s = load_surface(id.xy, d);
	const uint4 g1 = GBuffer1[id.xy];
	const float2 uv = (float2(id.xy) + 0.5) * c[8].zw;
	float4 bv, refl;
	float3 env;
	float sv, sun;
	upsample(id.xy, view_z(d), s.n, bv, sv, sun, refl, env);
	float3 total = 0.0;
	const bool di = (g_pass & 1u) != 0u;                     // v42: ReSTIR DI on (cs_di ran)
	// v22: the sun's shadow is the lighting's own mask t10.x as the add-on rewrote it before the lighting - the game's
	// shadow maps and cloud shadows times the ray-traced contact visibility of solid geometry (v19-v21: the rays alone,
	// which leaked through the casters the scene lacks and made foliage cards solid)
	// v23: with the Sun slider above 0 the add-on's own sun - its ray-traced visibility only (the game's masks keep the sun
	// out of overcast and indoor scenes)
	// v24: the add-on's sun traced here, one ray per full-resolution pixel from the true surface (geometric_normal): crisp
	// shadows, no half-resolution estimate filtered over the details (the upscaler's history smooths the penumbra)
	const float vis = saturate(sun_factor(s, g1, id.xy, uv) * InT10.SampleLevel(SamplerS0, uv, 0.0).x);   // (v97: the game's sun only)
	const float3 sun_light = pt_sun_colour(s.p + c[9].xyz, s.character, s.stencil);
	if (vis > 0.0 && any(sun_light > 0.0))
	{
		const float ndl = dot(s.n, c[6].xyz);
		const float3 diffuse = saturate(ndl) * (1.0 - s.metallic) * s.albedo;
		const float3 spec = game_specular(s, c[6].xyz * 10.0, 10.0, c[7].w, max(s.roughness, c[47].w), c[47].y, ndl);
		total += sun_light * vis * (diffuse + spec);
	}
	const uint i = id.y * size.x + id.x;
	if (!di && g_lights_on != 0 && asfloat(Data.Load(light_dword(LIGHT_RAW) + i * 4)) >= 0.0)
		total += unpack_r11g11b10(Data.Load(light_dword(LIGHT_SUM) + i * 4)) * lerp(1.0, saturate(asfloat(Data.Load(light_dword(LIGHT_FS) + i * 4))), g_light_shadows);
	if (di)                                                 // v42: the sun, the game's lights, the emitters
		total += di_shade(id.xy, s, d, max(lighting_factor(id.xy, d) * g_brightness, 1e-4));
	const float sky_shadow = lerp(1.0, InT10.SampleLevel(SamplerS0, uv, 0.0).y, g_baked);   // v28: the game's baked sky shadow, as Baked light
	// v41: the sky's directional shadow at full resolution - one ray per pixel and frame toward the ambient's dominant
	// direction within the Shadow softness cone, as the sun's: the half-resolution visibility S, filtered, gives the sky
	// light soft blobs; the ray's visibility over S sharpens the lobe's share of it (ratio clamped; the upscaler's history
	// smooths the ray's noise)
	float sky_sharp = 1.0;
	{
		const float3 axis = decode_ambient_direction(g1);
		const float facing = saturate(dot(s.n, axis));
		if (facing > 0.0)
		{
			const float3 ng = geometric_normal(id.xy, s.p, s.n);
			const float3 dir = cone_direction(axis, g_cone_cos, random3(id.xy, g_frame + 0x5C7Eu).xy);
			const float v = dot(dir, ng) > 0.0 ? trace_visible(s.p + ng * (0.01 + length(s.p) * 0.002), dir, g_ray_length) : 0.0;
			const float share = SKY_LOBE_SHARE * facing / (1.0 - SKY_LOBE_SHARE + SKY_LOBE_SHARE * facing);
			sky_sharp = lerp(1.0, clamp(v / max(sv, 0.05), 0.0, 4.0), share);
		}
	}
	// v40: contact shadows for the bounce and sky light: one cosine ray per full-resolution pixel and frame up to
	// CONTACT_RANGE; an occluder that near darkens by its distance (the half-resolution light is filtered over many pixels
	// and loses the shadows under and between objects; the upscaler's history smooths the ray's noise, as the sun's)
	float contact = 1.0;
	if (g_contact > 0.0)
	{
		const float3 ng = geometric_normal(id.xy, s.p, s.n);
		const float3 dir = cosine_direction(s.n, random3(id.xy, g_frame + 0xC0DEu).xy);
		if (dot(dir, ng) > 0.0)
		{
			const scene_hit h = trace_scene(s.p + ng * (0.01 + length(s.p) * 0.002), dir, CONTACT_RANGE);
			if (h.hit)
			{
				const float close = 1.0 - saturate(h.t / CONTACT_RANGE);   // v41: squared - dark only close to the occluder
				contact = 1.0 - g_contact * close * close;
			}
		}
		else
			contact = 1.0 - g_contact;                      // below the true surface: its own shadow
	}
	total += s.albedo * ((1.0 - s.metallic) * contact) *
		(max(bv.rgb, 0.0) * g_bounce + max(env, 0.0) * (g_sky * sky_shadow * sky_sharp));
	const float2 lut = InT19.SampleLevel(SamplerS3, float2(saturate(s.roughness), saturate(dot(s.n, s.v))) * 0.96875 + 0.015625, 0.0).xy;
	const float3 f0 = lerp(s.spec * (0.08 / 255.0), s.albedo, s.metallic);
	const float k = s.character ? c[74].w : (s.stencil & 32u) != 0 ? c[75].x : c[75].y;
	total += max((f0 * lut.x + lut.y) * InT18.SampleLevel(SamplerS2, uv, 0.0).rgb * k, 0.0);
	total += InT7.SampleLevel(SamplerS3, uv, 0.0).rgb * ((1.0 - InT22.SampleLevel(SamplerS3, uv, 0.0).w) * InT16.SampleLevel(SamplerS3, uv, 0.0).y) *
		(s.character ? s.albedo : float3(1.0, 1.0, 1.0));
	// v39: the emitters' light on the surface (ray-traced shadows) and the surface's own emission, both in the light buffer's
	// units (after the lighting's factor)
	float3 lamps = 0.0, self = 0.0;
	if (Data.Load(emit_list_offset() + 8) != 0u)
	{
		self = unpack_r11g11b10(Data.Load(emit_offset() + (id.y * size.x + id.x) * 4));
		if (!di)                                            // v42: else ReSTIR DI's
			lamps = s.albedo * ((1.0 - s.metallic) * 0.31830989 * g_emit) *
			emitter_irradiance(s.p, s.n, geometric_normal(id.xy, s.p, s.n), pcg(id.x + pcg(id.y + pcg(g_frame + 0xE417u))));
	}
	Light[id.xy] = float4(lerp(Light[id.xy].rgb, total * (lighting_factor(id.xy, d) * g_brightness) + lamps + self, g_full), 1.0);
}

// After the lighting: the lit result for the next frame's hits (Light after cs_full), with the bounce boost taken out so the
// feedback converges. The light buffer holds albedo B bounce G, G the factor on the bounce - about 1 in the game's own
// lighting (t4' = t4 V + B bounce), (1 - metallic) x the game's occlusion x the lighting's final factor in cs_full, blended
// by the Path tracing slider; the history keeps albedo B min(G, 1): one physical bounce, never amplified. (v11-v20 took out
// albedo B (bounce - 1): under full path tracing the final factor (up to 1.25) and a high Bounce let the loop grow.)
[numthreads(8, 8, 1)]
void cs_history(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	float3 l = Light[id.xy].rgb;
	const float d = Depth[id.xy];
	if (d < 1.0)
	{
		const uint4 g = GBuffer0[id.xy], g1 = GBuffer1[id.xy];
		float4 bv, refl;
		float3 env;
		float s, sun;
		upsample(id.xy, view_z(d), decode_normal(g), bv, s, sun, refl, env);
		float3 gain = 1.0;
		if (g_full > 0.0 && (pixel_flags(id.xy) & PIXEL_HAIR) == 0)
		{
			const float metallic = (g1.y & 0x2000u) != 0 ? ((g1.y >> 8) & 31u) / 31.0 : 0.0;
			const float2 uv = (float2(id.xy) + 0.5) * c[8].zw;
			gain = lerp(1.0, (1.0 - metallic) * lighting_factor(id.xy, d) * g_brightness, g_full);
		}
		l = max(l - decode_albedo(g) * max(bv.rgb, 0.0) * max(g_bounce * gain - min(gain, 1.0), 0.0), 0.0);
		if (Data.Load(emit_list_offset() + 8) != 0u)       // v39: the emitters' own light reaches the rays as lamps (cs_full)
			l = max(l - unpack_r11g11b10(Data.Load(emit_offset() + (id.y * size.x + id.x) * 4)), 0.0);
		// v23: a sixteenth of the pixels (a 4x4 pattern that moves every frame) put the light the camera sees into SHaRC
		if (all((id.xy & 3u) == uint2(g_frame & 3u, (g_frame >> 2) & 3u)) && (pixel_flags(id.xy) & PIXEL_HAIR) == 0)
		{
			sharc_add(camera_relative_position(float2(id.xy), d) + c[9].xyz, decode_normal(g), l);
			Data.InterlockedAdd(STATS + STAT_SHARC_INJECTED * 4, 1);
		}
	}
	Data.Store(history_offset() + (id.y * size.x + id.x) * 4, pack_r11g11b10(min(l, 256.0)));
}

// v23: SHaRC's resolve, per entry after the frame's samples: the accumulation's mean blended into the resolved light (up to
// SHARC_MAX_FRAMES frames), the accumulation cleared; an entry without samples for SHARC_STALE frames is freed
[numthreads(64, 1, 1)]
void cs_sharc_resolve(uint3 id : SV_DispatchThreadID)
{
	uint live = 0;
	if (id.x < SHARC_CAPACITY)
	{
		const uint keys = sharc_offset(), acc_at = keys + SHARC_CAPACITY * 4 + id.x * 16, res_at = keys + SHARC_CAPACITY * 20 + id.x * 8;
		if (Data.Load(keys + id.x * 4) != 0u)
		{
			const uint4 acc = Data.Load4(acc_at);
			const uint2 res = Data.Load2(res_at);
			const uint now = g_frame & 0xFFFFFFu;
			if (acc.w != 0u)
			{
				const uint frames = min((res.y & 0xFFu) + 1u, SHARC_MAX_FRAMES);
				const float3 radiance = lerp(unpack_r11g11b10(res.x), float3(acc.xyz) / (SHARC_FIXED * float(acc.w)), 1.0 / float(frames));
				Data.Store4(acc_at, uint4(0, 0, 0, 0));
				Data.Store2(res_at, uint2(pack_r11g11b10(radiance), frames | (now << 8)));
				live = 1;
			}
			else if (((now - (res.y >> 8)) & 0xFFFFFFu) > SHARC_STALE)
			{
				Data.Store(keys + id.x * 4, 0u);
				Data.Store2(res_at, uint2(0, 0));
			}
			else
				live = 1;
		}
	}
	live = WaveActiveSum(live);
	if (WaveIsFirstLane() && live != 0)
		Data.InterlockedAdd(STATS + STAT_SHARC_LIVE * 4, live);
}

// The lights the tiles name this frame, each counted once (LIGHT_MASK), for the overlay; v23: also listed (LIGHT_LIST) for
// the bounce hits' lamps - run before the near-field rays, with the real light counts bound
[numthreads(64, 1, 1)]
void cs_light_census(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= light_tile_count())
		return;
	const uint count = min(LightCounts.Load(id.x * 4), 16u);
	if (count != 0)
		Data.InterlockedAdd(STATS + STAT_LIGHT_TILES * 4, 1);
	for (uint i = 0; i < count; ++i)
	{
		const uint index = LightIndex[id.x * 16 + i];
		Data.InterlockedMax(STATS + STAT_LIGHT_MAX_INDEX * 4, index + 1);
		if (index >= 1024)
			continue;
		uint before;
		Data.InterlockedOr(LIGHT_MASK + (index >> 5) * 4, 1u << (index & 31u), before);
		if ((before & (1u << (index & 31u))) != 0)
			continue;                                       // another tile counted it
		uint slot;
		Data.InterlockedAdd(LIGHT_LIST_COUNT, 1, slot);
		if (slot < LIGHT_LIST_MAX)
			Data.Store(LIGHT_LIST + slot * 4, index);       // v23: the bounce hits' lamps
		const LightDesc L = Lights[index];
		const uint type = L.flags & 15u;
		Data.InterlockedAdd(STATS + STAT_LIGHTS * 4, 1);
		Data.InterlockedAdd(STATS + (type == 0 ? STAT_LIGHT_POINT : type == 1 ? STAT_LIGHT_BOX : STAT_LIGHT_SPOT) * 4, 1);
		if (L.shadow_map != -1)
		{
			Data.InterlockedAdd(STATS + STAT_LIGHT_SHADOWMAP * 4, 1);
			const uint near_mm = uint(clamp(shadow_near(L), 0.0, 1000.0) * 1000.0 + 0.5);
			Data.InterlockedMax(STATS + STAT_SHADOW_NEAR_MAX * 4, near_mm);
			Data.InterlockedMax(STATS + STAT_SHADOW_NEAR_MIN * 4, ~near_mm);   // complement: the smallest (0 = none)
		}
		if (((L.flags >> 9) & 7u) != 0)
			Data.InterlockedAdd(STATS + STAT_LIGHT_COOKIE * 4, 1);
		const uint mm = uint(clamp(L.radius, 0.0, 1000.0) * 1000.0 + 0.5);
		Data.InterlockedAdd(STATS + STAT_LIGHT_RADIUS * 4, mm);
		Data.InterlockedMax(STATS + STAT_LIGHT_RADIUS_MAX * 4, mm);
	}
}

// ---- capture of the lighting's inputs (Numpad -, v10) ----
// Reads every texture CS_DeferredLighting reads, through the game's own bindings (the same registers), plus the light
// buffer it wrote and its constants, into a dump buffer bound as Data (taken after the lighting: t4 / t5 show the
// add-on's rewritten ambient). Dump buffer: per slot uint4(width, height, downscale, kind) at slot * 16, c[0..84] from
// byte 1024, then the slots from byte 4096: DUMP_FLOAT_SLOTS of DUMP_W x DUMP_H half4 texels (8 bytes; the depth slot holds
// the raw float in the first 4), then the uint slots of uint4 texels (16 bytes). A larger texture is point-sampled down.

#define DUMP_W 1280
#define DUMP_H 720
#define DUMP_FLOAT_SLOTS 13                                 // t4 t5 t6 ray-albedo t10 ray-normal t16 t18 t19 ray-info t2-emission u0 t8
#define DUMP_SLOTS 16                                       // + t0 t1 t23 (uint)
#define DUMP_RAY_ALBEDO 3                                   // v14: the 16x16 placeholders t7, t11, t21 make way for the
#define DUMP_RAY_NORMAL 5                                   // "ray view": what a ray from the camera hits, through the scene
#define DUMP_RAY_INFO 9                                     // data a path tracer sees (material cache, triangle normals)

// The ray view at a render pixel: a camera ray to the depth buffer's surface. Albedo: the hit triangle's cached albedo
// (magenta = not seen yet, black = no hit); normal: its geometric normal turned toward the camera; info: (material
// known, (hit distance - depth distance) / depth distance, cached roughness, hit = 1)
float4 ray_view(uint2 px, uint slot)
{
	const float d = Depth[px];
	if (d >= 1.0)
		return 0.0;
	const float3 p = camera_relative_position(float2(px), d);
	const float dist = length(p);
	if (dist < 0.5)
		return 0.0;
	RayDesc ray;
	ray.Origin = c[9].xyz;
	ray.TMin = 0.3;
	ray.Direction = p / dist;
	ray.TMax = dist * 1.05 + 0.1;
	RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
	q.TraceRayInline(Scene, RAY_FLAG_NONE, MASK_DRAWN, ray);
	q.Proceed();
	if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
		return 0.0;
	const uint draw = SCENE_DRAW_COMMITTED(q), primitive = q.CommittedPrimitiveIndex();
	const uint m = material_slot(draw, primitive);
	if (slot == DUMP_RAY_ALBEDO)
		return m != 0 ? float4(unpack_material(m).rgb, 1.0) : float4(1.0, 0.0, 1.0, 1.0);
	if (slot == DUMP_RAY_NORMAL)
	{
		const float3 n = hit_normal(draw, primitive);
		return float4(dot(n, ray.Direction) > 0.0 ? -n : n, 1.0);
	}
	return float4(m != 0 ? 1.0 : 0.0, (q.CommittedRayT() - dist) / dist, m != 0 ? unpack_material(m).w : -1.0, 1.0);
}

uint dump_offset(uint slot)
{
	return 4096 + (slot < DUMP_FLOAT_SLOTS ? slot * DUMP_W * DUMP_H * 8 : DUMP_FLOAT_SLOTS * DUMP_W * DUMP_H * 8 + (slot - DUMP_FLOAT_SLOTS) * DUMP_W * DUMP_H * 16);
}

uint2 dump_size(uint slot)
{
	uint w = 0, h = 0;
	switch (slot)
	{
	case 0: InT4.GetDimensions(w, h); break;
	case 1: InT5.GetDimensions(w, h); break;
	case 2: InT6.GetDimensions(w, h); break;
	case DUMP_RAY_ALBEDO: case DUMP_RAY_NORMAL: case DUMP_RAY_INFO: return full_size();
	case 4: InT10.GetDimensions(w, h); break;
	case 6: InT16.GetDimensions(w, h); break;
	case 7: InT18.GetDimensions(w, h); break;
	case 8: InT19.GetDimensions(w, h); break;
	case 10: InT2.GetDimensions(w, h); break;                // v66: t2 decoded (the 16x16 t22 placeholder before)
	case 11: Light.GetDimensions(w, h); break;
	case 12: Depth.GetDimensions(w, h); break;
	case 13: GBuffer0.GetDimensions(w, h); break;
	case 14: GBuffer1.GetDimensions(w, h); break;
	default: InT23.GetDimensions(w, h); break;
	}
	return uint2(w, h);
}

[numthreads(8, 8, 1)]
void cs_dump(uint3 id : SV_DispatchThreadID)
{
	const uint slot = g_pass;
	const uint2 size = dump_size(slot);
	const uint scale = max(1u, max((size.x + DUMP_W - 1) / DUMP_W, (size.y + DUMP_H - 1) / DUMP_H));
	if (id.x == 0 && id.y == 0)
	{
		Data.Store4(slot * 16, uint4(size, scale, slot < DUMP_FLOAT_SLOTS ? 0u : 1u));
		if (slot == 0)
			for (uint i = 0; i < 85; ++i)
				Data.Store4(1024 + i * 16, asuint(c[i]));
	}
	if (id.x >= size.x / scale || id.y >= size.y / scale)
		return;
	const uint2 p = id.xy * scale;
	if (slot < DUMP_FLOAT_SLOTS)
	{
		float4 v = 0.0;
		switch (slot)
		{
		case 0: v = InT4[p]; break;
		case 1: v = InT5[p]; break;
		case 2: v = InT6[p]; break;
		case DUMP_RAY_ALBEDO: case DUMP_RAY_NORMAL: case DUMP_RAY_INFO: v = ray_view(p, slot); break;
		case 4: v = InT10[p]; break;
		case 6: v = InT16[p]; break;
		case 7: v = InT18[p]; break;
		case 8: v = InT19[p]; break;
		case 10: v = float4(game_emission(p), f16tof32(InT2[p].z & 0xFFFFu)); break;   // v66: the game's emission (raw), its f16
		case 11: v = Light[p]; break;
		default: v = float4(Depth[p], 0.0, 0.0, 0.0); break;
		}
		const uint at = dump_offset(slot) + (id.y * DUMP_W + id.x) * 8;
		if (slot == 12)
			Data.Store2(at, uint2(asuint(v.x), 0));
		else
			Data.Store2(at, uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16)));
	}
	else
	{
		uint4 u = 0;
		switch (slot)
		{
		case 13: u = GBuffer0[p]; break;
		case 14: u = GBuffer1[p]; break;
		default: u = InT23[p]; break;
		}
		Data.Store4(dump_offset(slot) + (id.y * DUMP_W + id.x) * 16, u);
	}
}

// The lights the tiles name, for the capture: the description of light index i (below 1024) at dump_offset(DUMP_SLOTS) +
// i * DUMP_LIGHT_BYTES (436 bytes as the game's struct), then 1 (named) and the number of tile entries naming it
#define DUMP_LIGHT_BYTES 448
void store_light(uint at, LightDesc L)
{
	Data.Store4(at + 0, uint4(asuint(L.cull_center), asuint(L.cull_radius)));
	Data.Store4(at + 16, uint4(asuint(L.cull_axis), asuint(L.cull_misc)));
	Data.Store4(at + 32, uint4(L.flags, asuint(L.color)));
	Data.Store4(at + 48, asuint(L.falloff));
	Data.Store4(at + 64, uint4(asuint(L.position), asuint(L.near_dist)));
	Data.Store4(at + 80, uint4(asuint(L.axis), L.cookie));
	Data.Store4(at + 96, uint4(asuint(L.cone), asuint(L.falloff_cubic), asuint(L.shadow_map)));
	Data.Store4(at + 112, uint4(asuint(L.depth_b), asuint(L.depth_a), asuint(L.depth_scale), L.shadow_misc));
	Data.Store4(at + 128, asuint(L.proj[0]));
	Data.Store4(at + 144, asuint(L.proj[1]));
	Data.Store4(at + 160, asuint(L.proj[2]));
	Data.Store4(at + 176, asuint(L.shadow_fade));
	Data.Store4(at + 192, uint4(asuint(L.cone2), asuint(L.min_roughness), asuint(L.spec_scale)));
	Data.Store4(at + 208, uint4(asuint(L.cull_misc2), asuint(L.unused212), asuint(L.near_fade), asuint(L.pcf_radius)));
	Data.Store4(at + 224, uint4(asuint(L.pcf_min), asuint(L.pcf_range), L.shadow_curve, L.unused236));
	Data.Store4(at + 240, asuint(L.cookie_a));
	Data.Store4(at + 256, asuint(L.cookie_b));
	[unroll]
	for (uint k = 0; k < 8; ++k)
		Data.Store4(at + 272 + k * 16, asuint(L.unused272[k]));
	Data.Store4(at + 400, uint4(asuint(L.box_x), asuint(L.box_y), asuint(L.box_soft), asuint(L.radius)));
	Data.Store4(at + 416, uint4(asuint(L.mask_axis), asuint(L.mask_hi)));
	Data.Store(at + 432, asuint(L.mask_lo));
}

// One thread per tile entry (tile * 16 + entry)
[numthreads(64, 1, 1)]
void cs_dump_lights(uint3 id : SV_DispatchThreadID)
{
	const uint tile = id.x >> 4, entry = id.x & 15u;
	if (tile >= light_tile_count() || entry >= min(LightCounts.Load(tile * 4), 16u))
		return;
	const uint index = LightIndex[tile * 16 + entry];
	if (index >= 1024)
		return;
	const uint at = dump_offset(DUMP_SLOTS) + index * DUMP_LIGHT_BYTES;
	store_light(at, Lights[index]);
	Data.Store(at + 436, 1u);
	Data.InterlockedAdd(at + 440, 1u);
}
