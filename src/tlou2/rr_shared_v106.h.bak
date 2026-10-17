// TLOU RT - Ray Reconstruction in a helper process (v61; protocol v2: v78). DLSS Ray Reconstruction cannot run inside the
// game: created next to RenoDX's DLSS Neural Rendering (nvngx_dlssnr.dll) in the same process it hangs both (v59, v60). The
// helper (src\rr_helper.cpp, tlou-rt-rr.exe in the game folder's tlou-rt-rr subfolder - not next to ReShade's dxgi.dll, which
// would load ReShade and every add-on into it) runs RR on the same GPU in a process of its own. This header is the protocol
// both sides share: a named file mapping (rr_shared), a frame event, two shared fences, TLOU_RR_SLOTS sets of shared
// textures and (v78) a shared buffer of the slots' frame constants, all named after the game's process id.
//
// Per frame M (v78): the add-on's lighting list writes slot M % TLOU_RR_SLOTS - RR's inputs, and the frame's constants, which
// the add-on's GPU derives from the game's matrices (rr_gpu_frame: the jitter, a reset at a cut) - once the helper is done
// with the slot (the output fence at M - TLOU_RR_SLOTS or beyond). After that list is submitted the add-on signals the input
// fence to M on the game's queue, stores the frame record and publishes M. The helper takes every published frame in order:
// its queue waits on the input fence for M and copies the slot's constants into its own readback buffer, its CPU reads them
// and records RR (DLAA: the render size in and out) into the slot's output, and its queue signals the output fence to M -
// also when the evaluation failed. The game's queue waits on the output fence for M before the NEXT frame's lighting list,
// whose composite reprojects RR's light of M onto that frame; the add-on ends such a wait from the CPU (signals the output
// fence itself) when the helper is gone or late, so the game never hangs on the helper.
#pragma once
#include <windows.h>
#include <dxgiformat.h>
#include <cstdint>
#include <cwchar>

#define TLOU_RR_MAGIC 0x31525254u                           // "TRR1"
#define TLOU_RR_VERSION 2u                                  // v78: the GPU's frame constants, every frame evaluated in order
#define TLOU_RR_SLOTS 3
#define TLOU_RR_CONSTANT_BYTES 256                          // one slot's rr_gpu_frame in the shared constants buffer

enum rr_kind
{
	rr_color,                   // noisy lighting (HDR)
	rr_diffuse_albedo,
	rr_specular_albedo,
	rr_normals,                 // world-space normal, roughness in w (RR's packed roughness)
	rr_depth,                   // the hardware depth value (0 near, 1 far)
	rr_motion,                  // pixels to where the surface was last frame, the jitter included (MVJittered)
	rr_spec_hit,                // specular hit distance (world units; v78: R32, not passed to RR yet)
	rr_output,                  // RR's result (UAV)
	rr_kinds
};
static const DXGI_FORMAT k_rr_formats[rr_kinds] = {
	DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT,
	DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT };
static const wchar_t *const k_rr_kind_names[rr_kinds] = { L"color", L"diffuse", L"specular", L"normals", L"depth", L"motion", L"spechit", L"output" };

enum rr_state : LONG
{
	rr_state_starting = 0,
	rr_state_device = 1,        // D3D12 device on the game's adapter
	rr_state_ngx = 2,           // NGX up, RR available
	rr_state_ready = 3,         // RR's feature created for helper_generation: evaluating
	rr_state_failed = -1,
};

// The record the add-on's CPU stores with each published frame
struct rr_frame
{
	int64_t value;              // the frame's value on the fences
	float jitter_x, jitter_y;   // (v61-v77; v78: rr_gpu_frame's)
	float mv_scale_x, mv_scale_y;
	int32_t reset;              // 1: RR's history is discarded (v78: the first frame of a generation)
	float pre_exposure;
	float world_to_view[16];    // row-major
	float view_to_clip[16];
};

// v78: a slot's frame constants, written by the add-on's GPU with the slot's inputs (re9.hlsl rr_store_constants), read by
// the helper through its readback buffer before it records RR
struct rr_gpu_frame
{
	float jitter_x, jitter_y;   // the projection's jitter in render pixels (x right, y down)
	float mv_scale_x, mv_scale_y;
	uint32_t reset;             // 1: a cut (the camera jumped or turned) or the first frames
	float pre_exposure;
	uint32_t frame;             // the add-on's frame count (diagnostics)
	uint32_t pad;
	float world_to_view[16];    // row-major; all 0: not given (the helper passes identities)
	float view_to_clip[16];
};
static_assert(sizeof(rr_gpu_frame) <= TLOU_RR_CONSTANT_BYTES, "one slot's constants");

struct rr_shared
{
	uint32_t magic, version;
	LUID adapter;               // add-on: the game's adapter
	uint32_t width, height;     // add-on: the render size of generation
	volatile LONG generation;   // add-on: the shared textures' generation (their names carry it; 0 = none yet)
	volatile LONG quit;         // add-on: 1 = the helper exits
	volatile LONG64 published;  // add-on: the newest frame value - its inputs are complete when the input fence reaches it
	rr_frame frames[TLOU_RR_SLOTS];
	volatile LONG helper_state; // helper: rr_state
	volatile LONG helper_result; // helper: the last NGX result
	volatile LONG helper_generation; // helper: the generation its feature was created for
	volatile LONG helper_pad;
	volatile LONG64 evaluated;  // helper: evaluations submitted
	volatile LONG64 submitted;  // helper: the last frame value submitted (the output fence reaches it when done)
	char message[256];          // helper: its status in words
};

inline void rr_name(wchar_t *out, size_t count, DWORD pid, const wchar_t *what)
{
	std::swprintf(out, count, L"Local\\tlou-rt-rr-%lu-%ls", static_cast<unsigned long>(pid), what);
}
inline void rr_texture_name(wchar_t *out, size_t count, DWORD pid, LONG generation, int slot, int kind)
{
	std::swprintf(out, count, L"Local\\tlou-rt-rr-%lu-g%ld-s%d-%ls", static_cast<unsigned long>(pid), static_cast<long>(generation), slot, k_rr_kind_names[kind]);
}
// v78: the slots' frame constants (TLOU_RR_SLOTS x TLOU_RR_CONSTANT_BYTES), one shared buffer per generation
inline void rr_constants_name(wchar_t *out, size_t count, DWORD pid, LONG generation)
{
	std::swprintf(out, count, L"Local\\tlou-rt-rr-%lu-g%ld-constants", static_cast<unsigned long>(pid), static_cast<long>(generation));
}
