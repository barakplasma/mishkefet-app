#ifndef MISHKEFET_ORT_H
#define MISHKEFET_ORT_H

// Thin C wrapper over the ONNX Runtime C++ API for the Mishkefet backbone:
// one float input "image" (1, 1, 64, W) and one float output "features" (1, T, 512).

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MishkefetOrt MishkefetOrt;

typedef struct {
    /// Intra-op threads (0 = ORT default).
    int32_t intraOpThreads;
    /// Execution provider: 0 = CPU; 1 = CoreML (MLProgram).
    int32_t provider;
    /// CoreML compute units: "ALL", "CPUAndNeuralEngine", "CPUAndGPU", "CPUOnly" (NULL = ALL).
    const char *_Nullable coreMLComputeUnits;
    /// Directory where CoreML caches the compiled model (NULL = no cache).
    const char *_Nullable coreMLCacheDirectory;
    /// Memory settings, each 0/1: CPU memory arena, weight pre-packing, memory-pattern planning.
    int32_t arena;
    int32_t prepacking;
    int32_t memPattern;
} MishkefetOrtOptions;

/// Loads the backbone. Returns NULL and writes a message into `err` on failure.
MishkefetOrt *_Nullable mishkefet_ort_create(const char *_Nonnull path, MishkefetOrtOptions options,
                                             char *_Nullable err, size_t errLen);

void mishkefet_ort_destroy(MishkefetOrt *_Nullable ort);

/// Runs the backbone on `image` laid out [64][width] float32 and copies the
/// features [frames][dim] into `out` (capacity `outCapacity` floats).
/// Returns 0 on success.
int32_t mishkefet_ort_run(MishkefetOrt *_Nonnull ort, const float *_Nonnull image, int64_t width,
                          float *_Nonnull out, int64_t outCapacity, int64_t *_Nonnull frames,
                          int64_t *_Nonnull dim, char *_Nullable err, size_t errLen);

#ifdef __cplusplus
}
#endif

#endif
