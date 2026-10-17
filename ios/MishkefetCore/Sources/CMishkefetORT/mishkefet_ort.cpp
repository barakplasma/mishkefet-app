#include "mishkefet_ort.h"

#include <onnxruntime/coreml_provider_factory.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

namespace {

void writeError(char *err, size_t errLen, const char *msg) {
    if (err && errLen > 0) {
        std::snprintf(err, errLen, "%s", msg);
    }
}

Ort::Env &sharedEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mishkefet");
    return env;
}

}  // namespace

struct MishkefetOrt {
    Ort::Session session{nullptr};
    Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

extern "C" MishkefetOrt *mishkefet_ort_create(const char *path, MishkefetOrtOptions options, char *err,
                                              size_t errLen) {
    try {
        auto o = std::make_unique<MishkefetOrt>();
        Ort::SessionOptions so;
        if (!options.arena) so.DisableCpuMemArena();
        if (!options.memPattern) so.DisableMemPattern();
        if (!options.prepacking) so.AddConfigEntry("session.disable_prepacking", "1");
        if (options.intraOpThreads > 0) so.SetIntraOpNumThreads(options.intraOpThreads);
        so.SetInterOpNumThreads(1);
        so.SetExecutionMode(ORT_SEQUENTIAL);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (options.provider == 1) {
            std::unordered_map<std::string, std::string> p;
            p[kCoremlProviderOption_ModelFormat] = "MLProgram";
            p[kCoremlProviderOption_MLComputeUnits] = options.coreMLComputeUnits ? options.coreMLComputeUnits : "ALL";
            p[kCoremlProviderOption_RequireStaticInputShapes] = "0";
            p[kCoremlProviderOption_EnableOnSubgraphs] = "0";
            if (options.coreMLCacheDirectory && options.coreMLCacheDirectory[0]) {
                p[kCoremlProviderOption_ModelCacheDirectory] = options.coreMLCacheDirectory;
            }
            so.AppendExecutionProvider("CoreML", p);
        }
        o->session = Ort::Session(sharedEnv(), path, so);
        return o.release();
    } catch (const std::exception &e) {
        writeError(err, errLen, e.what());
        return nullptr;
    }
}

extern "C" void mishkefet_ort_destroy(MishkefetOrt *ort) { delete ort; }

extern "C" int32_t mishkefet_ort_run(MishkefetOrt *o, const float *image, int64_t width, float *out,
                                     int64_t outCapacity, int64_t *frames, int64_t *dim, char *err,
                                     size_t errLen) {
    try {
        int64_t shape[4] = {1, 1, 64, width};
        Ort::Value input = Ort::Value::CreateTensor<float>(o->cpu, const_cast<float *>(image),
                                                           (size_t)(64 * width), shape, 4);
        const char *inName = "image";
        const char *outName = "features";
        auto result = o->session.Run(Ort::RunOptions{nullptr}, &inName, &input, 1, &outName, 1);
        auto info = result[0].GetTensorTypeAndShapeInfo();
        auto outShape = info.GetShape();
        if (outShape.size() != 3) throw std::runtime_error("features: expected rank 3");
        const int64_t total = (int64_t)info.GetElementCount();
        if (total > outCapacity) throw std::runtime_error("features: output buffer too small");
        std::memcpy(out, result[0].GetTensorData<float>(), (size_t)total * sizeof(float));
        *frames = outShape[1];
        *dim = outShape[2];
        return 0;
    } catch (const std::exception &e) {
        writeError(err, errLen, e.what());
        return 1;
    }
}
