#include <android/asset_manager_jni.h>
#include <jni.h>

#include <cstring>
#include <memory>
#include <string>

#include <net.h>

namespace {

constexpr int kInputSize = 640;
constexpr int kOutputChannels = 6;
constexpr int kOutputCandidates = 34000;
constexpr size_t kInputElements =
        static_cast<size_t>(3) * kInputSize * kInputSize;
constexpr size_t kOutputElements =
        static_cast<size_t>(kOutputChannels) * kOutputCandidates;

class UtfChars {
public:
    UtfChars(JNIEnv* environment, jstring value)
            : environment_(environment), value_(value), chars_(nullptr) {
        if (value_ != nullptr) {
            chars_ = environment_->GetStringUTFChars(value_, nullptr);
        }
    }

    ~UtfChars() {
        if (chars_ != nullptr) {
            environment_->ReleaseStringUTFChars(value_, chars_);
        }
    }

    const char* get() const { return chars_; }

private:
    JNIEnv* environment_;
    jstring value_;
    const char* chars_;
};

struct DetectorHandle {
    ncnn::Net network;
    std::string input_name;
    std::string output_name;
    int threads;
};

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_ar_glass_vision_realtime_NcnnFastenerDetector_nativeCreate(
        JNIEnv* environment,
        jclass,
        jobject asset_manager,
        jstring param_asset_name,
        jstring bin_asset_name,
        jstring input_blob_name,
        jstring output_blob_name,
        jint threads) {
    AAssetManager* assets = AAssetManager_fromJava(environment, asset_manager);
    UtfChars param_name(environment, param_asset_name);
    UtfChars bin_name(environment, bin_asset_name);
    UtfChars input_name(environment, input_blob_name);
    UtfChars output_name(environment, output_blob_name);
    if (assets == nullptr || param_name.get() == nullptr || bin_name.get() == nullptr
            || input_name.get() == nullptr || output_name.get() == nullptr) {
        return 0;
    }

    std::unique_ptr<DetectorHandle> handle(new DetectorHandle());
    handle->threads = threads > 0 ? threads : 1;
    handle->input_name = input_name.get();
    handle->output_name = output_name.get();
    handle->network.opt.num_threads = handle->threads;
    handle->network.opt.use_fp16_packed = false;
    handle->network.opt.use_fp16_storage = false;
    handle->network.opt.use_fp16_arithmetic = false;
    handle->network.opt.use_vulkan_compute = false;
#if CRRC_NCNN_EXACT_MATH
    handle->network.opt.use_packing_layout = false;
    handle->network.opt.use_winograd_convolution = false;
    handle->network.opt.use_sgemm_convolution = false;
    handle->network.opt.flush_denormals = 0;
#endif
    if (handle->network.load_param(assets, param_name.get()) != 0
            || handle->network.load_model(assets, bin_name.get()) != 0) {
        return 0;
    }
    return reinterpret_cast<jlong>(handle.release());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_ar_glass_vision_realtime_NcnnFastenerDetector_nativeInfer(
        JNIEnv* environment,
        jclass,
        jlong native_handle,
        jobject input_buffer,
        jobject output_buffer) {
    auto* handle = reinterpret_cast<DetectorHandle*>(native_handle);
    auto* input = static_cast<float*>(
            environment->GetDirectBufferAddress(input_buffer));
    auto* output = static_cast<float*>(
            environment->GetDirectBufferAddress(output_buffer));
    const jlong input_capacity = environment->GetDirectBufferCapacity(input_buffer);
    const jlong output_capacity = environment->GetDirectBufferCapacity(output_buffer);
    if (handle == nullptr || input == nullptr || output == nullptr
            || input_capacity < static_cast<jlong>(kInputElements)
            || output_capacity < static_cast<jlong>(kOutputElements)) {
        return JNI_FALSE;
    }

    ncnn::Mat input_tensor(kInputSize, kInputSize, 3, input);
    ncnn::Extractor extractor = handle->network.create_extractor();
    if (extractor.input(handle->input_name.c_str(), input_tensor) != 0) {
        return JNI_FALSE;
    }
    ncnn::Mat result;
    if (extractor.extract(handle->output_name.c_str(), result) != 0
            || result.dims != 2
            || result.w != kOutputCandidates
            || result.h != kOutputChannels
            || result.elempack != 1
            || result.elemsize != sizeof(float)) {
        return JNI_FALSE;
    }
    std::memcpy(output, result.data, kOutputElements * sizeof(float));
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_ar_glass_vision_realtime_NcnnFastenerDetector_nativeDestroy(
        JNIEnv*, jclass, jlong native_handle) {
    delete reinterpret_cast<DetectorHandle*>(native_handle);
}
