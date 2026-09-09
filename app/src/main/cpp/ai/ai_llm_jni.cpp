// llama.cpp JNI 封装：模型加载/流式生成/上下文重置/卸载
// GPU：llama.cpp 编译期启用 Vulkan，运行时 n_gpu_layers 决定分层层数，
//      显存放不下时 llama.cpp 自动把剩余层回退到 CPU。
#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <chrono>
#include <cstring>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define LOG_TAG "AiLlm"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct LlmSession {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_vocab const *vocab = nullptr;
    llama_sampler *smpl = nullptr;
    int n_ctx = 0;
    mtmd_context *mctx = nullptr; // 视觉编码器（加载 mmproj 后可用）
};

// 列出 ggml 全部后端设备（GPU/Vulkan 是否在列一目了然，附显存）
void log_ggml_devices() {
    size_t n_reg = ggml_backend_reg_count();
    for (size_t r = 0; r < n_reg; r++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(r);
        size_t n_dev = ggml_backend_reg_dev_count(reg);
        for (size_t i = 0; i < n_dev; i++) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            ALOGI("backend[%zu/%zu] %s | %s | mem %.2f/%.2f GB", r, i,
                  ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                  free_b / 1e9, total_b / 1e9);
        }
    }
}

// 全量转发 ggml/llama 日志到 logcat（设备枚举、Vulkan 初始化失败原因等）。
// 必须在任何后端初始化前安装，否则丢失关键诊断信息；纯进度点不转发。
void forward_log(ggml_log_level level, const char *text) {
    if (!text || !*text) return;
    bool only_progress = true;
    for (const char *p = text; *p; p++) {
        if (*p != '.' && *p != '\n' && *p != '\r' && *p != ' ') {
            only_progress = false;
            break;
        }
    }
    if (only_progress) return;
    std::string t(text);
    if (t.find("create_tensor: loading tensor") != std::string::npos) return;
    for (auto &c : t) if (c == '\n' || c == '\r') c = '|';
    if (level >= GGML_LOG_LEVEL_WARN) ALOGE("%s", t.c_str());
    else ALOGI("%s", t.c_str());
}

void install_log_hooks() {
    static bool installed = false;
    if (installed) return;
    installed = true;
    ggml_log_set([](ggml_log_level level, const char *text, void *) {
        forward_log(level, text);
    }, nullptr);
    llama_log_set([](ggml_log_level level, const char *text, void *) {
        forward_log(level, text);
    }, nullptr);
}

// GPU 判定必须包含 IGPU：ARM UMA 设备（Mali/Immortalis 等）注册为集成 GPU 类型，
// 只查 GPU 类型会把"正在用 Vulkan 跑"误报成 CPU
bool has_accel_dev() {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr
            || ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr;
}

jstring to_jstring_env(JNIEnv *env, const std::string &s) {
    return env->NewStringUTF(s.c_str());
}

std::string token_to_utf8(const llama_vocab *vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) return "";
    return std::string(buf, (size_t) n);
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_ar_glass_ai_LlmEngine_nativeCreateSession(
        JNIEnv *env, jobject /*thiz*/, jstring model_path,
        jint n_ctx, jint n_gpu_layers, jint n_threads) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);

    install_log_hooks();
    llama_backend_init();
    log_ggml_devices();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;   // 99 = 全部放 GPU（Vulkan 自动分层回退）
    ALOGI("loading model (n_gpu_layers=%d, threads=%d)...", n_gpu_layers, n_threads);
    auto t0 = std::chrono::steady_clock::now();

    llama_model *model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(model_path, path);
    if (!model) {
        ALOGE("model load failed: %s", path);
        llama_backend_free();
        return 0;
    }
    ALOGI("model loaded in %.1fs",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ALOGE("context init failed");
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    auto *session = new LlmSession{model, ctx, llama_model_get_vocab(model), nullptr,
                                   n_ctx};
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    ALOGI("session created, ctx=%d, accel_dev=%s (%s)", n_ctx, gpu_dev ? "yes" : "no",
          gpu_dev ? ggml_backend_dev_name(gpu_dev) : "cpu-only");
    return reinterpret_cast<jlong>(session);
}

JNIEXPORT void JNICALL
Java_com_ar_glass_ai_LlmEngine_nativeFreeSession(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    auto *s = reinterpret_cast<LlmSession *>(handle);
    if (!s) return;
    if (s->mctx) mtmd_free(s->mctx); // 依赖 model/ctx，需先于其释放
    if (s->smpl) llama_sampler_free(s->smpl);
    if (s->ctx) llama_free(s->ctx);
    if (s->model) llama_model_free(s->model);
    delete s;
    llama_backend_free();
}

// 预填完成后从当前上下文继续解码生成（tokens 为最近一步输入，首轮可为空向量）
std::string run_decode(LlmSession *s, std::vector<llama_token> tokens,
                       jint max_tokens, int32_t n_prompt) {
    std::string output;
    auto t_gen = std::chrono::steady_clock::now();
    int n_gen = 0;
    const llama_vocab *vocab = s->vocab;
    for (int i = 0; i < max_tokens; i++) {
        if (!tokens.empty()) {
            llama_batch batch = llama_batch_get_one(
                    tokens.data(), (int32_t) tokens.size());
            if (llama_decode(s->ctx, batch)) {
                ALOGE("decode failed");
                break;
            }
        }
        llama_token new_token = llama_sampler_sample(s->smpl, s->ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;
        output += token_to_utf8(vocab, new_token);
        tokens = {new_token};
        n_gen++;
        if ((int32_t) s->n_ctx > 0 &&
            (int) llama_memory_seq_pos_max(llama_get_memory(s->ctx), 0) >=
                    s->n_ctx - 4) {
            break; // 上下文将满，停止
        }
    }
    double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_gen).count();
    ALOGI("generated %d tokens in %.1fs (%.2f tok/s), prompt=%d tok",
          n_gen, secs, secs > 0 ? n_gen / secs : 0.0, n_prompt);
    return output;
}

void fresh_sampler(LlmSession *s, jfloat temperature) {
    if (s->smpl) llama_sampler_free(s->smpl);
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    s->smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(s->smpl, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(s->smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(s->smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

// 单轮流式生成：prompt 已由 Java 层套好模板。每次调用独立采样链，KV 在开头清空。
JNIEXPORT jstring JNICALL
Java_com_ar_glass_ai_LlmEngine_nativeGenerate(
        JNIEnv *env, jobject /*thiz*/, jlong handle, jstring jprompt,
        jint max_tokens, jfloat temperature) {
    auto *s = reinterpret_cast<LlmSession *>(handle);
    if (!s || !s->ctx) return env->NewStringUTF("");

    const char *prompt = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt_utf8(prompt);
    env->ReleaseStringUTFChars(jprompt, prompt);

    fresh_sampler(s, temperature);
    llama_memory_clear(llama_get_memory(s->ctx), true);

    const llama_vocab *vocab = s->vocab;
    const bool is_special = true; // 模板含 <|im_start|> 等特殊 token
    int n_prompt = -llama_tokenize(vocab, prompt_utf8.c_str(),
                                   (int32_t) prompt_utf8.size(), nullptr, 0,
                                   is_special, true);
    if (n_prompt <= 0) return env->NewStringUTF("");
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt_utf8.c_str(), (int32_t) prompt_utf8.size(),
                       tokens.data(), n_prompt, is_special, true) < 0) {
        return env->NewStringUTF("");
    }
    std::string output = run_decode(s, tokens, max_tokens, n_prompt);
    return env->NewStringUTF(output.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_ar_glass_ai_LlmEngine_nativeIsGpuActive(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return has_accel_dev();
}

// 加载 mmproj 视觉投影器，绑定到当前会话的 llama_model（多模态输入前提）
JNIEXPORT jboolean JNICALL
Java_com_ar_glass_ai_LlmEngine_nativeInitVision(
        JNIEnv *env, jobject /*thiz*/, jlong handle, jstring jmmproj) {
    auto *s = reinterpret_cast<LlmSession *>(handle);
    if (!s || !s->model) return JNI_FALSE;
    if (s->mctx) return JNI_TRUE; // 已加载
    const char *mmproj = env->GetStringUTFChars(jmmproj, nullptr);
    mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu = true;   // 视觉塔同样优先 Vulkan
    p.print_timings = true;
    p.n_threads = 4;
    s->mctx = mtmd_init_from_file(mmproj, s->model, p);
    env->ReleaseStringUTFChars(jmmproj, mmproj);
    ALOGI("vision init %s", s->mctx ? "ok" : "FAILED");
    return s->mctx ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_ar_glass_ai_LlmEngine_nativeHasVision(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    auto *s = reinterpret_cast<LlmSession *>(handle);
    return (s && s->mctx) ? JNI_TRUE : JNI_FALSE;
}

// 多模态生成：prompt 由 Java 层套模板并含媒体标记；pixels 为 ARGB_8888 位图
JNIEXPORT jstring JNICALL
Java_com_ar_glass_ai_LlmEngine_nativeGenerateWithImage(
        JNIEnv *env, jobject /*thiz*/, jlong handle, jstring jprompt,
        jintArray jpixels, jint width, jint height,
        jint max_tokens, jfloat temperature) {
    auto *s = reinterpret_cast<LlmSession *>(handle);
    if (!s || !s->ctx || !s->mctx) return env->NewStringUTF("");

    const char *prompt = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt_utf8(prompt);
    env->ReleaseStringUTFChars(jprompt, prompt);

    jsize n_px = env->GetArrayLength(jpixels);
    jint *px = env->GetIntArrayElements(jpixels, nullptr);
    std::vector<unsigned char> rgb((size_t) n_px * 3);
    for (jsize i = 0; i < n_px; i++) {
        uint32_t c = (uint32_t) px[i];
        rgb[i * 3 + 0] = (unsigned char) ((c >> 16) & 0xFF); // R
        rgb[i * 3 + 1] = (unsigned char) ((c >> 8) & 0xFF);  // G
        rgb[i * 3 + 2] = (unsigned char) (c & 0xFF);         // B
    }
    env->ReleaseIntArrayElements(jpixels, px, JNI_ABORT);

    mtmd_bitmap *bitmap = mtmd_bitmap_init((uint32_t) width, (uint32_t) height,
                                           rgb.data());
    if (!bitmap) return env->NewStringUTF("");

    mtmd_input_text text{prompt_utf8.c_str(), prompt_utf8.size(),
                         /*add_special=*/false, /*parse_special=*/true};
    const mtmd_bitmap *bitmaps[1] = {bitmap};
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    int32_t res = mtmd_tokenize(s->mctx, chunks, &text, bitmaps, 1);
    mtmd_bitmap_free(bitmap);
    if (res != 0) {
        ALOGE("mtmd_tokenize failed: %d", res);
        mtmd_input_chunks_free(chunks);
        return env->NewStringUTF("");
    }

    fresh_sampler(s, temperature);
    llama_memory_clear(llama_get_memory(s->ctx), true);
    auto t0 = std::chrono::steady_clock::now();
    llama_pos n_past = 0;
    res = mtmd_helper_eval_chunks(s->mctx, s->ctx, chunks,
                                  /*n_past=*/0, /*seq_id=*/0,
                                  /*n_batch=*/256, /*logits_last=*/true,
                                  &n_past);
    double prefill = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    mtmd_input_chunks_free(chunks);
    if (res != 0) {
        ALOGE("vision prefill failed: %d", res);
        return env->NewStringUTF("");
    }
    ALOGI("vision prefill done, n_past=%d in %.1fs", n_past, prefill);
    std::string output = run_decode(s, {}, max_tokens, (int32_t) n_past);
    return env->NewStringUTF(output.c_str());
}

} // extern "C"
