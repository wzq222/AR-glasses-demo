package com.ar.glass.sop;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;

import com.ar.glass.BuildConfig;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public final class SopApiClient {
    public interface Result<T> {
        void onSuccess(T value);
        void onError(String message);
    }

    private static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");
    private static final String PREFS = "crrc_sop_session";
    private static final String KEY_TOKEN = "access_token";

    private final OkHttpClient http = new OkHttpClient.Builder()
            .connectTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
            .readTimeout(45, java.util.concurrent.TimeUnit.SECONDS)
            .writeTimeout(45, java.util.concurrent.TimeUnit.SECONDS)
            .build();
    private final Gson gson = new Gson();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final SharedPreferences preferences;
    private final String baseUrl;

    public SopApiClient(Context context) {
        preferences = context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        baseUrl = BuildConfig.SOP_BASE_URL.replaceAll("/+$", "");
    }

    public boolean hasSession() {
        return !token().isEmpty();
    }

    public void clearSession() {
        preferences.edit().remove(KEY_TOKEN).apply();
    }

    public void login(String username, String password, Result<SopModels.LoginResponse> result) {
        Map<String, String> payload = new java.util.HashMap<>();
        payload.put("username", username);
        payload.put("password", password);
        execute(new Request.Builder().url(url("/api/v1/auth/login"))
                        .post(RequestBody.create(JSON, gson.toJson(payload))).build(),
                SopModels.LoginResponse.class,
                new Result<SopModels.LoginResponse>() {
                    @Override public void onSuccess(SopModels.LoginResponse value) {
                        preferences.edit().putString(KEY_TOKEN, value.accessToken).apply();
                        result.onSuccess(value);
                    }
                    @Override public void onError(String message) { result.onError(message); }
                });
    }

    public void me(Result<SopModels.User> result) {
        execute(authorized("/api/v1/users/me").get().build(), SopModels.User.class, result);
    }

    public void assignments(Result<List<SopModels.Assignment>> result) {
        Type type = new TypeToken<List<SopModels.Assignment>>() {}.getType();
        execute(authorized("/api/v1/assignments").get().build(), type, result);
    }

    public void startRun(int assignmentId, Result<SopModels.Run> result) {
        Map<String, Object> payload = new java.util.HashMap<>();
        payload.put("assignment_id", assignmentId);
        payload.put("device", Collections.singletonMap("source", "ANDROID_PHONE"));
        execute(authorized("/api/v1/runs").post(RequestBody.create(JSON, gson.toJson(payload))).build(),
                SopModels.Run.class, result);
    }

    public void saveStep(String runId, String stepKey, SopModels.StepPayload payload,
                         Result<JsonObject> result) {
        execute(authorized("/api/v1/runs/" + runId + "/steps/" + stepKey)
                        .put(RequestBody.create(JSON, gson.toJson(payload))).build(),
                JsonObject.class, result);
    }

    public void uploadEvidence(String runId, String stepKey, File file, Result<JsonObject> result) {
        MediaType mediaType = MediaType.parse(SopEvidenceFiles.mediaType(file));
        RequestBody body = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("file", file.getName(), RequestBody.create(mediaType, file)).build();
        execute(authorized("/api/v1/runs/" + runId + "/steps/" + stepKey + "/evidence")
                .post(body).build(), JsonObject.class, result);
    }

    public void submitRun(String runId, Result<JsonObject> result) {
        execute(authorized("/api/v1/runs/" + runId + "/submit")
                .post(RequestBody.create(JSON, "{}" )).build(), JsonObject.class, result);
    }

    private Request.Builder authorized(String path) {
        return new Request.Builder().url(url(path)).header("Authorization", "Bearer " + token());
    }

    private String token() {
        return preferences.getString(KEY_TOKEN, "");
    }

    private String url(String path) {
        return baseUrl + path;
    }

    private <T> void execute(Request request, Class<T> type, Result<T> result) {
        execute(request, (Type) type, result);
    }

    private <T> void execute(Request request, Type type, Result<T> result) {
        http.newCall(request).enqueue(new Callback() {
            @Override public void onFailure(Call call, IOException error) {
                postError(result, "网络连接失败，请检查手机网络");
            }

            @Override public void onResponse(Call call, Response response) {
                try (Response closeable = response) {
                    String body = response.body() == null ? "" : response.body().string();
                    if (!response.isSuccessful()) {
                        if (response.code() == 401) clearSession();
                        postError(result, errorMessage(response.code(), body));
                        return;
                    }
                    T value = gson.fromJson(body, type);
                    main.post(() -> result.onSuccess(value));
                } catch (Exception error) {
                    postError(result, "服务响应解析失败");
                }
            }
        });
    }

    private <T> void postError(Result<T> result, String message) {
        main.post(() -> result.onError(message));
    }

    private String errorMessage(int code, String body) {
        try {
            JsonElement detail = gson.fromJson(body, JsonObject.class).get("detail");
            if (detail != null && detail.isJsonPrimitive()) {
                String message = detail.getAsString();
                if ("invalid credentials".equals(message)) return "用户名或密码错误";
                if ("insufficient role".equals(message)) return "当前账号没有此操作权限";
                return message;
            }
        } catch (Exception ignored) {
        }
        if (code == 401) return "登录已失效，请重新登录";
        if (code == 403) return "当前账号没有执行权限";
        if (code == 409) return "步骤或证据尚未完整，请检查后重试";
        return "服务请求失败（" + code + "）";
    }
}
