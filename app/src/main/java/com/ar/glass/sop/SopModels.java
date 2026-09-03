package com.ar.glass.sop;

import com.google.gson.annotations.SerializedName;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class SopModels {
    private SopModels() {}

    public static final class User {
        public int id;
        public String username;
        @SerializedName("display_name") public String displayName;
        public String role;
    }

    public static final class LoginResponse {
        @SerializedName("access_token") public String accessToken;
        public User user;
    }

    public static final class Step {
        public String key;
        public String type;
        public String title;
        public String instruction;
        public boolean required = true;
        @SerializedName("require_evidence") public boolean requireEvidence = true;
        @SerializedName("require_human_confirmation") public boolean requireHumanConfirmation;
        public Map<String, Object> config;

        public String analyzer() {
            Object value = config == null ? null : config.get("analyzer");
            return value == null ? "manual" : String.valueOf(value);
        }
    }

    public static final class Assignment {
        public int id;
        @SerializedName("template_id") public int templateId;
        @SerializedName("assignee_id") public int assigneeId;
        @SerializedName("asset_code") public String assetCode;
        public String status;
        public String code;
        public int version;
        public String title;
        public List<Step> steps = new ArrayList<>();
    }

    public static final class Run {
        public String id;
        @SerializedName("assignment_id") public int assignmentId;
        public String status;
    }

    public static final class StepPayload {
        @SerializedName("idempotency_key") public String idempotencyKey;
        public String status;
        public Map<String, Object> value;
        public Double confidence;
        @SerializedName("requires_human_review") public boolean requiresHumanReview;
        @SerializedName("human_decision") public String humanDecision;
        @SerializedName("analyzer_version") public String analyzerVersion;
        @SerializedName("error_code") public String errorCode;
        @SerializedName("captured_at") public String capturedAt;
    }
}
