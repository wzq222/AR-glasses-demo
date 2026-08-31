import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.yolov8"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.yolov8"
        // minSdk 24 = Android 7.0，覆盖绝大多数设备，兼容性优先
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            // R8 混淆：缩小体积并剥离未用代码；keep 规则见 proguard-rules.pro
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // 本地验证用 debug 签名；正式分发前替换为 release keystore
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }
}

dependencies {
    // AndroidX 基础
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.recyclerview:recyclerview:1.3.2")

    // CameraX：相机预览 + 图像分析，1.3.x 稳定版
    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // ONNX Runtime：CPU + NNAPI（GPU 硬件加速，自动兜底 CPU，兼容性最优）
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    // PyTorch Mobile：直接加载 TorchScript 格式 .pt 模型（CPU 推理）
    implementation("org.pytorch:pytorch_android:2.1.0")

    // 生命周期
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // 测试
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}