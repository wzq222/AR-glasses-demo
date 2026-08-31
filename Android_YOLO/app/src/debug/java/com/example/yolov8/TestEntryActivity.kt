package com.example.yolov8

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * 仅 debug 构建包含的测试入口：
 * adb shell am start -n com.example.yolov8/.TestEntryActivity \
 *   --es target test --ez auto_run true
 *
 * target: "test" -> TestImageActivity（默认），"main" -> MainActivity
 * 其余 extras 原样透传（auto_run / start_virtual 等）。
 */
class TestEntryActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val target = intent.getStringExtra(EXTRA_TARGET) ?: TARGET_TEST
        val forward = Intent(
            this,
            if (target == TARGET_MAIN) MainActivity::class.java else TestImageActivity::class.java
        )
        forward.putExtras(intent)
        forward.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        startActivity(forward)
        finish()
    }

    companion object {
        const val EXTRA_TARGET = "target"
        const val TARGET_MAIN = "main"
        const val TARGET_TEST = "test"
    }
}
