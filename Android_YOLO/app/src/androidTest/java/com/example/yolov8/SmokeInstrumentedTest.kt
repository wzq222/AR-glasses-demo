package com.example.yolov8

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.core.app.ActivityScenario
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 冒烟测试（需模拟器/真机）：
 *  - 包名与应用上下文正确
 *  - MainActivity / TestImageActivity 可启动不崩溃
 */
@RunWith(AndroidJUnit4::class)
class SmokeInstrumentedTest {

    @Test
    fun packageNameCorrect() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.example.yolov8", ctx.packageName)
    }

    @Test
    fun mainActivityLaunches() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                assertNotNull(activity.window)
            }
        }
    }

    @Test
    fun testImageActivityLaunches() {
        ActivityScenario.launch(TestImageActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                assertNotNull(activity.window)
            }
        }
    }
}