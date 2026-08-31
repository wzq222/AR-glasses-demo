package com.example.yolov8.core.permissions

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.activity.result.ActivityResultLauncher
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import com.example.yolov8.R
import com.example.yolov8.core.log.AppLogger

/**
 * 权限统一处理：
 *  - 相机：运行时申请；拒绝后弹出说明；永久拒绝引导到系统设置
 *  - 存储（仅 API≤28 保存标注图需要）：WRITE_EXTERNAL_STORAGE
 *  - API29+ 保存使用 MediaStore 无需权限；图片选取统一走 SAF 无需权限
 */
object PermissionManager {

    private const val TAG = "Perm"

    fun hasCamera(context: android.content.Context): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    /** API≤28 保存图片需要旧存储权限 */
    fun needsLegacyWrite(): Boolean = Build.VERSION.SDK_INT <= 28

    fun hasLegacyWrite(context: android.content.Context): Boolean =
        !needsLegacyWrite() || ContextCompat.checkSelfPermission(
            context, Manifest.permission.WRITE_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED

    /**
     * 用 launcher 申请权限；被拒后展示 rationale 对话框，可选跳转设置。
     * @param rejectedPermanently 上一次拒绝且不再询问时为 true（由 Activity 判断 shouldShowRequestPermissionRationale）
     */
    fun requestWithRationale(
        activity: Activity,
        launcher: ActivityResultLauncher<String>,
        permission: String,
        rationaleTitle: String,
        rationaleText: String,
        onDenied: () -> Unit = {}
    ) {
        if (ContextCompat.checkSelfPermission(activity, permission) ==
            PackageManager.PERMISSION_GRANTED
        ) return

        val showRationale = activity.shouldShowRequestPermissionRationale(permission)
        if (showRationale) {
            AlertDialog.Builder(activity)
                .setTitle(rationaleTitle)
                .setMessage(rationaleText)
                .setPositiveButton(R.string.perm_go_settings) { _, _ ->
                    openAppSettings(activity)
                }
                .setNegativeButton(android.R.string.cancel) { _, _ -> onDenied() }
                .show()
        } else {
            AppLogger.i(TAG, "request $permission")
            launcher.launch(permission)
        }
    }

    fun openAppSettings(activity: Activity) {
        try {
            activity.startActivity(
                Intent(
                    Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    Uri.fromParts("package", activity.packageName, null)
                )
            )
        } catch (e: Exception) {
            AppLogger.w(TAG, "打开设置失败: ${e.message}")
        }
    }
}