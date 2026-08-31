package com.example.yolov8.core.log

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.yolov8.R
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton
import android.content.Intent
import androidx.appcompat.app.AlertDialog

/**
 * 应用内日志查看器（仅 debug 构建入口可见）。
 */
class LogActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_log)
        setSupportActionBar(findViewById(R.id.toolbar))
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = getString(R.string.log_title)

        val text = findViewById<TextView>(R.id.logText)
        text.text = AppLogger.recentLines().joinToString("\n")

        findViewById<ExtendedFloatingActionButton>(R.id.clearFab).setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle(R.string.log_clear_title)
                .setPositiveButton(android.R.string.ok) { _, _ ->
                    AppLogger.clearRecent()
                    text.text = ""
                }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
        }
        findViewById<ExtendedFloatingActionButton>(R.id.shareFab).setOnClickListener {
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_TEXT, AppLogger.recentLines().joinToString("\n"))
            }
            startActivity(Intent.createChooser(intent, getString(R.string.log_share)))
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }
}