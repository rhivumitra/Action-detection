package com.example.blackboxinvaders.activities

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import com.example.blackboxinvaders.R

/**
 * Main menu screen with a start button that navigates to sensor capture.
 */
class MainMenuActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_menu)

        val btnStart = findViewById<Button>(R.id.btnStart)
        btnStart.setOnClickListener {
            // Navigate to sensor data capture screen
            val intent = Intent(this, SensorCaptureActivity::class.java)
            startActivity(intent)
        }
    }
}