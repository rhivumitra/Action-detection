package com.example.blackboxinvaders.activities

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import com.example.blackboxinvaders.R

/**
 * Activity for data recording options or viewing collected data.
 * Currently shows a back button to return to MainMenuActivity.
 */
class DataRecordingActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_data_recording)

        val btnGoBack = findViewById<Button>(R.id.goBack)
        btnGoBack.setOnClickListener {
            // Return to main menu
            val intent = Intent(this, MainMenuActivity::class.java)
            startActivity(intent)
        }
    }
}
