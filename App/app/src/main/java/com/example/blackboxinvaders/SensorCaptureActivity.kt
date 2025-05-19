package com.example.blackboxinvaders.activities

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import com.example.blackboxinvaders.R
import com.example.blackboxinvaders.models.SensorData
import com.example.blackboxinvaders.utils.CsvWriter

/**
 * Activity to capture accelerometer sensor data live and display it.
 * Also supports saving collected data to CSV on demand.
 */
class SensorCaptureActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var accelerometer: Sensor

    private lateinit var tvX: TextView
    private lateinit var tvY: TextView
    private lateinit var tvZ: TextView

    private lateinit var btnStartRecording: Button
    private lateinit var btnStopRecording: Button

    // Buffer to hold recorded sensor data during session
    private val recordedData = mutableListOf<SensorData>()

    private var isRecording = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sensor_capture)

        // Initialize UI components
        tvX = findViewById(R.id.tvX)
        tvY = findViewById(R.id.tvY)
        tvZ = findViewById(R.id.tvZ)

        btnStartRecording = findViewById(R.id.btnStartRecording)
        btnStopRecording = findViewById(R.id.btnStopRecording)

        // Setup sensor manager and accelerometer
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        // Start recording on button click
        btnStartRecording.setOnClickListener {
            recordedData.clear()
            isRecording = true
        }

        // Stop recording and save to CSV on button click
        btnStopRecording.setOnClickListener {
            isRecording = false
            if (recordedData.isNotEmpty()) {
                CsvWriter.writeToCsv(this, recordedData)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Register listener to accelerometer sensor with normal delay
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL)
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    /**
     * Called every time new sensor data is available.
     * Update UI and save data if recording.
     */
    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            val x = event.values[0]
            val y = event.values[1]
            val z = event.values[2]
            val timestamp = System.currentTimeMillis()

            // Update UI with latest sensor values
            tvX.text = "X: %.2f".format(x)
            tvY.text = "Y: %.2f".format(y)
            tvZ.text = "Z: %.2f".format(z)

            // If recording, add sensor data to buffer
            if (isRecording) {
                recordedData.add(SensorData(x, y, z, timestamp))
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No operation needed here for this app
    }
}