package com.example.blackboxinvaders.models

/**
 * Data class representing a single accelerometer reading with timestamp.
 */
data class SensorData(
    val x: Float,
    val y: Float,
    val z: Float,
    val timestamp: Long
) {
    /**
     * Format the data as a CSV row.
     */
    fun toCSVRow(): String {
        return "$x,$y,$z,$timestamp"
    }
}
