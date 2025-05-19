package com.example.blackboxinvaders

import org.junit.Test

import org.junit.Assert.*

/**
 * Unit test for the SensorData data class. 
 */
class SensorDataTest {
    @Test
    fun testToCSVRow() {
        val sensorData = SensorData(1.23f, 4.56f, 2.34f, 12312312)
        val expected = "1.23, 4.56, 2.34, 12312312"
        asserEquals(expected, sensorData.toCSVRow())
    }
}