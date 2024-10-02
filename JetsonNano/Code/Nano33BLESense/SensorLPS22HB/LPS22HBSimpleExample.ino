/**
 * @file LPS22HBSimpleExample.ino
 *
 * @brief Example code for the sensor LPS22HB on the Arduino Nano 33 BLE Sense
 *
 * @author Elmar Wings
 *
 * @version 1.0
 */

#include <LPS22HB.h>

/**
 * @brief LPS22HB sensor object
 *
 * @details This object represents the LPS22HB sensor and provides methods for reading temperature and pressure values.
 */
LPS22HB lps22hb;

/**
 * @brief Setup function
 *
 * @details This function is called once at the beginning of the program and is used to initialize the sensor and serial communication.
 */
void setup() {
  // Initialize serial communication
  Serial.begin(9600);
 
  //Initialize the LPS22HB sensor
  lps22hb.begin();
}

/**
 * @brief Loop function
 *
 * @details This function is called repeatedly after the setup function and is used to read temperature and pressure values from the sensor.
 */
void loop() {
  // Read temperature value from the sensor
  int16_t temp = lps22hb.readTemperature();
 
  // Read pressure value from the sensor
  int32_t pressure = lps22hb.readPressure();
 
  // Print temperature and pressure values to the serial console
  Serial.print("Temperature: ");
  Serial.print(temp);
  Serial.println(" C");
  Serial.print("Pressure: ");
  Serial.print(pressure);
  Serial.println(" mbar");
 
  // Wait for 1 second before taking the next reading
  delay(1000);
}