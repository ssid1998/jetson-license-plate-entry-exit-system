/**
 * @file LPS22HBCalibration.ino
 *
 * @brief Calibration code for the sensor LPS22HB on the Arduino Nano 33 BLE Sense
 *
 * @details This code reads the temperature and pressure values from the LPS22HB sensor, stores them in the sensor's memory, and prints them to the serial console. The storeCalibration function is used to store the calibration values in the sensor's memory.
*
* Note that this is just an example code and you may need to modify it to suit your specific needs. Additionally, you will need to make sure that the LPS22HB sensor is properly connected to the Arduino Nano 33 BLE Sense and that the I2C interface is enabled.
*
*
 * @author Elmar Wings
 *
 * @version 1.0
 */

#include <Wire.h>
#include <LPS22HB.h>

/**
 * @brief LPS22HB sensor object
 */
LPS22HB lps22hb;

/**
 * @brief Setup function
 */
void setup() {
  Serial.begin(9600);
  Wire.begin();
  lps22hb.begin();
}

/**
 * @brief Loop function
 */
void loop() {
  // Read the temperature and pressure values from the sensor
  int16_t temp = lps22hb.readTemperature();
  int32_t pressure = lps22hb.readPressure();

  // Print the temperature and pressure values to the serial console
  Serial.print("Temperature: ");
  Serial.print(temp);
  Serial.println(" C");
  Serial.print("Pressure: ");
  Serial.print(pressure);
  Serial.println(" mbar");

  // Store the calibration values in the sensor's memory
  lps22hb.storeCalibration(temp, pressure);

  // Wait for 1 second before taking the next reading
  delay(1000);
}

/**
 * @brief Store calibration values in the sensor's memory
 * @param temp Temperature value
 * @param pressure Pressure value
 */
void storeCalibration(int16_t temp, int32_t pressure) {
  // Store the calibration values in the sensor's memory
  Wire.beginTransmission(0x5C); // LPS22HB I2C address
  Wire.write(0x00); // Calibration register
  Wire.write(temp >> 8); // High byte of temperature value
  Wire.write(temp & 0xFF); // Low byte of temperature value
  Wire.write(pressure >> 24); // High byte of pressure value
  Wire.write(pressure >> 16); // Middle byte of pressure value
  Wire.write(pressure >> 8); // Low byte of pressure value
  Wire.endTransmission();
}