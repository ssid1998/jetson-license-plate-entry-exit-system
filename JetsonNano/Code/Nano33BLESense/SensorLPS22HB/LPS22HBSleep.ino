/**
 * @file LPS22HBSleep.ino
 *
 * @brief Sketch to switch the sensor LPS22HB into sleep mode
 *
 * @details Sketch for the Arduino Nano 33 BLE Sense to switch the sensor LPS22HB into sleep mode
 *
 * @author Elmar Wings
 *
 * @version 1.0
  */

#include <Wire.h>
#include <LPS22HB.h>

/**
 * @brief LPS22HB sensor object
 * @details This object represents the LPS22HB sensor and provides methods for reading temperature and pressure values.
 */
LPS22HB lps22hb;

/**
 * @brief Setup function
 *
 * @details This function is called once at the beginning of the program and is used to 
 * - Initialize I2C communication
 * - initialize the sensor, and 
 * - initialize serial communication.
 */
void setup() {
  // initialize serial communication
  Serial.begin(9600);
 
  //Initialize I2C communication
  Wire.begin();
 
  // This line initializes the LPS22HB sensor and sets it up for use.
  lps22hb.begin();
}

/**
 * @brief Loop function
 *
 * @details This function is called repeatedly after the setup function and is used to switch the sensor into sleep mode and wake it up.
 */
void loop() {
  // Switch the sensor into sleep mode
  lps22hb.sleep();
 
  // Wait for 1 second
  delay(1000);
 
  // Switch the sensor out of sleep mode
  lps22hb.wakeUp();
 
  // Wait for 1 second
  delay(1000);
}