// Zauberstab LED on/off
// 240221 Thomas Peetz
//
// An ein Microcontrollerboard Arduino Nano 33 BLE ist eine RGB-LED ueber 180 Ohm Vorwiderstaende angeschlossen.
// Zuordnung der Pins GPIO 02 LED Blau
//                    GPIO 03 LED Gruen
//                    GPIO 04 LED Rot
// Die verwendeten GPIO Pins koennen ueber PWM Signale angesteuert werden, damit ist eine Farbmischung moeglich.
//
 
int ledBlau = 2;
int ledGruen = 3;
int ledRot = 4;
 
void setup() {
  Serial.begin(115200);
}
 
void loop() {
  Serial.println("Rot");
  analogWrite(ledRot, 0);
  delay (1000);
  analogWrite(ledRot, 255);
 
  Serial.println("Blau");
  analogWrite(ledBlau, 0);
  delay (1000);
  analogWrite(ledBlau, 255);
 
  Serial.println("Gruen");
  analogWrite(ledGruen, 0);
  delay (1000);
  analogWrite(ledGruen, 255);
}
