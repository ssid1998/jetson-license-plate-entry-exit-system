// Zauberstab LED PWM
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
int speed = 10;
 
void setup() {
}
 
void loop() {
  for (int i=255; i>=0; i--){
    analogWrite(ledRot, i);
    delay(speed);
  }
  for (int i=0; i<=255; i++){
    analogWrite(ledRot, i);
    delay(speed);
  }
  for (int i=255; i>=0; i--){
    analogWrite(ledBlau, i);
    delay(speed);
  }
  for (int i=0; i<=255; i++){
    analogWrite(ledBlau, i);
    delay(speed);
  }
    for (int i=255; i>=0; i--){
    analogWrite(ledGruen, i);
    delay(speed);
  }
  for (int i=0; i<=255; i++){
    analogWrite(ledGruen, i);
    delay(speed);
  }
}
