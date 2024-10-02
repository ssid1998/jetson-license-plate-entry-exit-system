void setup()
{
  Serial.begin(115200); // Sets the data rate in bits per second (baud) for serial datra transmission
  pinMode(LEDR,OUTPUT); // LEDB = blue, LEDG = LED_BUILTIN = green, LEDR = red (typically for errors)
  #ifdef  CORE_CM7      // shows how to specify a specific core 
    bootM4();
  #endif
}

void loop()
{
  Serial.println("Serial print works on the M7 core, just to ignored on the M4 core, unless use RPC!");
  digitalWrite(LEDR, LOW); // Portenta onboard LED connected to 3V3 so ground is to light
  delay(1000);             // wait 1s
  digitalWrite(LEDR, HIGH); // turn the LED off by not grounding it, weired eh.
  delay(3000);              // wait 3s
}