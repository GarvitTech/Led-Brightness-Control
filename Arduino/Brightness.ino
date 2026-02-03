// arduino_brightness_control.ino
// Connect LED to Pin 9 (PWM capable pin)

void setup() {
  Serial.begin(9600);        // Initialize serial communication
  pinMode(9, OUTPUT);        // Set pin 9 as output for LED (PWM)
  analogWrite(9, 0);         // Start with LED off (0 brightness)
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Brightness value should be 0-255
    int brightness = command.toInt();
    
    // Ensure brightness is within valid range
    if (brightness >= 0 && brightness <= 255) {
      analogWrite(9, brightness);
      
      // Optional: Send confirmation back
      Serial.print("Brightness set to: ");
      Serial.println(brightness);
    }
  }
}