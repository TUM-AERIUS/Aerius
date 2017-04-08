#include <Wire.h>
#include <Servo.h>

#define SERVO_PIN 3

#define SERVO_MIN 700
#define SERVO_MED 1500
#define SERVO_MAX 2300

// -+20 deg = -+250

Servo servo;

int servo_pos = 0;

// TODO: function that maps desired steering angle (-40,...,0,5,10,...,40) to required servo value (650,...,2000) 

// ---------- Initialization ----------

void setup() {
  Serial.begin(9600);
  delay(100);

  init_servo();
}

/**
 * Initializes the servo
 */
void init_servo() {
  servo.attach(SERVO_PIN);
   
  servo.write(SERVO_MED);
  
  delay(1000);
  Serial.println("servo init successful");
}

// ---------- Program ----------

/**
 * Main loop called repeatedly.
 */
void loop() {
  // TODO: loop every 2 ms (wait timer)
  readSerial();
  updateServo();
  delay(1000);
}

/**
 * Reads the throttle input from the USB interface.
 */
void readSerial() {
  static String dataString = "";

  while (Serial.available()) {
    char dataChar = (char)Serial.read();
    if (dataChar == '\n') { // end of string, value in dataString
      servo_pos = dataString.toInt();
              
      dataString = "";
      return;
    } else {
      dataString += dataChar;
    }
  }
}

/**
 * Gives the new throttles to the individual engines.
 */
void updateServo() {
  if (servo_pos == 0) {
    servo_pos = SERVO_MED; // move to center position
  } else {
    // servo_pos = constrain(servo_pos, SERVO_MIN, SERVO_MAX);
  }
   
  Serial.print(servo_pos); Serial.print(" ");
  Serial.print("\n");
  
  servo.writeMicroseconds(servo_pos); // move to desired position
}

