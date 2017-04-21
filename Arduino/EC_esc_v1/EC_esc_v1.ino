#include <PID_v1.h>
#include <Wire.h>
#include <Servo.h>

#define MOTOR_PIN 5
#define LED_PIN 12

#define PWM_MIN 700
#define PWM_MED 1500
#define PWM_MAX 2300 // 1700 2300

// 650 := full brake, 1450 := neutral, 2000 := full throttle

Servo motor;

int motor_speed = PWM_MED;

// TODO: function that maps desired steering angle (-40,...,0,5,10,...,40) to required servo value (650,...,2000)

// ---------- Initialization ----------

void setup() {
  Serial.begin(9600);
  delay(100);

  init_motor();
  init_led();

  led_blink();
  Serial.println("init done");
}

/**
 * Initializes the notor
 */
void init_motor() {
  motor.attach(MOTOR_PIN);

  motor.write(motor_speed);

  delay(1000);
  Serial.println("motor init successful");
}

void init_led() {
  pinMode(LED_PIN, OUTPUT);
}
// ---------- Program ----------

/**
 * Main loop called repeatedly.
 */
void loop() {
  // TODO: loop every 2 ms (wait timer)
  readSerial();
  updateMotor();
  delay(100);
  //calibrateESC();
  //while(true){
  //  delay(1000);
  //  Serial.print("...\n");
  //}
}

/**
 * Reads the throttle input from the USB interface.
 */
void readSerial() {
  static String dataString = "";

  while (Serial.available()) {
    char dataChar = (char)Serial.read();
    if (dataChar == '.') { // end of string, value in dataString
      motor_speed = dataString.toInt();

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
void updateMotor() {
  if (motor_speed == 0) {
    motor_speed = PWM_MED; // move to center position
  } else {
      motor_speed = constrain(motor_speed, PWM_MIN, PWM_MAX);
  }

  Serial.print(motor_speed); Serial.print(" ");
  Serial.print("\n");

  motor.writeMicroseconds(motor_speed); // move to desired position
}

void calibrateESC() {
  led_blink();
  Serial.print("neutral\n");
  motor.writeMicroseconds(PWM_MED); // neutral
  delay(3000);
  led_blink() ;
  Serial.print("full throttle\n");
  motor.writeMicroseconds(PWM_MAX); // full throttle
  delay(3000);
  led_blink() ;
  Serial.print("full brake\n");
  motor.writeMicroseconds(PWM_MIN); // full brake
  delay(3000);
  led_blink() ;
  Serial.print("neutral\n");
  motor.writeMicroseconds(PWM_MED); // neutral
  delay(100);
}

void led_blink() {
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
  delay(500);
}

