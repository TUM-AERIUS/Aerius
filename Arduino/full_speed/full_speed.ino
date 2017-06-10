#include <Wire.h>
#include <Servo.h>

#define SERVO_PIN 7
#define MOTOR_PIN 5

#define I2C_ADDRESS 0x04

#define SERVO_NEUTRAL       1500
#define SERVO_40_DEG_L      850
#define SERVO_40_DEG_R      2150

#define MOTOR_NEUTRAL       1520 // TODO: verify value, may vary depending on ESC calibration
#define MOTOR_BRAKE_MAX     650
#define MOTOR_THROTTLE_MAX  2000

#define SERVO 0
#define MOTOR 1

// 650 := full brake, 1450 := neutral, 2000 := full throttle
// -+20 deg = -+250

Servo servo;
Servo motor;

/* steering deg [-20,...,20] */
int steering = 0;
/* throttle pct [0,...,100] */
int velocity = 0;

int count = 0; //Variable to check for Failsafe

// ---------- Initialization ----------

void setup() {
    /*Serial.begin(9600);
    delay(100);*/
 
    init_motor();
    init_servo();

    Serial.println("Starting!");
    delay(10000);

    for(int j = 10; j <= 100; j+=10) {
      velocity = j;
      updateMotor();
      delay(100);
    }

    delay(1);
    velocity = 0;
    updateMotor();
    
}

void init_servo() {
    //Serial.println("Starting..");
    servo.attach(SERVO_PIN);
    servo.writeMicroseconds(SERVO_NEUTRAL);

    delay(1000);
    Serial.println("servo init successful");
}

void init_motor() {
    motor.attach(MOTOR_PIN);
    motor.writeMicroseconds(MOTOR_NEUTRAL);

    delay(1000);
    Serial.println("motor init successful");
}

// ---------- Program ----------

void loop() {
}

void updateServo() {
    /* map [-20,...,20] to [SERVO_20_DEG_L,...,SERVO_20_DEG_R] */
    int val = 0;
    if (steering == 0) {
        val = SERVO_NEUTRAL; // move to center position
    } else {
        val = map(steering, -40, 40, SERVO_40_DEG_L, SERVO_40_DEG_R);
    }

    servo.writeMicroseconds(val);
}

void updateMotor() {
    /* map [-100,...,100] to [MOTOR_BRAKE_MAX,...,MOTOR_THROTTLE_MAX] */
    int val = 0;
    if (velocity == 0) {
        val = MOTOR_NEUTRAL; // move to center position (lower than center should brake)
    } else {
        val = map(velocity, 0, 100, MOTOR_NEUTRAL, MOTOR_THROTTLE_MAX);
    }
    // TODO: implement braking, e.g. add -100 to value range

    motor.writeMicroseconds(val);
}
