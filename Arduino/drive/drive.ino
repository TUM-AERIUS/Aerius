#include <Wire.h>
#include <Servo.h>

#define SERVO_PIN 3
#define MOTOR_PIN 5

#define SERVO_NEUTRAL       1500
#define SERVO_20_DEG_L      1100
#define SERVO_20_DEG_R      1900

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

// ---------- Initialization ----------

void setup() {
    Serial.begin(9600);
    delay(100);

    init_motor();
    init_servo();
}

void init_servo() {
    Serial.println("Starting..");
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
    readSerial();

    updateServo();
    updateMotor();

    delay(100);
}

/**
* Reads the throttle and steering input from the USB Serial.
*/
void readSerial() {
    while (Serial.available()) {
        read(SERVO, ',');
        read(MOTOR, '.');
    }
}


void read(int mode, char divider) {
    char data[] = "    ";
    static byte i = 0;

    while (Serial.available()){
        char c = (char) Serial.read();

        // Check for overflow
        if (i >= 3) return;

        if (c == -1) { delay(5); } // No data to read
        else if (c == divider) {  // End of String
            switch (mode) {
                case SERVO: steering = atoi(&data[0]);
                case MOTOR: velocity = atoi(&data[0]);
            }
            return;
        } else { data[i++] = c; } // Standard Case
    }
}

void updateServo() {
    /* map [-20,...,20] to [SERVO_20_DEG_L,...,SERVO_20_DEG_R] */
    int val = 0;
    if (steering == 0) {
        val = SERVO_NEUTRAL; // move to center position
    } else {
        val = map(steering, -20, 20, SERVO_20_DEG_L, SERVO_20_DEG_R);
    }

    servo.writeMicroseconds(val);
}

void updateMotor() {
    /* map [0,...,100] to [MOTOR_NEUTRAL,...,MOTOR_THROTTLE_MAX] */
    int val = 0;
    if (velocity == 0) {
        val = MOTOR_NEUTRAL; // move to center position (lower than center should brake)
    } else {
        val = map(velocity, 0, 100, MOTOR_NEUTRAL, MOTOR_THROTTLE_MAX);
    }
    // TODO: implement braking, e.g. add -100 to value range

    motor.writeMicroseconds(val);
}
