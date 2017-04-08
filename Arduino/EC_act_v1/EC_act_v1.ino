#include <Wire.h>
#include <Servo.h>

#define SERVO_PIN 3
#define MOTOR_PIN 5
#define LED_PIN 12

#define SERVO_MIN 700
#define SERVO_MED 1500
#define SERVO_MAX 2300

#define PWM_MIN 700
#define PWM_MED 1500
#define PWM_MAX 2300 // 1700 2300


Servo motor;
Servo servo; // -+20 deg = -+250

/* Constants */
int SERVO = 0;
int MOTOR = 1;

int servo_pos = 0;
int motor_vel = PWM_MED;

// TODO: function that maps desired steering angle (-40,...,0,5,10,...,40) to required servo value (650,...,2000)
int steer(int deg) { return deg * 250/20; }
int speed(int vel) { return 1500 + vel*8; }

// ---------- Initialization ----------
void setup() {
    Serial.begin(9600);
    delay(100);

    init_servo();
    init_motor();
    init_led();

    led_blink();
    Serial.println("init done");
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

/**
* Initializes the motor
*/
void init_motor() {
    motor.attach(MOTOR_PIN);

    motor.write(motor_vel);

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
    readSerial();

    updateServo();
    updateMotor();

    delay(20);
}

// TODO: Find  

/**
* Reads the steering and speed input from the USB interface.
*/
void readSerial() {
    while (Serial.available()) {
        read(SERVO, ',');
        read(MOTOR, '.');
    }
}

void read(int mode, char divider) {
    static String data = "";

    while (true){
        char c = (char) Serial.read();

        if (c == -1) { delay(5); } // No data to read
        else if (c == divider) {  // End of String
            switch (mode) {
                case SERVO: servo_pos = data.toInt();
                case MOTOR: motor_vel = data.toInt();
            }
            return;
        } else { data += c; } // Standard Case
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
