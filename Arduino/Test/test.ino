#include <Wire.h>
#define LED_PIN 12

const int SERVO = 0;
const int MOTOR = 1;
int servo_pos = 0;
int motor_vel = 0;
boolean done = true;

// ---------- Initialization ----------
void setup() {
    Serial.begin(9600);
    delay(100);

    init_led();

    Serial.println("init done");
}

void init_led() {
    pinMode(LED_PIN, OUTPUT);
}

// ---------- Program ----------

/**
* Main loop called repeatedly.
*/
void loop() {
    readS();

    if (!done) { 
      done = true;
      Serial.println(servo_pos);
      Serial.println(motor_vel); 
    }

    if (servo_pos != 0 && motor_vel != 0) {
        digitalWrite(LED_BUILTIN, HIGH);
    } else { digitalWrite (LED_BUILTIN, LOW); }

    delay(20);
}

/**
* Reads the steering and speed input from the USB interface.
*/
void readS() {
    while (Serial.available()) {
        read(SERVO, ',');
        read(MOTOR, '.');
    }
}


void read(int mode, char divider) {
    String data = "";

    while (true){
        char c = (char) Serial.read();

        if (c == -1) { delay(5); } // No data to read
        else if (c == divider) {  // End of String
            switch (mode) {
                case SERVO: servo_pos = data.toInt();
                case MOTOR: motor_vel = data.toInt();
            }
            done = false;
            return;
        } else { data += c; } // Standard Case
    }
}
