/*
 * servo_control.ino
 * -----------------
 * Controlador de servos para el proyecto "Death Ray".
 *
 * Controla 2 servomotores (pan horizontal + tilt vertical) mediante comandos
 * recibidos por el puerto serie.
 *
 * Formato de comandos: "X,Y\n"
 *   X = ángulo horizontal (0-180)
 *   Y = ángulo vertical   (0-180)
 *
 * El servo se mueve suavemente de a 1 grado por paso.
 * Al iniciar, envía "READY\n" por el puerto serie.
 *
 * Conexiones:
 *   Servo horizontal (pan)  -> Pin 9
 *   Servo vertical  (tilt)  -> Pin 10
 */

#include <Servo.h>

// -----------------------------------------------------------------------
// Configuración
// -----------------------------------------------------------------------
const int PIN_SERVO_PAN  = 9;
const int PIN_SERVO_TILT = 10;
const int BAUD_RATE      = 9600;
const int STEP_DELAY_MS  = 15;   // ms entre cada paso de 1 grado

// -----------------------------------------------------------------------
// Variables globales
// -----------------------------------------------------------------------
Servo servoPan;
Servo servoTilt;

int currentPan  = 90;  // posición inicial (centro)
int currentTilt = 90;

int targetPan  = 90;
int targetTilt = 90;

String inputBuffer = "";

// -----------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------
void setup() {
  Serial.begin(BAUD_RATE);

  servoPan.attach(PIN_SERVO_PAN);
  servoTilt.attach(PIN_SERVO_TILT);

  servoPan.write(currentPan);
  servoTilt.write(currentTilt);

  delay(500);
  Serial.println("READY");
}

// -----------------------------------------------------------------------
// Loop principal
// -----------------------------------------------------------------------
void loop() {
  // Leer comandos del puerto serie
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // Mover servos suavemente hacia el objetivo
  moveServos();
}

// -----------------------------------------------------------------------
// Parsear y validar el comando "X,Y"
// -----------------------------------------------------------------------
void processCommand(String cmd) {
  cmd.trim();
  int commaIndex = cmd.indexOf(',');
  if (commaIndex < 0) return;  // formato incorrecto

  int x = cmd.substring(0, commaIndex).toInt();
  int y = cmd.substring(commaIndex + 1).toInt();

  // Validar rango 0-180
  x = constrain(x, 0, 180);
  y = constrain(y, 0, 180);

  targetPan  = x;
  targetTilt = y;
}

// -----------------------------------------------------------------------
// Mover servos un paso hacia el objetivo
// -----------------------------------------------------------------------
void moveServos() {
  bool moved = false;

  if (currentPan != targetPan) {
    currentPan += (targetPan > currentPan) ? 1 : -1;
    servoPan.write(currentPan);
    moved = true;
  }

  if (currentTilt != targetTilt) {
    currentTilt += (targetTilt > currentTilt) ? 1 : -1;
    servoTilt.write(currentTilt);
    moved = true;
  }

  if (moved) {
    delay(STEP_DELAY_MS);
  }
}
