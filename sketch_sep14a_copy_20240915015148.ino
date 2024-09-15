int sound_sensor = A3; //pin A3
#include <Wire.h>
#include "rgb_lcd.h"
#define LED 2
#include <Stepper.h>

const int stepsPerRevolution = 200;
Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);

rgb_lcd lcd;

const int colorR = 50;
const int colorG = 50;
const int colorB = 50;

void setup() 
{
  Serial.begin(9600); //begin Serial Communication
  pinMode(4, OUTPUT); //Buzzer
  pinMode(LED, OUTPUT); //LED
  lcd.begin(16, 2);
  lcd.setRGB(colorR, colorG, colorB);

  myStepper.setSpeed(60);

  lcd.print("safe driving :)");
  delay(1000);
}
 
void loop()
{
  lcd.setCursor(0, 1);

  int lightValue = analogRead(A0);
  // lcd.print(lightValue);
  myStepper.step(stepsPerRevolution);
  if (lightValue > 200) {
    // digitalWrite(4, HIGH); //buzz on
    lcd.clear();
    lcd.print("WAKE UP");
    digitalWrite(LED, HIGH); //LED on
    delay(1000);
    // digitalWrite(4, LOW); //buzz off
    digitalWrite(LED, LOW); //LED off
  }
  else {
    lcd.clear();
    lcd.print("Safe driving :)");
  }

  delay(100); //a shorter delay between readings
}