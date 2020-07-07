//created by Adafruit for use with the ina219 breakout board: https://learn.adafruit.com/adafruit-ina219-current-sensor-breakout/arduino-code

//slight modifications made to allow for direct measurements of energy and for csv-friendly serial output

#include <Wire.h>
#include <Adafruit_INA219.h>

unsigned long timer;

Adafruit_INA219 ina219;


void setup(void) 
{
  Serial.begin(115200);
  while (!Serial) {
      // will pause Zero, Leonardo, etc until serial console opens
      delay(1);
  }

  uint32_t currentFrequency;
    
  
  // Initialize the INA219.
  // By default the initialization will use the largest range (32V, 2A).  However
  // you can call a setCalibration function to change this range (see comments).
  ina219.begin();
  // To use a slightly lower 32V, 1A range (higher precision on amps):
  //ina219.setCalibration_32V_1A();
  // Or to use a lower 16V, 400mA range (higher precision on volts and amps):
  //ina219.setCalibration_16V_400mA();

  //first line for csv printed here with commas inbetween fields
  Serial.print("Bus_Voltage(V)"); Serial.print(",");
  Serial.print("Shunt_Voltage(mV)"); Serial.print(",");
  Serial.print("Load_Voltage(V)"); Serial.print(",");
  Serial.print("Current(mA)"); Serial.print(",");
  Serial.print("Power(mW)"); Serial.print(",");
  Serial.print("Time(s)"); Serial.print(",");
  Serial.println("Energy()");
}

void loop(void) 
{
  float shuntvoltage = 0;
  float busvoltage = 0;
  float current_mA = 0;
  float loadvoltage = 0;
  float power_mW = 0;
  float seconds = 0;
  float energy = 0;

  shuntvoltage = ina219.getShuntVoltage_mV();
  busvoltage = ina219.getBusVoltage_V();
  current_mA = ina219.getCurrent_mA();
  power_mW = ina219.getPower_mW();
  loadvoltage = busvoltage + (shuntvoltage / 1000);
  timer = millis();
  seconds = timer/1000;
  energy = power_mW*seconds;


  //serial print with commas in between fields
  //serial monitor on arduino is not used
  //instead the monitor_serial.py script was created to output serial directly to a csv file
  Serial.print(busvoltage); Serial.print(",");
  Serial.print(shuntvoltage); Serial.print(",");
  Serial.print(loadvoltage); Serial.print(",");
  Serial.print(current_mA); Serial.print(",");
  Serial.print(power_mW); Serial.print(",");
  Serial.print(seconds); Serial.print(",");
  Serial.println(energy); 

  //records reading every second
  delay(1000);
}
