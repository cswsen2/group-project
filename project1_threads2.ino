#include <ArduinoJson.h>

// ==============================
// Pin Definitions for Traffic Lights
// ==============================
// North Lane (Lane 0)
#define NORTH_RED_PIN     2
#define NORTH_YELLOW_PIN  3
#define NORTH_GREEN_PIN   4

// East Lane (Lane 1)
#define EAST_RED_PIN      5
#define EAST_YELLOW_PIN   6
#define EAST_GREEN_PIN    7

// South Lane (Lane 2)
#define SOUTH_RED_PIN     8
#define SOUTH_YELLOW_PIN  9
#define SOUTH_GREEN_PIN   10

// West Lane (Lane 3)
#define WEST_RED_PIN      11
#define WEST_YELLOW_PIN   12
#define WEST_GREEN_PIN    13

#define pedestrian0 22
#define pedestrian1 23
#define pedestrian2 24
#define pedestrian3 25


// Optional: Status LED and Buzzer
#define STATUS_LED_PIN    A0
#define BUZZER_PIN        A1

// Rain Sensor
#define RAIN_SENSOR_PIN   A2
#define RAIN_STATUS_LED   A3

// ==============================
// Configuration
// ==============================
#define SERIAL_BAUDRATE   9600
#define JSON_BUFFER_SIZE  1024
#define PEDESTRIAN_CROSSING_TIME 15000 // 15 seconds for pedestrian crossing
#define RAINY_PEDESTRIAN_TIME 20000    // 20 seconds for pedestrian crossing when raining

// Rain Detection
#define RAIN_THRESHOLD    500     // Analog threshold for rain detection (adjust as needed)
#define RAIN_CHECK_INTERVAL 2000  // Check rain sensor every 2 seconds

// ==============================
// Global Variables
// ==============================
int currentPriorityLane = 0;
int previousPriorityLane = -1;  // Track previous priority lane
String currentPriorityType = "normal";
bool pedestrianActive = false;
bool emergencyActive = false;

// Rain Detection Variables
bool rainDetected = false;
unsigned long lastRainCheck = 0;

unsigned long lastUpdate = 0;
unsigned long pedestrianStartTime = 0;

// Traffic light state arrays
int redPins[4] = {NORTH_RED_PIN, EAST_RED_PIN, SOUTH_RED_PIN, WEST_RED_PIN};
int yellowPins[4] = {NORTH_YELLOW_PIN, EAST_YELLOW_PIN, SOUTH_YELLOW_PIN, WEST_YELLOW_PIN};
int greenPins[4] = {NORTH_GREEN_PIN, EAST_GREEN_PIN, SOUTH_GREEN_PIN, WEST_GREEN_PIN};
int pedestrianLights[4] = {pedestrian0,pedestrian1,pedestrian2,pedestrian3};

String laneNames[4] = {"North", "East", "South", "West"};

// ==============================
// Setup Function
// ==============================
void setup() {
  // Initialize serial communication
  Serial.begin(SERIAL_BAUDRATE);
  
  // Initialize all traffic light pins
  pinMode(pedestrian0, OUTPUT);
  pinMode(pedestrian1, OUTPUT);
  pinMode(pedestrian2, OUTPUT);
  pinMode(pedestrian3, OUTPUT);

  for (int i = 0; i < 4; i++) {
    pinMode(redPins[i], OUTPUT);
    pinMode(yellowPins[i], OUTPUT);
    pinMode(greenPins[i], OUTPUT);
  }
  
  // Initialize status indicators
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RAIN_STATUS_LED, OUTPUT);
  
  // Initialize rain sensor
  pinMode(RAIN_SENSOR_PIN, INPUT);
  
  // Start with all lanes red for safety
  setAllLanesRed();
  
  // Status indication
  digitalWrite(STATUS_LED_PIN, HIGH);
  digitalWrite(RAIN_STATUS_LED, LOW);
  
  Serial.println("=================================");
  Serial.println("🚦 Arduino Traffic Light Controller");
  Serial.println("🎯 Features: Emergency Priority, Pedestrian Safety, Rain Detection");
  Serial.println("🌧️ Rain sensor initialized on pin A2");
  Serial.println("🤖 Full Python control - Arduino only executes commands");
  Serial.println("📡 Waiting for Python data...");
  Serial.println("=================================");
  
  lastUpdate = millis();
  lastRainCheck = millis();
}

// ==============================
// Main Loop
// ==============================
void loop() {
  // Check for rain
  checkRainSensor();
  
  // Check for incoming serial data
  if (Serial.available()) {
    processSerialData();
  }
  
  // Only manage pedestrian crossing timing (safety requirement)
  managePedestrianCrossing();
  
  // Status LED heartbeat (blink every second)
  if (millis() % 1000 < 100) {
    digitalWrite(STATUS_LED_PIN, HIGH);
  } else {
    digitalWrite(STATUS_LED_PIN, LOW);
  }
}

// ==============================
// Rain Detection
// ==============================
void checkRainSensor() {
  unsigned long currentTime = millis();
  
  // Check rain sensor every RAIN_CHECK_INTERVAL milliseconds
  if (currentTime - lastRainCheck >= RAIN_CHECK_INTERVAL) {
    int rainValue = analogRead(RAIN_SENSOR_PIN);
    bool previousRainState = rainDetected;
    
    // Rain is detected when sensor value is below threshold (wet sensor has lower resistance)
    rainDetected = (rainValue < RAIN_THRESHOLD);
    
    // Update rain status LED
    digitalWrite(RAIN_STATUS_LED, rainDetected ? HIGH : LOW);
    
    // Print status when rain state changes
    if (rainDetected != previousRainState) {
      if (rainDetected) {
        Serial.println("🌧️ RAIN DETECTED! Extended pedestrian crossing time available");
        Serial.println("   Rain sensor value: " + String(rainValue));
      } else {
        Serial.println("☀️ Rain stopped. Normal pedestrian crossing time");
        Serial.println("   Rain sensor value: " + String(rainValue));
      }
    }
    
    lastRainCheck = currentTime;
  }
}

// ==============================
// Get Current Pedestrian Crossing Time Based on Rain Status
// ==============================
unsigned long getCurrentPedestrianTime() {
  return rainDetected ? RAINY_PEDESTRIAN_TIME : PEDESTRIAN_CROSSING_TIME;
}

// ==============================
// Serial Data Processing
// ==============================
void processSerialData() {
  String jsonString = Serial.readStringUntil('\n');
  jsonString.trim();
  
  if (jsonString.length() == 0) return;
  
  // Parse JSON
  StaticJsonDocument<JSON_BUFFER_SIZE> doc;
  DeserializationError error = deserializeJson(doc, jsonString);
  
  if (error) {
    Serial.println("❌ JSON parsing failed: " + String(error.c_str()));
    return;
  }
  
  // Extract data
  int priorityLane = doc["priority_lane"];
  int prevPriorityLane = doc["previous_priority_lane"];
  String priorityType = doc["priority_type"].as<String>();
  
  Serial.println("\n📨 Received data from Python:");
  Serial.println("  Priority Lane: " + String(priorityLane));
  Serial.println("  Previous Priority Lane: " + String(prevPriorityLane));
  Serial.println("  Priority Type: " + priorityType);
  if (rainDetected) {
    Serial.println("  🌧️ Rain Mode: Extended pedestrian time if needed");
  }
  

  if (pedestrianActive && priorityType != "emergency") {
    Serial.println("🚶 Pedestrian crossing in progress. Ignoring new command until finished.");
    return;
  }

  // Update global state
  currentPriorityLane = priorityLane;
  previousPriorityLane = prevPriorityLane;
  currentPriorityType = priorityType;
  
  // Handle different priority types
  if (priorityType == "emergency") {
    digitalWrite(pedestrianLights[prevPriorityLane], LOW);
    handleEmergencyVehicle(priorityLane, prevPriorityLane);
    
  } else if (priorityType == "pedestrian_safety") {
    digitalWrite(pedestrianLights[prevPriorityLane], HIGH);
    handlePedestrianSafety(priorityLane, prevPriorityLane);
  } else {
    digitalWrite(pedestrianLights[prevPriorityLane], LOW);
    handleNormalTraffic(priorityLane, prevPriorityLane);
  }

  
  
  lastUpdate = millis();
}

// ==============================
// Pedestrian Crossing Management (Only for pedestrians)
// ==============================
void managePedestrianCrossing() {
  if (!pedestrianActive) return;
  
  unsigned long currentTime = millis();
  unsigned long pedestrianElapsed = currentTime - pedestrianStartTime;
  
  // Check if pedestrian crossing time is complete
  if (pedestrianElapsed >= getCurrentPedestrianTime()) {
    pedestrianActive = false;
    Serial.println("🚶 Pedestrian crossing time completed");
    Serial.println("✅ Ready for next Python command");
  }
}

// ==============================
// Priority Handlers
// ==============================
void handlePedestrianSafety(int lane, int prevLane) {
  Serial.println("🚶 PEDESTRIAN SAFETY MODE ACTIVATED");
  Serial.println("🔴 All lanes set to RED for pedestrian crossing");
  if (rainDetected) {
    Serial.println("🌧️ Extended pedestrian crossing time due to rain: " + String(getCurrentPedestrianTime()/1000) + " seconds");
  } else {
    Serial.println("⏱️ Pedestrian crossing time: " + String(getCurrentPedestrianTime()/1000) + " seconds");
  }
  
  pedestrianActive = true;
  emergencyActive = false;


  
  // Set previous priority lane to yellow, then all red
  if (prevLane >= 0 && prevLane <= 3) {
    Serial.println("🟡 Setting " + laneNames[prevLane] + " to YELLOW (transition)");
    digitalWrite(yellowPins[prevLane], HIGH);
    digitalWrite(greenPins[prevLane], LOW);
    delay(2000); // Yellow for 2 seconds
  }
  
  setAllLanesRed();

  
  activatePedestrianSignal();
  
  pedestrianStartTime = millis();
}

void handleEmergencyVehicle(int lane, int prevLane) {
  Serial.println("🚨 EMERGENCY VEHICLE DETECTED!");
  Serial.println("🟢 Immediate priority given to " + laneNames[lane] + " lane");
  Serial.println("⚠️  Emergency overrides pedestrian safety");
  
  emergencyActive = true;
  pedestrianActive = false;  // Emergency overrides pedestrian mode

   if (previousPriorityLane == lane && emergencyActive) {
    Serial.println("⏱️ Continuing EMERGENCY GREEN for " + laneNames[lane]);
    digitalWrite(redPins[lane], LOW);
    digitalWrite(greenPins[lane], HIGH);
    return;
  }
  
  // Set previous priority lane to yellow if different from emergency lane
  if (prevLane >= 0 && prevLane <= 3 ) {
    Serial.println("🟡 Setting " + laneNames[prevLane] + " to YELLOW (transition)");
    digitalWrite(yellowPins[prevLane], HIGH);
    digitalWrite(greenPins[prevLane], LOW);
    delay(1500); // Shorter yellow for emergency
  }
  
  // All red for safety
  setAllLanesRed();
  delay(5000); // Brief all-red for safety
  
  // Green for emergency lane
  digitalWrite(redPins[lane], LOW);
  digitalWrite(greenPins[lane], HIGH);
  delay(2000);
  Serial.println("🟢 " + laneNames[lane] + " lane activated for EMERGENCY");
  
  activateEmergencyAlert();
  
  Serial.println("🤖 Python controls emergency duration");
}

void handleNormalTraffic(int lane, int prevLane) {
  Serial.println("🚦 Normal traffic command received");
  Serial.println("🟢 Setting " + laneNames[lane] + " lane to GREEN");
  
  emergencyActive = false;
  pedestrianActive = false;

  if (previousPriorityLane == lane && emergencyActive == false && pedestrianActive == false) {
    Serial.println("⏱️ Continuing EMERGENCY GREEN for " + laneNames[lane]);
    return;
  }
  
  // Set previous priority lane to yellow if different from new priority lane
  if (prevLane >= 0 && prevLane <= 3 && prevLane != lane) {
    Serial.println("🟡 Setting " + laneNames[prevLane] + " to YELLOW (transition)");
    digitalWrite(yellowPins[prevLane], HIGH);
    digitalWrite(greenPins[prevLane], LOW);
    delay(2000); // Yellow for 2 seconds
  }


  
  // All red first
  setAllLanesRed();
  delay(500); // Brief safety delay

  digitalWrite(yellowPins[lane], HIGH);
  delay(500);
  digitalWrite(yellowPins[lane], LOW);

  
  // Green for priority lane
  digitalWrite(redPins[lane], LOW);
  digitalWrite(greenPins[lane], HIGH);
  Serial.println("🟢 " + laneNames[lane] + " lane activated");
  
  Serial.println("✅ Lane activated - Python controls duration");
}

// ==============================
// Utility Functions
// ==============================
void setAllLanesRed() {
  // Turn on all red lights, turn off yellow and green
  for (int i = 0; i < 4; i++) {
    digitalWrite(redPins[i], HIGH);
    digitalWrite(yellowPins[i], LOW);
    digitalWrite(greenPins[i], LOW);
  }
}

void setAllLanesOff() {
  // Turn off all lights (emergency/maintenance mode)
  for (int i = 0; i < 4; i++) {
    digitalWrite(redPins[i], LOW);
    digitalWrite(yellowPins[i], LOW);
    digitalWrite(greenPins[i], LOW);
  }
}

void activatePedestrianSignal() {
  // Visual and audio indication for pedestrian crossing
  Serial.println("🔊 Pedestrian crossing signal activated");
  
  // Sound pattern for pedestrian crossing
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);
  }
}

void activateEmergencyAlert() {
  // Visual and audio indication for emergency vehicle
  Serial.println("🔊 Emergency vehicle alert activated");
  
  // Sound pattern for emergency
  for (int i = 0; i < 5; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(100);
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
  }
}
