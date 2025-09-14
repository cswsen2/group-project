#include <ArduinoJson.h>

// ==============================
// PIN DEFINITIONS
// ==============================
// Traffic Light Pins (4 lanes: North, East, South, West)
// Each lane has Red, Yellow, Green
const int LANE_PINS[4][3] = {
  {2, 3, 4},    // North Lane: Red, Yellow, Green
  {5, 6, 7},    // East Lane: Red, Yellow, Green
  {8, 9, 10},   // South Lane: Red, Yellow, Green
  {11, 12, 13}  // West Lane: Red, Yellow, Green
};

// Sensor and Indicator Pins
const int RAIN_SENSOR_PIN = A0;  // Rain sensor analog input
const int RAIN_LED_PIN = 22;     // LED to indicate rain detection
const int STATUS_LED_PIN = 23;   // System status LED
const int EMERGENCY_LED_PIN = 24; // Emergency mode indicator LED

// ==============================
// TIMING CONSTANTS
// ==============================
const int NORMAL_GREEN_TIME = 15000;    // 15 seconds normal green
const int NORMAL_YELLOW_TIME = 3000;    // 3 seconds yellow
const int EMERGENCY_RESPONSE_TIME = 500; // 0.5 seconds emergency response
const int PEDESTRIAN_YELLOW_TIME = 2000; // 2 seconds gradual yellow for pedestrians
const int RAIN_DELAY_MULTIPLIER = 2;     // Double timing when raining

// Rain detection threshold (adjust based on your sensor)
const int RAIN_THRESHOLD = 300;  // Analog reading threshold for rain

// ==============================
// SYSTEM VARIABLES
// ==============================
struct LaneState {
  int red_pin, yellow_pin, green_pin;
  bool is_green, is_yellow, is_red;
  unsigned long state_start_time;
};

LaneState lanes[4];
int current_priority_lane = 0;
String current_priority_type = "normal";
bool emergency_active = false;
bool rain_detected = false;
bool system_initialized = false;

unsigned long last_data_received = 0;
const unsigned long DATA_TIMEOUT = 10000; // 10 seconds timeout

// JSON document for parsing
StaticJsonDocument<1024> json_doc;

// ==============================
// SETUP FUNCTION
// ==============================
void setup() {
  Serial.begin(9600);
  
  // Initialize lane structures and pins
  for (int i = 0; i < 4; i++) {
    lanes[i].red_pin = LANE_PINS[i][0];
    lanes[i].yellow_pin = LANE_PINS[i][1];
    lanes[i].green_pin = LANE_PINS[i][2];
    
    // Set pins as output
    pinMode(lanes[i].red_pin, OUTPUT);
    pinMode(lanes[i].yellow_pin, OUTPUT);
    pinMode(lanes[i].green_pin, OUTPUT);
    
    // Initialize all lights as RED (safe state)
    digitalWrite(lanes[i].red_pin, HIGH);
    digitalWrite(lanes[i].yellow_pin, LOW);
    digitalWrite(lanes[i].green_pin, LOW);
    
    lanes[i].is_red = true;
    lanes[i].is_yellow = false;
    lanes[i].is_green = false;
    lanes[i].state_start_time = millis();
  }
  
  // Initialize sensor and indicator pins
  pinMode(RAIN_SENSOR_PIN, INPUT);
  pinMode(RAIN_LED_PIN, OUTPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(EMERGENCY_LED_PIN, OUTPUT);
  
  // Turn on status LED to indicate system is ready
  digitalWrite(STATUS_LED_PIN, HIGH);
  
  Serial.println("🚦 AI Traffic Management System - Arduino Controller");
  Serial.println("✅ System initialized - All lanes set to RED (Safe Mode)");
  Serial.println("📡 Waiting for Python script data...");
  Serial.println("==========================================");
}

// ==============================
// MAIN LOOP
// ==============================
void loop() {
  // Check for rain
  checkRainSensor();
  
  // Check for incoming serial data
  if (Serial.available()) {
    String json_string = Serial.readStringUntil('\n');
    parseTrafficData(json_string);
    last_data_received = millis();
  }
  
  // Handle traffic light control
  controlTrafficLights();
  
  // Safety timeout - if no data received for too long, go to safe mode
  if (millis() - last_data_received > DATA_TIMEOUT && system_initialized) {
    enterSafeMode();
  }
  
  // Blink status LED to show system is running
  blinkStatusLED();
  
  delay(50); // Small delay to prevent excessive processing
}

// ==============================
// RAIN SENSOR FUNCTIONS
// ==============================
void checkRainSensor() {
  int rain_value = analogRead(RAIN_SENSOR_PIN);
  bool previous_rain_state = rain_detected;
  
  rain_detected = (rain_value > RAIN_THRESHOLD);
  
  // Update rain LED
  digitalWrite(RAIN_LED_PIN, rain_detected ? HIGH : LOW);
  
  // Print rain status change
  if (rain_detected != previous_rain_state) {
    Serial.println("==========================================");
    if (rain_detected) {
      Serial.print("🌧️ RAIN DETECTED! Sensor value: ");
      Serial.println(rain_value);
      Serial.println("⏰ Switching to slower timing mode");
    } else {
      Serial.print("☀️ Rain stopped. Sensor value: ");
      Serial.println(rain_value);
      Serial.println("⏰ Returning to normal timing mode");
    }
    Serial.println("==========================================");
  }
}

// ==============================
// DATA PARSING FUNCTIONS
// ==============================
void parseTrafficData(String json_string) {
  // Clear previous data
  json_doc.clear();
  
  // Parse JSON
  DeserializationError error = deserializeJson(json_doc, json_string);
  
  if (error) {
    Serial.print("❌ JSON parsing error: ");
    Serial.println(error.c_str());
    return;
  }
  
  // Extract data
  int new_priority_lane = json_doc["priority_lane"];
  String new_priority_type = json_doc["priority_type"].as<String>();
  
  // Print received data
  Serial.println("==========================================");
  Serial.println("📡 RECEIVED TRAFFIC DATA");
  Serial.println("==========================================");
  Serial.print("🎯 Priority Lane: ");
  Serial.print(getLaneName(new_priority_lane));
  Serial.print(" (ID: ");
  Serial.print(new_priority_lane);
  Serial.println(")");
  Serial.print("🔥 Priority Type: ");
  new_priority_type.toUpperCase();
  Serial.println(new_priority_type);
  
  // Print lane detection data
  JsonObject lane_data = json_doc["lane_data"];
  for (int i = 0; i < 4; i++) {
    Serial.print("📊 ");
    Serial.print(getLaneName(i));
    Serial.print(" Lane: ");
    
    JsonObject lane = lane_data[String(i)];
    Serial.print("Cars:");
    Serial.print((int)lane["car"]);
    Serial.print(" Emergency:");
    Serial.print((int)lane["emergency"]);
    Serial.print(" Heavy:");
    Serial.print((int)lane["heavy"]);
    Serial.print(" Pedestrian:");
    Serial.print((int)lane["pedestrian"]);
    Serial.print(" Public:");
    Serial.println((int)lane["public"]);
  }
  
  // Check if this is an emergency situation
  bool new_emergency = (new_priority_type == "emergency");
  
  if (new_emergency && !emergency_active) {
    Serial.println("🚨 EMERGENCY MODE ACTIVATED! 🚨");
    Serial.println("⚡ Immediate response required!");
  } else if (!new_emergency && emergency_active) {
    Serial.println("✅ Emergency cleared - returning to normal operation");
  }
  
  // Update system state
  current_priority_lane = new_priority_lane;
  current_priority_type = new_priority_type;
  emergency_active = new_emergency;
  system_initialized = true;
  
  Serial.println("==========================================");
}

// ==============================
// TRAFFIC LIGHT CONTROL
// ==============================
void controlTrafficLights() {
  if (!system_initialized) {
    return; // Wait for first data
  }
  
  if (emergency_active) {
    handleEmergencyMode();
  } else if (current_priority_type == "pedestrian") {
    handlePedestrianMode();
  } else {
    handleNormalMode();
  }
}

void handleEmergencyMode() {
  // Turn on emergency LED
  digitalWrite(EMERGENCY_LED_PIN, HIGH);
  
  // Immediately set all lanes to RED except priority lane
  for (int i = 0; i < 4; i++) {
    if (i != current_priority_lane) {
      setLaneState(i, "red");
    }
  }
  
  // Give GREEN to emergency lane immediately
  setLaneState(current_priority_lane, "green");
}

void handlePedestrianMode() {
  // Turn off emergency LED
  digitalWrite(EMERGENCY_LED_PIN, LOW);
  
  // Gradually transition other lanes to RED via YELLOW
  for (int i = 0; i < 4; i++) {
    if (i != current_priority_lane) {
      // If currently green, go to yellow first
      if (lanes[i].is_green) {
        setLaneState(i, "yellow");
      }
      // If yellow for enough time, go to red
      else if (lanes[i].is_yellow && 
               (millis() - lanes[i].state_start_time > getYellowTime())) {
        setLaneState(i, "red");
      }
    }
  }
  
  // Set priority lane to GREEN (pedestrian can cross)
  setLaneState(current_priority_lane, "green");
}

void handleNormalMode() {
  // Turn off emergency LED
  digitalWrite(EMERGENCY_LED_PIN, LOW);
  
  // FIXED: When priority lane changes, transition other lanes through yellow to red
  static int last_priority_lane = -1;
  
  if (last_priority_lane != current_priority_lane) {
    // Priority lane has changed - transition all other green lanes to yellow first
    for (int i = 0; i < 4; i++) {
      if (i != current_priority_lane && lanes[i].is_green) {
        setLaneState(i, "yellow");
      }
    }
    last_priority_lane = current_priority_lane;
  }
  
  // Handle transitions for all lanes
  for (int i = 0; i < 4; i++) {
    if (i == current_priority_lane) {
      // Priority lane: Red -> Green (after others are clear)
      bool others_cleared = true;
      for (int j = 0; j < 4; j++) {
        if (j != i && !lanes[j].is_red) {
          others_cleared = false;
          break;
        }
      }
      
      if (others_cleared) {
        setLaneState(i, "green");
      }
    } else {
      // Non-priority lanes: handle Yellow -> Red transition
      if (lanes[i].is_yellow) {
        if (millis() - lanes[i].state_start_time > getYellowTime()) {
          setLaneState(i, "red");
        }
      }
    }
  }
}

void setLaneState(int lane_id, String state) {
  if (lane_id < 0 || lane_id > 3) return;
  
  LaneState* lane = &lanes[lane_id];
  bool state_changed = false;
  
  // Update state
  if (state == "red" && !lane->is_red) {
    digitalWrite(lane->red_pin, HIGH);
    digitalWrite(lane->yellow_pin, LOW);
    digitalWrite(lane->green_pin, LOW);
    lane->is_red = true;
    lane->is_yellow = false;
    lane->is_green = false;
    state_changed = true;
  } else if (state == "yellow" && !lane->is_yellow) {
    digitalWrite(lane->red_pin, LOW);
    digitalWrite(lane->yellow_pin, HIGH);
    digitalWrite(lane->green_pin, LOW);
    lane->is_red = false;
    lane->is_yellow = true;
    lane->is_green = false;
    state_changed = true;
  } else if (state == "green" && !lane->is_green) {
    digitalWrite(lane->red_pin, LOW);
    digitalWrite(lane->yellow_pin, LOW);
    digitalWrite(lane->green_pin, HIGH);
    lane->is_red = false;
    lane->is_yellow = false;
    lane->is_green = true;
    state_changed = true;
  }
  
  // Update timestamp if state changed
  if (state_changed) {
    lane->state_start_time = millis();
    Serial.print("🚦 ");
    Serial.print(getLaneName(lane_id));
    Serial.print(" Lane -> ");
    state.toUpperCase();
    Serial.println(state);
  }
}

// ==============================
// TIMING FUNCTIONS
// ==============================
unsigned long getGreenTime() {
  return rain_detected ? (NORMAL_GREEN_TIME * RAIN_DELAY_MULTIPLIER) : NORMAL_GREEN_TIME;
}

unsigned long getYellowTime() {
  // Use standard yellow time for normal and emergency modes
  return rain_detected ? (NORMAL_YELLOW_TIME * RAIN_DELAY_MULTIPLIER) : NORMAL_YELLOW_TIME;
}

unsigned long getPedestrianYellowTime() {
  // Special longer yellow time for pedestrian safety (gradual transition)
  return rain_detected ? (PEDESTRIAN_YELLOW_TIME * RAIN_DELAY_MULTIPLIER) : PEDESTRIAN_YELLOW_TIME;
}

// ==============================
// UTILITY FUNCTIONS
// ==============================
String getLaneName(int lane_id) {
  switch (lane_id) {
    case 0: return "North";
    case 1: return "East";
    case 2: return "South";
    case 3: return "West";
    default: return "Unknown";
  }
}

void enterSafeMode() {
  Serial.println("⚠️ ENTERING SAFE MODE - No data from Python script");
  Serial.println("🔴 All lanes set to RED for safety");
  
  // Set all lanes to RED
  for (int i = 0; i < 4; i++) {
    setLaneState(i, "red");
  }
  
  // Turn off emergency LED
  digitalWrite(EMERGENCY_LED_PIN, LOW);
  
  system_initialized = false;
  emergency_active = false;
}

void blinkStatusLED() {
  static unsigned long last_blink = 0;
  static bool led_state = false;
  
  if (millis() - last_blink > 1000) { // Blink every second
    led_state = !led_state;
    digitalWrite(STATUS_LED_PIN, led_state ? HIGH : LOW);
    last_blink = millis();
  }
}

// ==============================
// DEBUG FUNCTIONS
// ==============================
void printSystemStatus() {
  Serial.println("==========================================");
  Serial.println("🔍 SYSTEM STATUS");
  Serial.println("==========================================");
  Serial.print("⏰ Current Time: ");
  Serial.println(millis());
  Serial.print("🌧️ Rain Detected: ");
  Serial.println(rain_detected ? "YES" : "NO");
  Serial.print("🚨 Emergency Active: ");
  Serial.println(emergency_active ? "YES" : "NO");
  Serial.print("🎯 Priority Lane: ");
  Serial.println(getLaneName(current_priority_lane));
  Serial.print("📡 Last Data: ");
  Serial.print((millis() - last_data_received) / 1000);
  Serial.println(" seconds ago");
  
  // Print lane states
  for (int i = 0; i < 4; i++) {
    Serial.print("🚦 ");
    Serial.print(getLaneName(i));
    Serial.print(": ");
    if (lanes[i].is_red) Serial.print("RED");
    else if (lanes[i].is_yellow) Serial.print("YELLOW");
    else if (lanes[i].is_green) Serial.print("GREEN");
    Serial.print(" (");
    Serial.print((millis() - lanes[i].state_start_time) / 1000);
    Serial.println("s)");
  }
  Serial.println("==========================================");
}
