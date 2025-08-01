#include <Wire.h>
#include <Adafruit_INA219.h>
#include <Servo.h> 
#include "AS5600.h"
#include "protocol.h"

const int numSamples = 10;
const int sampleDelay = 250;

Adafruit_INA219 ina219(0x41);
AS5600L as5600;
Servo servo;
unsigned long _time;
int pwmPin = A1;

// Protocol variables
uint16_t sample_rate = 100; // Hz

struct SensorReading {
  float current_mA;
  float voltage_V;
  float power_mW;
};

SensorReading getSampledReading(Adafruit_INA219 &sensor) {
  float currentSum = 0;
  float voltageSum = 0;
  float powerSum = 0;
  float peakCurrent = 0;
  
  SensorReading result;
  result.current_mA = sensor.getCurrent_mA();
  result.voltage_V = sensor.getBusVoltage_V();
  result.power_mW = sensor.getPower_mW();
  return result;
}

void send_binary_message(uint8_t msg_type, const void* payload, uint8_t payload_len) {
  struct ProtocolMessage msg;
  
  // Set header
  msg.header.header1 = PROTOCOL_HEADER_1;
  msg.header.header2 = PROTOCOL_HEADER_2;
  msg.header.msg_type = msg_type;
  msg.header.payload_len = payload_len;
  
  // Copy payload
  if (payload && payload_len > 0) {
    memcpy(msg.payload, payload, payload_len);
  }
  
  // Calculate checksum
  msg.checksum = calculate_checksum((const uint8_t*)&msg, 
                                   sizeof(struct ProtocolHeader) + payload_len);
  
  // Send message
  Serial.write((const uint8_t*)&msg, sizeof(struct ProtocolHeader) + payload_len + 1);
}

void send_sensor_data(uint32_t delta_time, uint16_t pwm, float current, float voltage, int32_t position) {
  struct SensorDataPayload payload;
  payload.delta_time_us = delta_time;
  payload.pwm_value = pwm;
  payload.current_mA = current;
  payload.voltage_V = voltage;
  payload.position = position;
  
  send_binary_message(MSG_TYPE_DATA, &payload, sizeof(payload));
}

void send_command_ack(uint8_t command_type, bool success) {
  uint8_t payload[2] = {command_type, success ? 1 : 0};
  send_binary_message(MSG_TYPE_COMMAND_ACK, payload, 2);
}

void send_error(uint8_t error_code) {
  send_binary_message(MSG_TYPE_ERROR, &error_code, 1);
}

void process_command(uint8_t cmd_type, const uint8_t* payload, uint8_t payload_len) {
  bool success = true;
  
  switch (cmd_type) {
    case CMD_SET_SERVO_POSITION: {
      if (payload_len == sizeof(struct SetServoPositionPayload)) {
        struct SetServoPositionPayload* cmd = (struct SetServoPositionPayload*)payload;
        uint16_t position_deg = cmd->position_deg_x10 / 10;
        if (position_deg <= 180) {
          servo.write(position_deg);
        } else {
          success = false;
        }
      } else {
        success = false;
      }
      break;
    }
    
    case CMD_RESET_ENCODER: {
      as5600.resetPosition();
      break;
    }
    
    case CMD_SET_SAMPLE_RATE: {
      if (payload_len == sizeof(struct SetSampleRatePayload)) {
        struct SetSampleRatePayload* cmd = (struct SetSampleRatePayload*)payload;
        if (cmd->samples_per_sec > 0 && cmd->samples_per_sec <= 1000) {
          sample_rate = cmd->samples_per_sec;
        } else {
          success = false;
        }
      } else {
        success = false;
      }
      break;
    }
    
    default:
      success = false;
      break;
  }
  
  send_command_ack(cmd_type, success);
}

void check_for_commands() {
  static uint8_t rx_buffer[sizeof(struct ProtocolMessage)];
  static uint8_t rx_index = 0;
  static bool in_message = false;
  static uint8_t expected_length = 0;
  
  while (Serial.available()) {
    uint8_t byte = Serial.read();
    
    if (!in_message) {
      // Look for sync bytes
      if (rx_index == 0 && byte == PROTOCOL_HEADER_1) {
        rx_buffer[rx_index++] = byte;
      } else if (rx_index == 1 && byte == PROTOCOL_HEADER_2) {
        rx_buffer[rx_index++] = byte;
      } else if (rx_index == 2) {
        rx_buffer[rx_index++] = byte; // msg_type
      } else if (rx_index == 3) {
        rx_buffer[rx_index++] = byte; // payload_len
        expected_length = sizeof(struct ProtocolHeader) + byte + 1; // +1 for checksum
        in_message = true;
      } else {
        rx_index = 0; // Reset on invalid sequence
      }
    } else {
      // Collecting message body
      rx_buffer[rx_index++] = byte;
      
      if (rx_index >= expected_length) {
        // Complete message received
        struct ProtocolMessage* msg = (struct ProtocolMessage*)rx_buffer;
        
        if (validate_message(msg)) {
          if (msg->header.payload_len > 0) {
            process_command(msg->payload[0], &msg->payload[1], msg->header.payload_len - 1);
          } else {
            send_error(ERROR_INVALID_PAYLOAD);
          }
        } else {
          send_error(ERROR_CHECKSUM);
        }
        
        rx_index = 0;
        in_message = false;
      } else if (rx_index >= sizeof(rx_buffer)) {
        // Buffer overflow protection
        send_error(ERROR_BUFFER_OVERFLOW);
        rx_index = 0;
        in_message = false;
      }
    }
  }
}

void setup() {
  Serial.begin(2000000);
  delay(1000);
  Wire.begin();
  if (!ina219.begin()) {
    Serial.println("ERROR: Failed to find INA219 sensor");
    while (1) { delay(10); }
  }
  
  as5600.begin();
  as5600.setAddress(0x36);
  as5600.resetPosition();

  ina219.setCalibration_32V_2A();

  Serial.print("AS5600 ADDR: ");
  Serial.println(as5600.getAddress());
  int b = as5600.isConnected();
  Serial.print("Connect: ");
  Serial.println(b);

  servo.attach(D0,544,2400);

  _time = micros();
}

void loop() {
  unsigned long delta = micros() - _time;
  _time += delta;

  // Check for incoming commands
  check_for_commands();

  SensorReading reading = getSampledReading(ina219);
  int32_t position = as5600.getCumulativePosition();
  int pwm = analogRead(A1);

  // Send binary sensor data
  send_sensor_data(
    delta,                    // delta_time_us
    pwm,                      // pwm_value
    reading.current_mA,       // current_mA
    reading.voltage_V,        // voltage_V
    position                  // position
  );
}
