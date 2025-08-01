#include <Wire.h>
#include <Adafruit_INA219.h>
#include <Servo.h>
#include "AS5600.h"
#include "protocol.h"

Adafruit_INA219 ina219(0x41);
AS5600L as5600;
Servo servo;
uint64_t last_time_us;
uint64_t accumulator;

// Protocol variables
uint16_t sample_rate = 100; // Hz
uint64_t sample_interval;

void send_binary_message(uint8_t msg_type, const void *payload, uint8_t payload_len) {
  struct ProtocolHeader header;
  header.header1 = PROTOCOL_HEADER_1;
  header.header2 = PROTOCOL_HEADER_2;
  header.msg_type = msg_type;
  header.payload_len = payload_len;

  uint8_t message_buffer[sizeof(struct ProtocolHeader) + PROTOCOL_MAX_PAYLOAD];
  memcpy(message_buffer, &header, sizeof(struct ProtocolHeader));
  if (payload && payload_len > 0) {
    memcpy(message_buffer + sizeof(struct ProtocolHeader), payload, payload_len);
  }
  uint8_t checksum = calculate_checksum(message_buffer, sizeof(struct ProtocolHeader) + payload_len);
  Serial.write((const uint8_t *) &header, sizeof(struct ProtocolHeader));
  if (payload && payload_len > 0) {
    Serial.write((const uint8_t *) payload, payload_len);
  }
  Serial.write(&checksum, 1);
}

void send_sensor_data(uint32_t time_ms, uint16_t pwm, float current, float voltage, int32_t position) {
  struct SensorDataPayload payload;
  payload.time_ms = time_ms;
  payload.pwm_value = pwm;
  payload.current_mA = current;
  payload.voltage_V = voltage;
  payload.position = position;

  send_binary_message(MSG_TYPE_DATA, &payload, sizeof(payload));
}

void send_command_ack(uint8_t command_type, bool success) {
  uint8_t payload[2] = {command_type, success ? (uint8_t)1 : (uint8_t)0};
  send_binary_message(MSG_TYPE_COMMAND_ACK, payload, 2);
}

void send_error(uint8_t error_code) {
  send_binary_message(MSG_TYPE_ERROR, &error_code, 1);
}

void process_command(uint8_t cmd_type, const uint8_t *payload, uint8_t payload_len) {
  bool success = true;

  switch (cmd_type) {
    case CMD_SET_SERVO_POSITION: {
      if (payload_len == sizeof(struct SetServoPositionPayload)) {
        struct SetServoPositionPayload *cmd = (struct SetServoPositionPayload *) payload;
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
        struct SetSampleRatePayload *cmd = (struct SetSampleRatePayload *) payload;
        if (cmd->samples_per_sec > 0 && cmd->samples_per_sec <= 1000) {
          sample_rate = cmd->samples_per_sec;
          sample_interval = 1000000 / sample_rate;
        } else {
          success = false;
        }
      } else {
        success = false;
      }
      break;
    }

    default:success = false;
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
        struct ProtocolMessage *msg = (struct ProtocolMessage *) rx_buffer;

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
  Serial.begin(1000000);
  delay(1000);
  Wire.begin();
  if (!ina219.begin()) {
    // TODO: Implement an error report packet
    // INA219 sensor not found - halt execution
    while (1) { delay(10); }
  }

  as5600.begin();
  as5600.setAddress(0x36);
  as5600.resetPosition();

  ina219.setCalibration_32V_2A();

  // TODO: Implement and send an error report packet if this fails
  as5600.isConnected(); // Still check connection but don't print

  servo.attach(D0, 544, 2400);
  last_time_us = micros();
  accumulator = 0;
  sample_interval = 1000000 / sample_rate;
}

void loop() {
  uint64_t time_us = micros();
  uint32_t time_ms = (uint32_t)(time_us / 1000);
  accumulator += (time_us - last_time_us);
  last_time_us = time_us;

  if (accumulator >= sample_interval) {
    accumulator -= sample_interval;

    // Read sensors
    int pwm = analogRead(A1);
    float current_mA = ina219.getCurrent_mA();
    float voltage_V = ina219.getBusVoltage_V();
    int32_t position = as5600.getCumulativePosition();

    send_sensor_data(
        time_ms,
        pwm,
        current_mA,
        voltage_V,
        position
    );
  }

  check_for_commands();
}
