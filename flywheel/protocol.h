#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

// Protocol constants
#define PROTOCOL_HEADER_1 0xAA
#define PROTOCOL_HEADER_2 0x55
#define PROTOCOL_MAX_PAYLOAD 64

// Message types
#define MSG_TYPE_DATA 0x01
#define MSG_TYPE_COMMAND_ACK 0x02
#define MSG_TYPE_ERROR 0x03

// Command types
#define CMD_SET_SERVO_POSITION 0x10
#define CMD_RESET_ENCODER 0x13
#define CMD_SET_SAMPLE_RATE 0x14
#define CMD_SET_SERVO_ENABLE 0x15

// Message header structure
struct ProtocolHeader {
  uint8_t header1;      // 0xAA
  uint8_t header2;      // 0x55
  uint8_t msg_type;     // Message type
  uint8_t payload_len;  // Payload length in bytes
} __attribute__((packed));

// Sensor data payload (23 bytes)
struct SensorDataPayload {
  uint64_t time_us;           // Time since boot in microseconds (8 bytes)
  uint16_t pwm_value;         // PWM reading (2 bytes)
  float current_mA;           // Current in milliamps (4 bytes)
  float voltage_V;            // Voltage in volts (4 bytes)
  float position;             // Encoder position in degrees (4 bytes)
  uint8_t servo_enabled;      // Servo enable state: 1 = enabled, 0 = disabled (1 byte)
} __attribute__((packed));

// Command payloads
struct SetServoPositionPayload {
  uint16_t position_deg_x10;  // Position in degrees * 10
} __attribute__((packed));

struct SetSampleRatePayload {
  uint16_t samples_per_sec;  // Sampling rate
} __attribute__((packed));

struct SetServoEnablePayload {
  uint8_t enabled;  // 1 = enabled, 0 = disabled
} __attribute__((packed));

// Complete message structure
struct ProtocolMessage {
  struct ProtocolHeader header;
  uint8_t payload[PROTOCOL_MAX_PAYLOAD];
  uint8_t checksum;
} __attribute__((packed));

// Error codes
#define ERROR_CHECKSUM 0x01
#define ERROR_INVALID_PAYLOAD 0x02
#define ERROR_INVALID_COMMAND 0x03
#define ERROR_PARAMETER_OUT_OF_RANGE 0x04
#define ERROR_BUFFER_OVERFLOW 0x05

// Helper functions
inline uint8_t calculate_checksum(const uint8_t* data, uint8_t len) {
  uint8_t checksum = 0;
  for (uint8_t i = 0; i < len; i++) {
    checksum ^= data[i];
  }
  return checksum;
}

inline bool validate_message(const struct ProtocolMessage* msg) {
  if (msg->header.header1 != PROTOCOL_HEADER_1 || 
      msg->header.header2 != PROTOCOL_HEADER_2) {
    return false;
  }
  
  // Calculate checksum on header + payload (excluding checksum byte)
  uint8_t expected_checksum = calculate_checksum(
    (const uint8_t*)msg, 
    sizeof(struct ProtocolHeader) + msg->header.payload_len
  );
  
  // Get checksum from correct position (immediately after payload)
  uint8_t received_checksum = *((uint8_t*)msg + sizeof(struct ProtocolHeader) + msg->header.payload_len);
  
  return received_checksum == expected_checksum;
}

#endif // PROTOCOL_H