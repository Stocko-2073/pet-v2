#!/usr/bin/env python3
"""
Binary serial communication protocol for flywheel Arduino controller.
Provides high-performance bidirectional communication with structured data.
"""

import serial
import struct
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

# Protocol constants
PROTOCOL_HEADER_1 = 0xAA
PROTOCOL_HEADER_2 = 0x55

class MessageType(IntEnum):
    DATA = 0x01
    COMMAND_ACK = 0x02
    ERROR = 0x03

class CommandType(IntEnum):
    SET_SERVO_POSITION = 0x10
    SET_POSITION_TARGET = 0x11
    SET_CONTROL_MODE = 0x12
    RESET_ENCODER = 0x13
    SET_SAMPLE_RATE = 0x14

class ControlMode(IntEnum):
    MANUAL = 0
    POSITION = 1

class ErrorCode(IntEnum):
    CHECKSUM = 0x01
    INVALID_PAYLOAD = 0x02
    INVALID_COMMAND = 0x03
    PARAMETER_OUT_OF_RANGE = 0x04
    BUFFER_OVERFLOW = 0x05

ERROR_MESSAGES = {
    ErrorCode.CHECKSUM: "Checksum validation failed",
    ErrorCode.INVALID_PAYLOAD: "Invalid payload length or format",
    ErrorCode.INVALID_COMMAND: "Unknown command type",
    ErrorCode.PARAMETER_OUT_OF_RANGE: "Parameter value out of valid range",
    ErrorCode.BUFFER_OVERFLOW: "Receive buffer overflow"
}

@dataclass
class SensorData:
    """Sensor data received from Arduino"""
    delta_time_us: int
    pwm_value: int
    current_mA: float
    voltage_V: float
    position: int
    servo_position_deg: float
    bound_top: int
    bound_bottom: int
    timestamp: float  # Host-side timestamp

@dataclass 
class CommandAck:
    """Command acknowledgment from Arduino"""
    command_type: int
    success: bool

class FlywheelComm:
    """High-performance binary communication with flywheel Arduino controller"""
    
    def __init__(self, port: str, baudrate: int = 2000000, timeout: float = 0.1):
        """
        Initialize communication with Arduino.
        
        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0' or 'COM3')
            baudrate: Communication speed (default: 2MHz)
            timeout: Read timeout in seconds
        """
        self.serial = serial.Serial(port, baudrate, timeout=timeout)
        self.rx_buffer = bytearray()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """Close serial connection"""
        if self.serial.is_open:
            self.serial.close()
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate XOR checksum"""
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum
    
    def _send_message(self, msg_type: int, payload: bytes = b'') -> None:
        """Send binary message to Arduino"""
        header = struct.pack('<BBBB', 
                           PROTOCOL_HEADER_1, 
                           PROTOCOL_HEADER_2, 
                           msg_type, 
                           len(payload))
        
        message = header + payload
        checksum = self._calculate_checksum(message)
        
        full_message = message + struct.pack('<B', checksum)
        self.serial.write(full_message)
        self.serial.flush()
    
    def _read_message(self) -> Optional[Tuple[int, bytes]]:
        """Read and parse binary message from Arduino"""
        # Read available data into buffer
        if self.serial.in_waiting:
            self.rx_buffer.extend(self.serial.read(self.serial.in_waiting))
        
        # Look for complete message
        while len(self.rx_buffer) >= 4:  # Minimum header size
            # Find sync bytes
            sync_pos = -1
            for i in range(len(self.rx_buffer) - 1):
                if (self.rx_buffer[i] == PROTOCOL_HEADER_1 and 
                    self.rx_buffer[i + 1] == PROTOCOL_HEADER_2):
                    sync_pos = i
                    break
            
            if sync_pos == -1:
                # No sync found, clear buffer
                self.rx_buffer.clear()
                return None
                
            # Remove data before sync
            if sync_pos > 0:
                del self.rx_buffer[:sync_pos]
                
            # Check if we have complete header
            if len(self.rx_buffer) < 4:
                return None
                
            # Parse header
            msg_type = self.rx_buffer[2]
            payload_len = self.rx_buffer[3]
            total_len = 4 + payload_len + 1  # header + payload + checksum
            
            # Check if we have complete message
            if len(self.rx_buffer) < total_len:
                return None
                
            # Extract complete message
            message_data = bytes(self.rx_buffer[:total_len-1])  # Without checksum
            received_checksum = self.rx_buffer[total_len-1]
            
            # Validate checksum
            calculated_checksum = self._calculate_checksum(message_data)
            if received_checksum != calculated_checksum:
                # Bad checksum, remove this message and continue
                del self.rx_buffer[:total_len]
                continue
                
            # Extract payload
            payload = bytes(self.rx_buffer[4:4+payload_len])
            
            # Remove processed message from buffer
            del self.rx_buffer[:total_len]
            
            return msg_type, payload
            
        return None
    
    def read_sensor_data(self) -> Optional[SensorData]:
        """Read sensor data from Arduino"""
        msg = self._read_message()
        if msg is None or msg[0] != MessageType.DATA:
            return None
            
        msg_type, payload = msg
        if len(payload) != 24:  # Expected sensor data size
            return None
            
        # Unpack sensor data (little-endian)
        data = struct.unpack('<IHffiHhh', payload)
        
        return SensorData(
            delta_time_us=data[0],
            pwm_value=data[1],
            current_mA=data[2],
            voltage_V=data[3],
            position=data[4],
            servo_position_deg=data[5] / 10.0,  # Convert from degrees*10
            bound_top=data[6],
            bound_bottom=data[7],
            timestamp=time.time()
        )
    
    def read_command_ack(self) -> Optional[CommandAck]:
        """Read command acknowledgment from Arduino"""
        msg = self._read_message()
        if msg is None or msg[0] != MessageType.COMMAND_ACK:
            return None
            
        msg_type, payload = msg
        if len(payload) != 2:
            return None
            
        return CommandAck(
            command_type=payload[0],
            success=bool(payload[1])
        )
    
    def read_error(self) -> Optional[int]:
        """Read error code from Arduino"""
        msg = self._read_message()
        if msg is None or msg[0] != MessageType.ERROR:
            return None
            
        msg_type, payload = msg
        if len(payload) != 1:
            return None
            
        return payload[0]
    
    def set_servo_position(self, position_deg: float) -> bool:
        """
        Set servo position in degrees (0-180)
        
        Args:
            position_deg: Target position in degrees
            
        Returns:
            True if command was acknowledged successfully
        """
        if not (0 <= position_deg <= 180):
            return False
            
        payload = struct.pack('<H', int(position_deg * 10))
        cmd_payload = struct.pack('<B', CommandType.SET_SERVO_POSITION) + payload
        self._send_message(MessageType.DATA, cmd_payload)
        
        # Wait for acknowledgment
        start_time = time.time()
        while time.time() - start_time < 1.0:  # 1 second timeout
            ack = self.read_command_ack()
            if ack and ack.command_type == CommandType.SET_SERVO_POSITION:
                return ack.success
        return False
    
    def set_position_target(self, target_position: int) -> bool:
        """
        Set target position for position control mode
        
        Args:
            target_position: Target encoder position
            
        Returns:
            True if command was acknowledged successfully
        """
        payload = struct.pack('<i', target_position)
        cmd_payload = struct.pack('<B', CommandType.SET_POSITION_TARGET) + payload
        self._send_message(MessageType.DATA, cmd_payload)
        
        start_time = time.time()
        while time.time() - start_time < 1.0:
            ack = self.read_command_ack()
            if ack and ack.command_type == CommandType.SET_POSITION_TARGET:
                return ack.success
        return False
    
    def set_control_mode(self, mode: ControlMode) -> bool:
        """
        Set control mode (manual or position control)
        
        Args:
            mode: Control mode (MANUAL or POSITION)
            
        Returns:
            True if command was acknowledged successfully
        """
        payload = struct.pack('<B', int(mode))
        cmd_payload = struct.pack('<B', CommandType.SET_CONTROL_MODE) + payload
        self._send_message(MessageType.DATA, cmd_payload)
        
        start_time = time.time()
        while time.time() - start_time < 1.0:
            ack = self.read_command_ack()
            if ack and ack.command_type == CommandType.SET_CONTROL_MODE:
                return ack.success
        return False
    
    def reset_encoder(self) -> bool:
        """
        Reset encoder position to zero
        
        Returns:
            True if command was acknowledged successfully
        """
        cmd_payload = struct.pack('<B', CommandType.RESET_ENCODER)
        self._send_message(MessageType.DATA, cmd_payload)
        
        start_time = time.time()
        while time.time() - start_time < 1.0:
            ack = self.read_command_ack()
            if ack and ack.command_type == CommandType.RESET_ENCODER:
                return ack.success
        return False
    
    def set_sample_rate(self, samples_per_sec: int) -> bool:
        """
        Set sensor sampling rate
        
        Args:
            samples_per_sec: Sampling rate in Hz (1-1000)
            
        Returns:
            True if command was acknowledged successfully
        """
        if not (1 <= samples_per_sec <= 1000):
            return False
            
        payload = struct.pack('<H', samples_per_sec)
        cmd_payload = struct.pack('<B', CommandType.SET_SAMPLE_RATE) + payload
        self._send_message(MessageType.DATA, cmd_payload)
        
        start_time = time.time()
        while time.time() - start_time < 1.0:
            ack = self.read_command_ack()
            if ack and ack.command_type == CommandType.SET_SAMPLE_RATE:
                return ack.success
        return False


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Flywheel communication test')
    parser.add_argument('port', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baudrate', type=int, default=2000000, help='Baud rate')
    args = parser.parse_args()
    
    print(f"Connecting to {args.port} at {args.baudrate} baud...")
    
    try:
        with FlywheelComm(args.port, args.baudrate) as comm:
            print("Connected! Reading sensor data...")
            
            # Test commands
            print("Testing commands...")
            
            print("Setting control mode to manual...")
            if comm.set_control_mode(ControlMode.MANUAL):
                print("✓ Control mode set successfully")
            else:
                print("✗ Failed to set control mode")
            
            print("Setting servo position to 90 degrees...")
            if comm.set_servo_position(90.0):
                print("✓ Servo position set successfully")
            else:
                print("✗ Failed to set servo position")
            
            print("Resetting encoder...")
            if comm.reset_encoder():
                print("✓ Encoder reset successfully")
            else:
                print("✗ Failed to reset encoder")
            
            # Read sensor data
            print("\nReading sensor data for 10 seconds...")
            start_time = time.time()
            count = 0
            
            while time.time() - start_time < 10.0:
                data = comm.read_sensor_data()
                if data:
                    count += 1
                    if count % 10 == 0:  # Print every 10th reading
                        print(f"Position: {data.position:6d}, "
                              f"Current: {data.current_mA:6.1f}mA, "
                              f"Voltage: {data.voltage_V:5.2f}V, "
                              f"Servo: {data.servo_position_deg:5.1f}°")
                
                # Check for errors
                error = comm.read_error()
                if error is not None:
                    error_msg = ERROR_MESSAGES.get(error, f"Unknown error code: {error}")
                    print(f"Error received: {error_msg}")
                
                time.sleep(0.01)  # Small delay
            
            print(f"\nReceived {count} sensor readings in 10 seconds")
            print(f"Average rate: {count/10.0:.1f} Hz")
            
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")