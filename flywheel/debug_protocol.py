#!/usr/bin/env python3
"""
Debug script for flywheel binary protocol.
Helps diagnose communication issues by showing raw serial data.
"""

import serial
import time
import argparse
import sys


def hex_dump(data: bytes, prefix: str = "") -> None:
    """Display data as hex dump"""
    if not data:
        return
    
    hex_str = ' '.join(f'{b:02x}' for b in data)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
    print(f"{prefix}{hex_str} | {ascii_str}")


def monitor_serial(port: str, baudrate: int = 2000000, duration: float = 10.0):
    """Monitor raw serial data"""
    try:
        with serial.Serial(port, baudrate, timeout=0.1) as ser:
            print(f"Monitoring {port} at {baudrate} baud for {duration} seconds...")
            print("Raw data (hex | ascii):")
            print("-" * 60)
            
            start_time = time.time()
            while time.time() - start_time < duration:
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting)
                    timestamp = time.time() - start_time
                    hex_dump(data, f"{timestamp:6.2f}s: ")
                
                time.sleep(0.01)
            
            print("-" * 60)
            print("Monitoring complete")
            
    except Exception as e:
        print(f"Error: {e}")


def test_text_vs_binary(port: str, baudrate: int = 2000000):
    """Test if Arduino is sending text or binary data"""
    try:
        with serial.Serial(port, baudrate, timeout=1.0) as ser:
            print(f"Testing data format from {port}...")
            print("Collecting 5 seconds of data...")
            
            # Collect data for analysis
            all_data = bytearray()
            start_time = time.time()
            
            while time.time() - start_time < 5.0:
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting)
                    all_data.extend(data)
                time.sleep(0.01)
            
            if not all_data:
                print("No data received!")
                return
            
            print(f"Received {len(all_data)} bytes")
            
            # Check for text patterns
            text_chars = sum(1 for b in all_data if 32 <= b < 127 or b in [10, 13])
            text_ratio = text_chars / len(all_data)
            
            print(f"Text characters: {text_ratio:.1%}")
            
            # Look for binary protocol headers
            binary_headers = 0
            for i in range(len(all_data) - 1):
                if all_data[i] == 0xAA and all_data[i + 1] == 0x55:
                    binary_headers += 1
            
            print(f"Binary protocol headers found: {binary_headers}")
            
            # Show first 200 bytes
            sample = all_data[:200]
            print("\nFirst 200 bytes:")
            hex_dump(sample, "  ")
            
            # Analysis
            if text_ratio > 0.8:
                print("\n❌ Arduino appears to be sending TEXT data")
                print("   Need to flash binary protocol firmware")
            elif binary_headers > 0:
                print("\n✅ Arduino appears to be sending BINARY data") 
                print("   Protocol headers detected")
            else:
                print("\n❓ Unclear data format - may be corrupted or wrong baud rate")
            
    except Exception as e:
        print(f"Error: {e}")


def decode_message(data: bytes, offset: int = 0) -> tuple:
    """Manually decode a binary protocol message"""
    if len(data) < offset + 4:
        return None, "Too short for header"
    
    # Parse header
    header1 = data[offset]
    header2 = data[offset + 1] 
    msg_type = data[offset + 2]
    payload_len = data[offset + 3]
    
    if header1 != 0xAA or header2 != 0x55:
        return None, f"Invalid header: {header1:02x} {header2:02x}"
    
    total_len = 4 + payload_len + 1  # header + payload + checksum
    if len(data) < offset + total_len:
        return None, f"Incomplete message: need {total_len}, have {len(data) - offset}"
    
    # Extract payload and checksum
    payload = data[offset + 4:offset + 4 + payload_len]
    checksum = data[offset + 4 + payload_len]
    
    # Validate checksum
    message_for_checksum = data[offset:offset + 4 + payload_len]
    calculated_checksum = 0
    for b in message_for_checksum:
        calculated_checksum ^= b
    
    checksum_valid = (checksum == calculated_checksum)
    
    # Decode message type
    msg_types = {0x01: "DATA", 0x02: "COMMAND_ACK", 0x03: "ERROR"}
    msg_type_str = msg_types.get(msg_type, f"UNKNOWN({msg_type:02x})")
    
    # Decode payload based on message type
    payload_decoded = None
    if msg_type == 0x01 and payload_len == 18:  # Sensor data
        import struct
        try:
            sensor_data = struct.unpack('<IHffi', payload)
            payload_decoded = {
                "delta_time_us": sensor_data[0],
                "pwm_value": sensor_data[1], 
                "current_mA": sensor_data[2],
                "voltage_V": sensor_data[3],
                "position": sensor_data[4]
            }
        except:
            payload_decoded = f"Failed to decode sensor data: {payload.hex()}"
    elif msg_type == 0x02 and payload_len == 2:  # Command ACK
        payload_decoded = {
            "command_type": f"0x{payload[0]:02x}",
            "success": bool(payload[1])
        }
    elif msg_type == 0x03 and payload_len == 1:  # Error
        error_codes = {0x01: "CHECKSUM", 0x02: "INVALID_PAYLOAD", 0x03: "INVALID_COMMAND", 
                      0x04: "PARAMETER_OUT_OF_RANGE", 0x05: "BUFFER_OVERFLOW"}
        payload_decoded = {"error_code": error_codes.get(payload[0], f"UNKNOWN({payload[0]:02x})")}
    else:
        payload_decoded = f"Raw: {payload.hex()}"
    
    result = {
        "header": f"{header1:02x} {header2:02x}",
        "msg_type": msg_type_str,
        "payload_len": payload_len,
        "payload": payload_decoded,
        "checksum": f"{checksum:02x} (calculated: {calculated_checksum:02x})",
        "checksum_valid": checksum_valid,
        "total_length": total_len
    }
    
    return result, None

def decode_stream(data: bytes):
    """Decode all messages in a data stream"""
    offset = 0
    message_count = 0
    
    print(f"Decoding {len(data)} bytes of data...")
    print("=" * 60)
    
    while offset < len(data):
        # Look for sync bytes
        sync_found = False
        for i in range(offset, len(data) - 1):
            if data[i] == 0xAA and data[i + 1] == 0x55:
                if i > offset:
                    print(f"Skipped {i - offset} bytes before sync")
                offset = i
                sync_found = True
                break
        
        if not sync_found:
            print(f"No more sync bytes found, {len(data) - offset} bytes remaining")
            break
        
        result, error = decode_message(data, offset)
        if error:
            print(f"Offset {offset}: {error}")
            offset += 1  # Skip one byte and try again
            continue
        
        message_count += 1
        print(f"Message {message_count} at offset {offset}:")
        print(f"  Type: {result['msg_type']}")
        print(f"  Payload length: {result['payload_len']}")
        print(f"  Checksum: {result['checksum']} {'✓' if result['checksum_valid'] else '✗'}")
        
        if isinstance(result['payload'], dict):
            print(f"  Payload:")
            for key, value in result['payload'].items():
                print(f"    {key}: {value}")
        else:
            print(f"  Payload: {result['payload']}")
        
        print()
        offset += result['total_length']
    
    print(f"Decoded {message_count} messages")

def send_test_command(port: str, baudrate: int = 2000000):
    """Send a test command and monitor response"""
    try:
        with serial.Serial(port, baudrate, timeout=1.0) as ser:
            print(f"Sending test command to {port}...")
            
            # Send reset encoder command
            # Header: AA 55 01 01 (DATA message, 1 byte payload)  
            # Payload: 13 (CMD_RESET_ENCODER)
            # Checksum: XOR of all previous bytes
            header = bytes([0xAA, 0x55, 0x01, 0x01])
            payload = bytes([0x13])  # CMD_RESET_ENCODER
            message = header + payload
            
            # Calculate checksum
            checksum = 0
            for b in message:
                checksum ^= b
            
            full_message = message + bytes([checksum])
            
            print("Sending command:")
            hex_dump(full_message, "TX: ")
            
            ser.write(full_message)
            ser.flush()
            
            # Wait for response
            print("Waiting for response...")
            time.sleep(0.5)
            
            if ser.in_waiting:
                response = ser.read(ser.in_waiting)
                print("Received response:")
                hex_dump(response, "RX: ")
                print("\nDecoding response:")
                decode_stream(response)
            else:
                print("No response received")
                
    except Exception as e:
        print(f"Error: {e}")

def decode_hex_string(hex_string: str):
    """Decode a hex string containing protocol messages"""
    try:
        # Remove spaces and convert to bytes
        hex_clean = hex_string.replace(' ', '').replace('\n', '')
        data = bytes.fromhex(hex_clean)
        decode_stream(data)
    except ValueError as e:
        print(f"Invalid hex string: {e}")


def main():
    parser = argparse.ArgumentParser(description='Debug flywheel binary protocol')
    parser.add_argument('port', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baudrate', type=int, default=2000000, help='Baud rate')
    
    subparsers = parser.add_subparsers(dest='command', help='Debug commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor raw serial data')
    monitor_parser.add_argument('--duration', type=float, default=10.0, help='Duration in seconds')
    
    # Test format command  
    subparsers.add_parser('test-format', help='Test if data is text or binary')
    
    # Send command
    subparsers.add_parser('send-command', help='Send test command')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'monitor':
        monitor_serial(args.port, args.baudrate, args.duration)
    elif args.command == 'test-format':
        test_text_vs_binary(args.port, args.baudrate)
    elif args.command == 'send-command':
        send_test_command(args.port, args.baudrate)


if __name__ == "__main__":
    main()