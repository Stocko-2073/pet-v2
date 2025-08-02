#!/usr/bin/env python3
"""
Host-side test script for flywheel binary protocol.
Implements the 500ms servo cycling (0° to 180°) that was removed from Arduino code,
while collecting and displaying real-time statistics from sensor data.
"""

import time
import threading
import signal
import sys
import argparse
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import statistics

from host_comm import FlywheelComm, SensorData, ErrorCode, ERROR_MESSAGES


@dataclass
class Statistics:
    """Container for tracking sensor statistics"""
    # Data collection
    message_count: int = 0
    error_count: int = 0
    start_time: float = field(default_factory=time.time)
    
    # Current measurements (mA)
    current_readings: deque = field(default_factory=lambda: deque(maxlen=1000))
    current_min: float = float('inf')
    current_max: float = float('-inf')
    current_sum: float = 0.0
    
    # Voltage measurements (V)
    voltage_readings: deque = field(default_factory=lambda: deque(maxlen=1000))
    voltage_min: float = float('inf')
    voltage_max: float = float('-inf')
    voltage_sum: float = 0.0
    
    # Position tracking
    position_readings: deque = field(default_factory=lambda: deque(maxlen=100))
    last_position: Optional[int] = None
    position_changes: int = 0

    # Timing
    delta_times: deque = field(default_factory=lambda: deque(maxlen=100))
    host_process_times: deque = field(default_factory=lambda: deque(maxlen=100))

    # Protocol stats
    last_update_time: float = field(default_factory=time.time)
    messages_per_second: float = 0.0
    
    # Packet loss stats
    servo_commands_sent: int = 0
    servo_acks_received: int = 0
    servo_timeouts: int = 0


class FlywheelHostTest:
    """Host-side test controller for flywheel system"""
    
    def __init__(self, port: str, baudrate: int = 2000000):
        self.port = port
        self.baudrate = baudrate
        self.comm: Optional[FlywheelComm] = None
        self.running = False
        self.stats = Statistics()
        
        # Servo control state
        self.servo_position = 0  # degrees
        self.servo_target = 0
        self.last_servo_change = time.time()
        self.servo_cycle_period = 0.5  # 500ms as per original Arduino code
        
        # Threading
        self.communication_thread: Optional[threading.Thread] = None
        self.display_thread: Optional[threading.Thread] = None
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)

        self.last_time_us = 0
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop()
    
    def connect(self) -> bool:
        """Connect to the Arduino"""
        try:
            self.comm = FlywheelComm(self.port, self.baudrate, debug=False)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            
            # Test connection by trying to read sensor data
            print("Testing connection by reading sensor data...")
            
            # Wait a moment for Arduino to send data
            time.sleep(0.5)
            
            # Try to read sensor data
            for attempt in range(10):  # Try up to 10 times
                data = self.comm.read_sensor_data()
                if data:
                    print(f"✓ Connection test successful - received sensor data")
                    print(f"  Sample data: Current={data.current_mA:.1f}mA, Voltage={data.voltage_V:.2f}V, Position={data.position}")
                    return True
                
                # Check for errors
                error = self.comm.read_error()
                if error is not None:
                    error_msg = ERROR_MESSAGES.get(error, f"Unknown error: {error}")
                    print(f"Protocol error during connection test: {error_msg}")

            print("✗ Connection test failed - no sensor data received")
            return False
                
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.comm:
            self.comm.close()
            self.comm = None
    
    def start(self):
        """Start the test - servo control and data collection"""
        if not self.comm:
            print("Not connected to Arduino")
            return False
        
        print("Starting flywheel test...")
        print("Press Ctrl+C to stop")
        print()
        
        self.running = True
        self.stats.start_time = time.time()
        
        # Start threads
        self.communication_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        
        self.communication_thread.start()
        self.display_thread.start()
        
        return True
    
    def stop(self):
        """Stop all threads and disconnect"""
        self.running = False
        
        if self.communication_thread:
            self.communication_thread.join(timeout=1.0)
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        self.disconnect()
        print("Test stopped")
    
    def wait_for_completion(self):
        """Wait for threads to complete (or Ctrl+C)"""
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _communication_loop(self):
        """Thread function: Unified serial communication - handles both servo control and data collection"""
        while self.running:
            if not self.comm:
                break
            
            current_time = time.time()
            start_time = time.perf_counter()
            
            # Check if it's time to change servo position
            if current_time - self.last_servo_change >= self.servo_cycle_period:
                # Toggle between 0 and 180 degrees
                self.servo_target = 180 if self.servo_position == 0 else 0
                
                # Send servo command and track statistics
                self.stats.servo_commands_sent += 1
                try:
                    if self.comm.set_servo_position(self.servo_target):
                        self.stats.servo_acks_received += 1
                        self.servo_position = self.servo_target
                        self.last_servo_change = current_time
                    else:
                        self.stats.servo_timeouts += 1
                except Exception:
                    self.stats.servo_timeouts += 1
                    # Suppress timeout exception messages to avoid spam
            
            # Read sensor data
            data = self.comm.read_sensor_data()
            if data:
                self._process_sensor_data(data)

                # Track processing time
                process_time = time.perf_counter() - start_time
                self.stats.host_process_times.append(process_time * 1000)  # Convert to ms

            # Check for errors
            error = self.comm.read_error()
            if error is not None:
                self.stats.error_count += 1
                error_msg = ERROR_MESSAGES.get(error, f"Unknown error: {error}")
                print(f"Protocol error: {error_msg}")

            # Small sleep to prevent overwhelming the system
            time.sleep(0.001)  # 1ms

    def _process_sensor_data(self, data: SensorData):
        """Process incoming sensor data and update statistics"""
        self.stats.message_count += 1
        
        # Current statistics
        current = data.current_mA
        self.stats.current_readings.append(current)
        self.stats.current_sum += current
        self.stats.current_min = min(self.stats.current_min, current)
        self.stats.current_max = max(self.stats.current_max, current)
        
        # Voltage statistics
        voltage = data.voltage_V
        self.stats.voltage_readings.append(voltage)
        self.stats.voltage_sum += voltage
        self.stats.voltage_min = min(self.stats.voltage_min, voltage)
        self.stats.voltage_max = max(self.stats.voltage_max, voltage)
        
        # Position tracking
        if self.stats.last_position is not None:
            if data.position != self.stats.last_position:
                self.stats.position_changes += 1
        self.stats.last_position = data.position
        self.stats.position_readings.append(data.position)

        # Delta time tracking
        delta_us = data.time_us - self.last_time_us
        self.last_time_us = data.time_us
        self.stats.delta_times.append(delta_us)

    def _calculate_message_rate(self):
        """Calculate messages per second"""
        current_time = time.time()
        time_elapsed = current_time - self.stats.last_update_time
        
        if time_elapsed >= 1.0:  # Update every second
            self.stats.messages_per_second = self.stats.message_count / (current_time - self.stats.start_time)
            self.stats.last_update_time = current_time
    
    def _display_loop(self):
        """Thread function: Display real-time statistics"""
        while self.running:
            self._display_stats()
            time.sleep(1.0)  # Update display every second
    
    def _display_stats(self):
        """Display current statistics"""
        self._calculate_message_rate()
        
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")
        
        # Header
        print("=" * 80)
        print("FLYWHEEL HOST TEST - Real-time Statistics")
        print("=" * 80)
        
        # Test duration and data rate
        duration = time.time() - self.stats.start_time
        print(f"Duration: {duration:.1f}s | Messages: {self.stats.message_count:,} | Rate: {self.stats.messages_per_second:.1f} msg/s")
        print(f"Errors: {self.stats.error_count} | Servo Position: {self.servo_position}° → {self.servo_target}°")
        print()
        
        # Current statistics
        if self.stats.current_readings:
            current_avg = self.stats.current_sum / len(self.stats.current_readings)
            current_recent = list(self.stats.current_readings)[-10:]  # Last 10 readings
            current_recent_avg = statistics.mean(current_recent) if current_recent else 0
            
            print("CURRENT CONSUMPTION:")
            print(f"  Current:    {current_recent_avg:8.1f} mA  (live)")
            print(f"  Average:    {current_avg:8.1f} mA  (overall)")
            print(f"  Range:      {self.stats.current_min:8.1f} - {self.stats.current_max:8.1f} mA")
            print()
        
        # Voltage statistics
        if self.stats.voltage_readings:
            voltage_avg = self.stats.voltage_sum / len(self.stats.voltage_readings)
            voltage_recent = list(self.stats.voltage_readings)[-10:]
            voltage_recent_avg = statistics.mean(voltage_recent) if voltage_recent else 0
            
            print("VOLTAGE MONITORING:")
            print(f"  Voltage:    {voltage_recent_avg:8.2f} V   (live)")
            print(f"  Average:    {voltage_avg:8.2f} V   (overall)")
            print(f"  Range:      {self.stats.voltage_min:8.2f} - {self.stats.voltage_max:8.2f} V")
            print()
        
        # Position and movement
        if self.stats.position_readings:
            current_pos = self.stats.position_readings[-1]
            position_range = max(self.stats.position_readings) - min(self.stats.position_readings)
            
            print("ENCODER POSITION:")
            print(f"  Position:   {current_pos:8d} counts")
            print(f"  Changes:    {self.stats.position_changes:8d} times")
            print(f"  Range:      {position_range:8d} counts")
            print()

        # Timing statistics
        if self.stats.delta_times and self.stats.host_process_times:
            arduino_avg_us = statistics.mean(self.stats.delta_times)
            arduino_avg_ms = arduino_avg_us / 1000  # Convert to ms for display
            host_avg = statistics.mean(self.stats.host_process_times)

            print("TIMING PERFORMANCE:")
            print(f"  Arduino Δt: {arduino_avg_ms:8.2f} ms   (average)")
            print(f"  Host proc:  {host_avg:8.3f} ms   (average)")
            print(f"  Data rate:  {1000/arduino_avg_ms if arduino_avg_ms > 0 else 0:8.1f} Hz   (theoretical)")
            print()

        # Packet loss statistics
        if self.stats.servo_commands_sent > 0:
            success_rate = (self.stats.servo_acks_received / self.stats.servo_commands_sent) * 100
            loss_rate = (self.stats.servo_timeouts / self.stats.servo_commands_sent) * 100
            
            print("SERVO COMMAND RELIABILITY:")
            print(f"  Commands:   {self.stats.servo_commands_sent:8d} sent")
            print(f"  ACKs:       {self.stats.servo_acks_received:8d} received")
            print(f"  Timeouts:   {self.stats.servo_timeouts:8d} lost")
            print(f"  Success:    {success_rate:8.1f} %")
            print(f"  Loss:       {loss_rate:8.1f} %")
            print()
        
        print("Press Ctrl+C to stop")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Flywheel host test with servo control and statistics')
    parser.add_argument('port', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baudrate', type=int, default=2000000, help='Baud rate (default: 2000000)')
    parser.add_argument('--duration', type=float, help='Test duration in seconds (default: run until Ctrl+C)')
    
    args = parser.parse_args()
    
    # Create test instance
    test = FlywheelHostTest(args.port, args.baudrate)
    
    # Connect to Arduino
    if not test.connect():
        sys.exit(1)
    
    # Start the test
    if not test.start():
        sys.exit(1)
    
    # Run for specified duration or until interrupted
    try:
        if args.duration:
            print(f"Running test for {args.duration} seconds...")
            time.sleep(args.duration)
            test.stop()
        else:
            test.wait_for_completion()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        test.stop()
    
    # Final statistics
    print("\nFinal Statistics:")
    print(f"  Total messages: {test.stats.message_count:,}")
    print(f"  Total errors: {test.stats.error_count}")
    print(f"  Average rate: {test.stats.messages_per_second:.1f} msg/s")
    if test.stats.current_readings:
        avg_current = test.stats.current_sum / len(test.stats.current_readings)
        print(f"  Average current: {avg_current:.1f} mA")
    if test.stats.voltage_readings:
        avg_voltage = test.stats.voltage_sum / len(test.stats.voltage_readings)
        print(f"  Average voltage: {avg_voltage:.2f} V")
    if test.stats.servo_commands_sent > 0:
        success_rate = (test.stats.servo_acks_received / test.stats.servo_commands_sent) * 100
        print(f"  Servo commands: {test.stats.servo_commands_sent} sent, {test.stats.servo_acks_received} ACKs ({success_rate:.1f}% success)")
        print(f"  Servo timeouts: {test.stats.servo_timeouts}")


if __name__ == "__main__":
    main()