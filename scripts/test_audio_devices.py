#!/usr/bin/env python3
"""
Test script to list available audio devices and test PyAudio functionality.
"""

import pyaudio
import sys

def list_audio_devices():
    """List all available audio devices."""
    p = pyaudio.PyAudio()
    
    print("Available audio devices:")
    print("=" * 60)
    
    device_count = p.get_device_count()
    print(f"Total devices: {device_count}")
    
    if device_count == 0:
        print("No audio devices found!")
        return
    
    for i in range(device_count):
        try:
            info = p.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']}")
            print(f"  Channels: {info['maxInputChannels']} input, {info['maxOutputChannels']} output")
            print(f"  Sample Rate: {info['defaultSampleRate']}")
            print(f"  Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
            print()
        except Exception as e:
            print(f"Device {i}: Error - {e}")
    
    # Check default devices
    try:
        default_input = p.get_default_input_device_info()
        print(f"Default input device: {default_input['name']}")
    except Exception as e:
        print(f"No default input device: {e}")
    
    try:
        default_output = p.get_default_output_device_info()
        print(f"Default output device: {default_output['name']}")
    except Exception as e:
        print(f"No default output device: {e}")
    
    p.terminate()

def test_audio_capture():
    """Test basic audio capture."""
    p = pyaudio.PyAudio()
    
    try:
        # Try to open a basic audio stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
        )
        
        print("Audio capture test successful!")
        
        # Read a few frames
        for i in range(5):
            data = stream.read(1024)
            print(f"Read frame {i+1}: {len(data)} bytes")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"Audio capture test failed: {e}")
    
    p.terminate()

if __name__ == "__main__":
    print("Testing audio devices and PyAudio functionality...")
    print()
    
    list_audio_devices()
    print()
    test_audio_capture()