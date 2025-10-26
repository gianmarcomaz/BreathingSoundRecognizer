#!/usr/bin/env python3
"""
Test the medical monitoring system with generated audio
Simulates real child breathing and crying sounds
"""

import numpy as np
import requests
import json
import time
from datetime import datetime

def create_breathing_audio(breath_rate=40, duration=5):
    """Generate realistic breathing audio"""
    sample_rate = 44100
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Create breathing pattern
    freq = breath_rate / 60  # Convert bpm to Hz
    breathing = 0.4 * np.sin(2 * np.pi * freq * t)
    
    # Add some natural variation
    variation = 0.1 * np.sin(2 * np.pi * 0.1 * t)
    breathing += variation
    
    # Add mild noise
    noise = np.random.normal(0, 0.05, samples)
    audio = breathing + noise
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.tolist()

def create_crying_audio(duration=3):
    """Generate realistic crying audio"""
    sample_rate = 44100
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Fundamental frequency (around 400 Hz)
    freq1 = 400 + 50 * np.sin(2 * np.pi * 2 * t)  # Varying pitch
    cry = 0.3 * np.sin(2 * np.pi * freq1 * t)
    
    # Add harmonics
    cry += 0.2 * np.sin(2 * np.pi * freq1 * 2 * t)
    cry += 0.1 * np.sin(2 * np.pi * freq1 * 3 * t)
    
    # Modulate amplitude (cry isn't constant)
    envelope = np.sin(2 * np.pi * 0.5 * t) ** 2
    cry *= envelope
    
    # Add noise
    noise = np.random.normal(0, 0.03, samples)
    cry += noise
    
    # Normalize
    if np.max(np.abs(cry)) > 0:
        cry = cry / np.max(np.abs(cry)) * 0.7
    
    return cry.tolist()

def test_with_api(audio_data, label):
    """Send audio to API and print results"""
    print(f"\nTesting: {label}")
    print(f"Audio length: {len(audio_data)} samples ({len(audio_data)/44100:.2f} seconds)")
    
    try:
        response = requests.post(
            'http://localhost:5000/analyze_audio',
            json={
                'audio_data': audio_data,
                'sample_rate': 44100,
                'real_time': True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  SUCCESS: Analysis completed")
            print(f"  - Breathing Rate: {data['breathing_rate']:.1f} bpm")
            print(f"  - Pattern: {data['breathing_pattern']}")
            print(f"  - Alert Level: {data['alert_level']}")
            print(f"  - Signal Quality: {data['signal_quality']}")
            print(f"  - Cry Quality: {data['cry_quality']}")
            print(f"  - VAD Activity: {data.get('vad_activity', 'N/A')}")
            if 'error' not in data:
                return True
            else:
                print(f"  - ERROR in analysis: {data['error']}")
                return False
        else:
            print(f"  ERROR: HTTP {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  ERROR: Cannot connect to server")
        print("  Make sure app.py is running first!")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    print("="*70)
    print("  Live Testing with Generated Audio")
    print("  Simulates real child breathing and crying")
    print("="*70)
    
    print("\nPlease make sure app.py is running first!")
    print("Run: python app.py")
    time.sleep(2)
    
    results = []
    
    # Test 1: Healthy breathing (normal newborn)
    print("\n" + "="*70)
    audio = create_breathing_audio(breath_rate=45, duration=5)
    results.append(("Healthy Breathing (45 bpm)", test_with_api(audio, "Healthy Breathing (45 bpm)")))
    
    # Test 2: Rapid breathing (respiratory distress)
    audio = create_breathing_audio(breath_rate=80, duration=5)
    results.append(("Rapid Breathing (80 bpm)", test_with_api(audio, "Rapid Breathing (80 bpm)")))
    
    # Test 3: Slow breathing (bradycardia/issue)
    audio = create_breathing_audio(breath_rate=25, duration=5)
    results.append(("Slow Breathing (25 bpm)", test_with_api(audio, "Slow Breathing (25 bpm)")))
    
    # Test 4: Very rapid (severe distress)
    audio = create_breathing_audio(breath_rate=100, duration=5)
    results.append(("Very Rapid (100 bpm)", test_with_api(audio, "Very Rapid Breathing (100 bpm)")))
    
    # Test 5: Crying audio
    cry_audio = create_crying_audio(duration=3)
    results.append(("Crying Audio", test_with_api(cry_audio, "Crying Audio (3 seconds)")))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All audio tests passed!")
        print("The system is working correctly with generated audio.")
    else:
        print(f"\nWARNING: {total - passed} test(s) failed")
        print("Check if app.py is running and try again.")

if __name__ == "__main__":
    main()

