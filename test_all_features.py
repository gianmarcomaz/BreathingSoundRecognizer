#!/usr/bin/env python3
"""
Comprehensive Testing Script for HackAudioFeature
Tests all breathing and sound recognition features
"""

import os
import sys
import numpy as np
import requests
import time
from datetime import datetime

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_section(text):
    """Print a section header"""
    print(f"\n--- {text} ---")

def test_imports():
    """Test that all required libraries are available"""
    print_header("Testing Library Imports")
    
    libraries = {
        'numpy': np,
        'requests': requests,
    }
    
    # Test optional libraries
    try:
        import scipy
        libraries['scipy'] = scipy
        print("OK scipy: Available")
    except ImportError:
        print("WARNING scipy: Not available (will use fallback)")
    
    try:
        import librosa
        libraries['librosa'] = librosa
        print("OK librosa: Available")
    except ImportError:
        print("WARNING librosa: Not available")
    
    try:
        import webrtcvad
        libraries['webrtcvad'] = webrtcvad
        print("OK webrtcvad: Available")
    except ImportError:
        print("WARNING webrtcvad: Not available (will use fallback)")
    
    try:
        from app import voice_app, analyze_neonatal_audio
        print("OK app module: Imported successfully")
        return True
    except ImportError as e:
        print(f"ERROR: Cannot import app module: {e}")
        return False
    
    return True

def test_health_endpoint():
    """Test the health endpoint"""
    print_header("Testing Health Endpoint")
    
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"OK Health endpoint responded")
            print(f"  Status: {data.get('status')}")
            print(f"  Lava: {'OK' if data.get('services', {}).get('lava') else 'Disabled'}")
            print(f"  Phoenix: {'OK' if data.get('services', {}).get('phoenix') else 'Disabled'}")
            return True
        else:
            print(f"ERROR: Health endpoint returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Is app.py running?")
        print("  Run: python app.py")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_audio_analysis():
    """Test the enhanced audio analysis features"""
    print_header("Testing Audio Analysis Features")
    
    try:
        from app import analyze_neonatal_audio
        
        # Generate different test audio patterns
        test_cases = [
            ("Random noise (baseline)", np.random.normal(0, 0.1, 44100 * 2)),
            ("Simulated breathing", np.sin(2 * np.pi * 0.8 * np.linspace(0, 2, 44100*2)) * 0.3),
            ("Simulated cry", np.sin(2 * np.pi * 400 * np.linspace(0, 2, 44100*2)) * 0.4),
        ]
        
        for name, audio in test_cases:
            print_section(f"Testing: {name}")
            result = analyze_neonatal_audio(audio)
            
            print(f"  Breathing Rate: {result['breathing_rate']:.1f} bpm")
            print(f"  Pattern: {result['breathing_pattern']}")
            print(f"  Confidence: {result['breathing_confidence']:.2f}")
            print(f"  Cry Quality: {result['cry_quality']}")
            print(f"  Signal Quality: {result['signal_quality']}")
            print(f"  VAD Activity: {result['vad_activity']:.2f}")
            print(f"  Alert Level: {result['alert_level']}")
            
        return True
    except Exception as e:
        print(f"ERROR: Audio analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the API endpoints"""
    print_header("Testing API Endpoints")
    
    base_url = "http://localhost:5000"
    
    # Test analyze_audio endpoint
    print_section("POST /analyze_audio")
    try:
        test_data = {
            'condition': 'healthy',
            'severity': 'normal',
            'real_time': False
        }
        
        response = requests.post(f"{base_url}/analyze_audio", json=test_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("OK Analyzed audio successfully")
            print(f"  Breathing Rate: {data.get('breathing_rate', 0):.1f} bpm")
            print(f"  Alert Level: {data.get('alert_level', 'unknown')}")
            print(f"  Signal Quality: {data.get('signal_quality', 'unknown')}")
        else:
            print(f"ERROR: Status {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test start_monitoring endpoint
    print_section("POST /start_monitoring")
    try:
        response = requests.post(f"{base_url}/start_monitoring", json={}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("OK Monitoring started successfully")
            print(f"  Status: {data.get('status')}")
        else:
            print(f"ERROR: Status {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test get_alerts endpoint
    print_section("GET /get_alerts")
    try:
        response = requests.get(f"{base_url}/get_alerts", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("OK Retrieved alerts successfully")
            print(f"  Alert Count: {data.get('alert_count', 0)}")
        else:
            print(f"ERROR: Status {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    return True

def test_medical_datasets():
    """Test medical dataset generation"""
    print_header("Testing Medical Datasets")
    
    try:
        from medical_audio_datasets import medical_datasets
        
        conditions = ['healthy', 'asphyxia', 'jaundice', 'cyanosis']
        severities = ['mild', 'moderate', 'severe']
        
        for condition in conditions:
            for severity in severities:
                print_section(f"Generating: {condition} ({severity})")
                audio, metadata = medical_datasets.generate_medical_audio(condition, severity, 2.0)
                print(f"  Generated audio: {len(audio)} samples")
                print(f"  Condition: {metadata.get('condition')}")
                print(f"  Breathing Rate: {metadata.get('breathing_rate', 0):.1f} bpm")
        
        print("\nOK All medical datasets generated successfully")
        return True
    except Exception as e:
        print(f"ERROR: Medical dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n")
    print("="*60)
    print("  HackAudioFeature - Comprehensive Testing Suite")
    print("="*60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Library Imports", test_imports()))
    
    # Test 2: Health endpoint (requires server to be running)
    results.append(("Health Endpoint", test_health_endpoint()))
    
    # Test 3: Audio analysis
    results.append(("Audio Analysis", test_audio_analysis()))
    
    # Test 4: API endpoints (requires server to be running)
    results.append(("API Endpoints", test_api_endpoints()))
    
    # Test 5: Medical datasets
    results.append(("Medical Datasets", test_medical_datasets()))
    
    # Summary
    print_header("Test Summary")
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\nSUCCESS: All tests passed!")
    else:
        print(f"\nWARNING: {total_count - passed_count} test(s) failed")
        print("\nNote: Some tests require the Flask server to be running.")
        print("  Start server with: python app.py")
        print("  Then open: http://localhost:5000")

if __name__ == "__main__":
    main()

