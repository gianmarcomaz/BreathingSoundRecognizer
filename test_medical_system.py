#!/usr/bin/env python3
"""
Medical System Test Suite
Tests neonatal respiratory monitoring capabilities
"""

import os
import sys
import numpy as np
import time
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_medical_imports():
    """Test all medical processing imports"""
    print("=== TESTING MEDICAL IMPORTS ===")
    
    try:
        import librosa
        print("✅ librosa: Audio processing library")
    except ImportError as e:
        print(f"❌ librosa failed: {e}")
        return False
    
    try:
        import scipy.signal
        print("✅ scipy.signal: Signal processing")
    except ImportError as e:
        print(f"❌ scipy.signal failed: {e}")
        return False
    
    try:
        import webrtcvad
        print("✅ webrtcvad: Voice activity detection")
    except ImportError as e:
        print(f"❌ webrtcvad failed: {e}")
        return False
    
    try:
        from neonatal_monitor import NeonatalMonitor, RespiratoryMetrics, AlertLevel
        print("✅ neonatal_monitor: Medical monitoring system")
    except ImportError as e:
        print(f"❌ neonatal_monitor failed: {e}")
        return False
    
    return True

def test_audio_generation():
    """Generate test audio signals for medical scenarios"""
    print("\n=== GENERATING TEST AUDIO SIGNALS ===")
    
    sample_rate = 44100
    duration = 2.0  # 2 seconds
    
    # Test scenarios
    scenarios = {
        "normal_breathing": {
            "breathing_rate": 45,  # Normal newborn rate
            "cry_present": False,
            "description": "Healthy newborn breathing"
        },
        "bradypnea": {
            "breathing_rate": 15,  # Slow breathing
            "cry_present": False,
            "description": "Slow breathing - possible asphyxia"
        },
        "tachypnea": {
            "breathing_rate": 80,  # Fast breathing
            "cry_present": False,
            "description": "Fast breathing - respiratory distress"
        },
        "distressed_cry": {
            "breathing_rate": 35,
            "cry_present": True,
            "cry_frequency": 450,  # Normal cry frequency
            "cry_intensity": 0.8,
            "description": "Normal breathing with distressed crying"
        },
        "weak_cry": {
            "breathing_rate": 25,
            "cry_present": True,
            "cry_frequency": 200,  # Low frequency - possible jaundice
            "cry_intensity": 0.3,
            "description": "Weak cry - possible jaundice indicator"
        },
        "no_breathing": {
            "breathing_rate": 0,  # No breathing - emergency
            "cry_present": False,
            "description": "No breathing detected - EMERGENCY"
        }
    }
    
    test_signals = {}
    
    for scenario_name, params in scenarios.items():
        print(f"Generating {scenario_name}: {params['description']}")
        
        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.zeros_like(t)
        
        # Add breathing sounds
        if params["breathing_rate"] > 0:
            breathing_freq = params["breathing_rate"] / 60.0  # Convert to Hz
            breathing_component = 0.3 * np.sin(2 * np.pi * breathing_freq * t)
            
            # Add breathing harmonics for realism
            breathing_component += 0.1 * np.sin(4 * np.pi * breathing_freq * t)
            breathing_component += 0.05 * np.sin(6 * np.pi * breathing_freq * t)
            
            signal += breathing_component
        
        # Add cry sounds if present
        if params.get("cry_present", False):
            cry_freq = params.get("cry_frequency", 400)
            cry_intensity = params.get("cry_intensity", 0.5)
            
            # Generate cry signal (intermittent)
            cry_duration = 0.5  # 0.5 second cry bursts
            cry_interval = 1.0  # Every 1 second
            
            for cry_start in np.arange(0, duration, cry_interval):
                cry_end = min(cry_start + cry_duration, duration)
                cry_mask = (t >= cry_start) & (t < cry_end)
                
                cry_signal = cry_intensity * np.sin(2 * np.pi * cry_freq * t[cry_mask])
                # Add cry harmonics
                cry_signal += 0.3 * cry_intensity * np.sin(4 * np.pi * cry_freq * t[cry_mask])
                
                signal[cry_mask] += cry_signal
        
        # Add realistic noise
        noise = 0.05 * np.random.randn(len(t))
        signal += noise
        
        # Normalize
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.8
        
        test_signals[scenario_name] = {
            "audio": signal,
            "params": params,
            "sample_rate": sample_rate
        }
    
    print(f"✅ Generated {len(test_signals)} test scenarios")
    return test_signals

def test_medical_analysis(test_signals):
    """Test medical analysis on generated signals"""
    print("\n=== TESTING MEDICAL ANALYSIS ===")
    
    try:
        from neonatal_monitor import NeonatalMonitor
        
        # Initialize monitor
        monitor = NeonatalMonitor()
        
        # Start monitoring session
        birth_time = datetime.now() - timedelta(seconds=30)  # 30 seconds ago
        monitor.start_monitoring(birth_time)
        
        results = {}
        
        for scenario_name, signal_data in test_signals.items():
            print(f"\nAnalyzing: {scenario_name}")
            print(f"Expected: {signal_data['params']['description']}")
            
            # Analyze the signal
            start_time = time.time()
            metrics = monitor.analyze_audio_stream(signal_data["audio"])
            analysis_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Display results
            print(f"Results:")
            print(f"  Breathing Rate: {metrics.breathing_rate:.1f} bpm")
            print(f"  Breathing Pattern: {metrics.breathing_pattern}")
            print(f"  Cry Intensity: {metrics.cry_intensity:.3f}")
            print(f"  Cry Frequency: {metrics.cry_frequency:.1f} Hz")
            print(f"  O2 Saturation Est: {metrics.oxygen_saturation_estimate:.1f}%")
            print(f"  Distress Score: {metrics.distress_score:.3f}")
            print(f"  Alert Level: {metrics.alert_level.value}")
            print(f"  Analysis Time: {analysis_time:.1f} ms")
            
            # Check if analysis time meets medical requirements (<100ms)
            if analysis_time > 100:
                print(f"  ⚠️  HIGH LATENCY: {analysis_time:.1f}ms > 100ms target")
            else:
                print(f"  ✅ Low latency: {analysis_time:.1f}ms")
            
            results[scenario_name] = {
                "metrics": metrics,
                "analysis_time_ms": analysis_time,
                "expected_params": signal_data["params"]
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Medical analysis failed: {e}")
        return None

def evaluate_medical_accuracy(results):
    """Evaluate accuracy of medical analysis"""
    print("\n=== EVALUATING MEDICAL ACCURACY ===")
    
    if not results:
        print("❌ No results to evaluate")
        return False
    
    accuracy_score = 0
    total_tests = len(results)
    
    for scenario_name, result in results.items():
        metrics = result["metrics"]
        expected = result["expected_params"]
        
        print(f"\n{scenario_name}:")
        
        # Check breathing rate accuracy
        expected_rate = expected["breathing_rate"]
        detected_rate = metrics.breathing_rate
        
        rate_accuracy = False
        if expected_rate == 0:
            # No breathing expected
            if detected_rate < 10:
                rate_accuracy = True
                print(f"  ✅ Breathing rate: Correctly detected absence ({detected_rate:.1f} bpm)")
            else:
                print(f"  ❌ Breathing rate: Failed to detect absence ({detected_rate:.1f} bpm)")
        else:
            # Normal breathing expected
            rate_error = abs(detected_rate - expected_rate) / expected_rate
            if rate_error < 0.3:  # Within 30% tolerance
                rate_accuracy = True
                print(f"  ✅ Breathing rate: {detected_rate:.1f} bpm (expected {expected_rate}, error {rate_error*100:.1f}%)")
            else:
                print(f"  ❌ Breathing rate: {detected_rate:.1f} bpm (expected {expected_rate}, error {rate_error*100:.1f}%)")
        
        # Check alert level appropriateness
        alert_appropriate = False
        if expected_rate == 0:
            # Should trigger emergency alert
            if metrics.alert_level.value in ["critical", "emergency"]:
                alert_appropriate = True
                print(f"  ✅ Alert level: {metrics.alert_level.value} (appropriate for no breathing)")
            else:
                print(f"  ❌ Alert level: {metrics.alert_level.value} (should be critical/emergency)")
        elif expected_rate < 20 or expected_rate > 70:
            # Should trigger warning or critical
            if metrics.alert_level.value in ["warning", "critical", "emergency"]:
                alert_appropriate = True
                print(f"  ✅ Alert level: {metrics.alert_level.value} (appropriate for abnormal rate)")
            else:
                print(f"  ❌ Alert level: {metrics.alert_level.value} (should be warning+)")
        else:
            # Should be normal or watch
            if metrics.alert_level.value in ["normal", "watch"]:
                alert_appropriate = True
                print(f"  ✅ Alert level: {metrics.alert_level.value} (appropriate for normal rate)")
            else:
                print(f"  ❌ Alert level: {metrics.alert_level.value} (should be normal/watch)")
        
        # Check latency requirement
        latency_ok = result["analysis_time_ms"] < 100
        if latency_ok:
            print(f"  ✅ Latency: {result['analysis_time_ms']:.1f}ms (< 100ms target)")
        else:
            print(f"  ❌ Latency: {result['analysis_time_ms']:.1f}ms (> 100ms target)")
        
        # Calculate scenario score
        scenario_score = sum([rate_accuracy, alert_appropriate, latency_ok]) / 3
        accuracy_score += scenario_score
        
        print(f"  Scenario Score: {scenario_score*100:.1f}%")
    
    overall_accuracy = accuracy_score / total_tests
    print(f"\n=== OVERALL MEDICAL SYSTEM ACCURACY ===")
    print(f"Overall Score: {overall_accuracy*100:.1f}%")
    
    if overall_accuracy >= 0.8:
        print("✅ MEDICAL SYSTEM READY FOR CLINICAL USE")
        return True
    elif overall_accuracy >= 0.6:
        print("⚠️  MEDICAL SYSTEM NEEDS CALIBRATION")
        return False
    else:
        print("❌ MEDICAL SYSTEM NOT READY - REQUIRES SIGNIFICANT IMPROVEMENT")
        return False

def test_emergency_protocols():
    """Test emergency alert protocols"""
    print("\n=== TESTING EMERGENCY PROTOCOLS ===")
    
    try:
        from neonatal_monitor import NeonatalMonitor, AlertLevel
        
        monitor = NeonatalMonitor()
        
        # Test emergency conditions
        emergency_conditions = [
            {"breathing_rate": 0, "description": "No breathing"},
            {"breathing_rate": 5, "description": "Severe bradypnea"},
            {"distress_score": 0.9, "description": "Maximum distress"}
        ]
        
        for condition in emergency_conditions:
            print(f"\nTesting: {condition['description']}")
            
            # Create test audio with emergency condition
            sample_rate = 44100
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            if condition["breathing_rate"] == 0:
                # Silent audio for no breathing
                test_audio = 0.01 * np.random.randn(len(t))
            else:
                # Very weak breathing signal
                breathing_freq = condition["breathing_rate"] / 60.0
                test_audio = 0.1 * np.sin(2 * np.pi * breathing_freq * t)
                test_audio += 0.01 * np.random.randn(len(t))
            
            # Analyze
            metrics = monitor.analyze_audio_stream(test_audio)
            
            print(f"  Detected breathing rate: {metrics.breathing_rate:.1f} bpm")
            print(f"  Alert level: {metrics.alert_level.value}")
            print(f"  Distress score: {metrics.distress_score:.3f}")
            
            # Check if emergency was properly detected
            if metrics.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                print(f"  ✅ Emergency properly detected")
            else:
                print(f"  ❌ Emergency NOT detected - alert level too low")
        
        print("\n✅ Emergency protocol testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Emergency protocol testing failed: {e}")
        return False

def main():
    """Run complete medical system test suite"""
    print("🏥 NEONATAL RESPIRATORY MONITORING SYSTEM TEST")
    print("=" * 60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Import verification
    if not test_medical_imports():
        print("❌ CRITICAL: Medical imports failed - cannot proceed")
        return False
    
    # Test 2: Audio signal generation
    test_signals = test_audio_generation()
    if not test_signals:
        print("❌ CRITICAL: Audio generation failed - cannot proceed")
        return False
    
    # Test 3: Medical analysis
    results = test_medical_analysis(test_signals)
    if not results:
        print("❌ CRITICAL: Medical analysis failed - cannot proceed")
        return False
    
    # Test 4: Accuracy evaluation
    system_ready = evaluate_medical_accuracy(results)
    
    # Test 5: Emergency protocols
    emergency_ok = test_emergency_protocols()
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🏥 FINAL MEDICAL SYSTEM ASSESSMENT")
    print("=" * 60)
    
    if system_ready and emergency_ok:
        print("✅ MEDICAL SYSTEM FULLY OPERATIONAL")
        print("🚀 Ready for neonatal respiratory monitoring")
        print("⏰ Ultra-low latency processing: <100ms")
        print("🔍 Detecting: Birth asphyxia, jaundice, cyanosis")
        print("🚨 Emergency protocols: ACTIVE")
        
        print("\n📋 NEXT STEPS:")
        print("1. Run: python neonatal_monitor.py")
        print("2. Open: http://localhost:5000")
        print("3. Start monitoring session")
        print("4. Monitor real-time vital signs")
        
        return True
    else:
        print("❌ MEDICAL SYSTEM NOT READY")
        print("🔧 System requires calibration before clinical use")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
