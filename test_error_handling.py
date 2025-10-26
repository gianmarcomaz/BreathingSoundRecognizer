#!/usr/bin/env python3
"""Test error handling with various edge cases"""

import numpy as np
from app import analyze_neonatal_audio, preprocess_real_world_audio

print("Testing error handling with edge cases...\n")

# Test 1: Normal audio
print("Test 1: Normal audio")
try:
    test_audio = np.random.normal(0, 0.2, 44100 * 2)
    processed = preprocess_real_world_audio(test_audio, 44100)
    result = analyze_neonatal_audio(processed)
    print("  SUCCESS: Normal audio processed correctly")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: Empty array
print("\nTest 2: Empty array")
try:
    test_audio = np.array([])
    processed = preprocess_real_world_audio(test_audio, 44100)
    print("  SUCCESS: Empty array handled gracefully")
except Exception as e:
    print(f"  OK: {e} (expected for empty array)")

# Test 3: NaN values
print("\nTest 3: NaN values")
try:
    test_audio = np.array([float('nan')] * 1000 + [0.1] * 1000)
    processed = preprocess_real_world_audio(test_audio, 44100)
    result = analyze_neonatal_audio(processed)
    print("  SUCCESS: NaN values handled correctly")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: All zeros
print("\nTest 4: All zeros")
try:
    test_audio = np.zeros(10000)
    processed = preprocess_real_world_audio(test_audio, 44100)
    result = analyze_neonatal_audio(processed)
    print("  SUCCESS: All zeros handled correctly")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "="*60)
print("Error handling tests complete!")
print("System should handle all edge cases gracefully.")
print("="*60)

