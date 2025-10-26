# How to Test HackAudioFeature

This guide shows you how to test all the breathing and sound recognition features.

## Quick Testing (Recommended)

### Method 1: Run the Test Suite
```bash
python test_all_features.py
```

This will test:
- ✅ All library imports
- ✅ Audio analysis features
- ✅ Medical dataset generation
- ✅ API endpoints (if server is running)

### Method 2: Start the Web Application

**Step 1: Start the Flask server**
```bash
python app.py
```

**Step 2: Open your browser**
```
http://localhost:5000
```

**Step 3: Use the Medical Dashboard**
- The homepage shows the Medical Monitoring Dashboard
- Click "Start Monitoring" button
- Use the test audio buttons to simulate different medical conditions
- Watch real-time analysis update

## Testing Options

### Option 1: Automated Test Suite
**Best for: Quick verification that everything works**

```bash
# Run comprehensive tests
python test_all_features.py

# Run medical system tests
python test_medical_system.py

# Test Lava integration
python test_lava.py
```

### Option 2: Web Interface  
**Best for: Visual testing and real-time monitoring**

1. Start app: `python app.py`
2. Open browser: `http://localhost:5000`
3. Click "Start Monitoring"
4. Test with:
   - 🔊 Test Audio buttons (simulated conditions)
   - Real-time microphone input
   - Different medical scenarios

### Option 3: API Testing
**Best for: Integration testing and programmatic access**

```bash
# Test health endpoint
curl http://localhost:5000/health

# Test audio analysis
curl -X POST http://localhost:5000/analyze_audio \
  -H "Content-Type: application/json" \
  -d '{"condition": "healthy", "severity": "normal"}'

# Test monitoring start
curl -X POST http://localhost:5000/start_monitoring

# Get alerts
curl http://localhost:5000/get_alerts
```

### Option 4: Python Script Testing
**Best for: Custom testing and validation**

Create a test script:

```python
from app import analyze_neonatal_audio
import numpy as np

# Generate test audio
audio = np.random.normal(0, 0.2, 44100 * 2)  # 2 seconds

# Analyze
result = analyze_neonatal_audio(audio)

# Check results
print(f"Breathing Rate: {result['breathing_rate']:.1f} bpm")
print(f"Signal Quality: {result['signal_quality']}")
print(f"VAD Activity: {result['vad_activity']:.2f}")
```

## What Gets Tested

### ✅ Enhanced Features Enabled

1. **WebRTC VAD Integration**
   - Voice activity detection
   - Audio preprocessing
   - Activity scoring

2. **Breathing Analysis**
   - Multi-band frequency analysis
   - Peak detection with outlier filtering
   - Pattern classification (gasping, deep, normal, shallow)
   - Extended range: 0-150 bpm

3. **Sound/Cry Analysis**
   - Windowed FFT (reduces spectral leakage)
   - 7 medical quality indicators
   - Advanced spectral features
   - Jaundice detection

4. **Signal Quality**
   - SNR calculation
   - Combined VAD + SNR assessment
   - Quality levels: excellent, good, fair, poor

## Expected Results

### Healthy Condition
```
Breathing Rate: 40-60 bpm
Breathing Pattern: regular
Alert Level: normal
Cry Quality: strong_clear
Signal Quality: good
```

### Asphyxia Condition
```
Breathing Rate: < 20 bpm
Breathing Pattern: absent or gasping
Alert Level: critical/emergency
Cry Quality: weak_intermittent
Signal Quality: fair
```

### Jaundice Indicators
```
Cry Quality: weak_monotone
Jaundice Risk: moderate/high
Breathing Rate: normal (30-60 bpm)
Alert Level: warning
```

## Troubleshooting

### "Cannot connect to server"
**Solution:** Make sure Flask server is running
```bash
python app.py
```

### "Library not found" errors
**Solution:** Install required packages
```bash
pip install -r requirements_medical.txt
```

### No audio detection
**Solution:** Check microphone permissions or use test audio buttons

### Slow performance
**Solution:** Close other applications, check system resources

## Testing Checklist

- [ ] Libraries installed (numpy, scipy, librosa, webrtcvad)
- [ ] Test suite passes (`python test_all_features.py`)
- [ ] Web interface loads (`http://localhost:5000`)
- [ ] Test audio buttons work
- [ ] Real-time monitoring updates
- [ ] API endpoints respond correctly
- [ ] Medical conditions detected properly
- [ ] Signal quality assessed accurately

## Next Steps

After successful testing:

1. **Use the medical dashboard** for real-time monitoring
2. **Integrate API endpoints** in your application
3. **Customize analysis** for your specific use case
4. **Monitor performance** using health endpoints

All breathing and sound recognition features are now fully enabled and ready to use!

