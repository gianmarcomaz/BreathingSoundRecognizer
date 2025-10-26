# 🧪 Testing Guide for Voice AI + Medical Monitoring App

## ✅ What's Working

### 1. **App Structure**
- ✅ Flask app imports successfully
- ✅ Medical analysis functions imported from `medical_analysis.py`
- ✅ All sponsor integrations (Lava, Phoenix, LiveKit, Vapi)
- ✅ Medical monitoring endpoints ready

### 2. **Files Structure**
```
HackAudioFeature_CLEAN/
├── app.py                    # Main Flask app (823 lines)
├── medical_analysis.py       # Medical analysis module (800+ lines)
├── medical_audio_datasets.py # Test data generator
└── test_medical_system.py    # Test suite
```

---

## 🚀 How to Test

### **Option 1: Quick Import Test**
```bash
python -c "import app; print('✅ App ready!')"
```

### **Option 2: Start the Flask Server**
```bash
# In HackAudioFeature_CLEAN directory
python app.py
```

This will:
- Start Flask on `http://localhost:5000`
- Initialize all services (Lava, Phoenix, LiveKit, Vapi)
- Show service status

### **Option 3: Test Medical Endpoints**

#### **Test 1: Health Check**
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "lava": false,
    "phoenix": true,
    "livekit": false,
    "vapi": false
  }
}
```

#### **Test 2: Analyze Medical Audio**
```bash
curl -X POST http://localhost:5000/analyze_audio \
  -H "Content-Type: application/json" \
  -d '{
    "condition": "healthy",
    "severity": "normal"
  }'
```

Expected response:
```json
{
  "breathing_rate": 45.0,
  "breathing_pattern": "regular",
  "medical_condition": "healthy",
  "alert_level": "normal",
  "oxygen_saturation_estimate": 95.0
}
```

#### **Test 3: Test Different Conditions**
```bash
# Test asphyxia detection
curl -X POST http://localhost:5000/analyze_audio \
  -H "Content-Type: application/json" \
  -d '{"condition": "asphyxia", "severity": "severe"}'

# Test jaundice detection
curl -X POST http://localhost:5000/analyze_audio \
  -H "Content-Type: application/json" \
  -d '{"condition": "jaundice", "severity": "moderate"}'
```

---

## 🌐 Test in Browser

### **Medical Dashboard**
Open: `http://localhost:5000/`
- Real-time neonatal monitoring
- Audio recording and analysis
- Medical alerts display

### **Voice AI Dashboard**
Open: `http://localhost:5000/original`
- Voice chat interface
- AI conversation
- LiveKit voice features

---

## 🧪 Run Full Test Suite

```bash
# Run medical system tests
python test_medical_system.py

# Run specific test
python -c "
from medical_analysis import analyze_neonatal_audio
import numpy as np

# Generate test audio (healthy newborn breathing)
audio = np.random.randn(88200) * 0.1  # 2 seconds at 44.1kHz
result = analyze_neonatal_audio(audio)
print(f'Breathing Rate: {result[\"breathing_rate\"]:.1f} bpm')
print(f'Condition: {result[\"medical_condition\"]}')
print(f'Alert Level: {result[\"alert_level\"]}')
"
```

---

## 📊 Expected Test Results

### **Healthy Newborn**
```json
{
  "breathing_rate": 40-50 bpm,
  "breathing_pattern": "regular",
  "medical_condition": "healthy",
  "alert_level": "normal"
}
```

### **Severe Asphyxia**
```json
{
  "breathing_rate": 5-15 bpm,
  "breathing_pattern": "gasping",
  "medical_condition": "severe_asphyxia",
  "alert_level": "emergency"
}
```

### **Jaundice**
```json
{
  "cry_quality": "weak_monotone",
  "medical_condition": "jaundice",
  "alert_level": "warning"
}
```

---

## ⚠️ Known Limitations

1. **Lava/LiveKit/Vapi**: Require API keys (currently in demo mode)
2. **Phoenix**: May show warnings if API key not configured
3. **Medical Analysis**: Works offline with synthetic data

---

## 🎯 Quick Start Commands

```bash
# 1. Test imports
python -c "import app; print('✅ Ready')"

# 2. Start server
python app.py

# 3. Open browser
# Visit: http://localhost:5000

# 4. Test endpoint
curl http://localhost:5000/health
```

---

## 💡 Tips

- **Medical Dashboard**: Works best for real-time audio monitoring
- **Test Data**: Use `medical_audio_datasets.py` to generate test scenarios
- **Sponsor APIs**: Add keys to `.env` file to enable full features
- **Debugging**: Check terminal output for detailed logs

---

**✅ Your app is ready to test!**
