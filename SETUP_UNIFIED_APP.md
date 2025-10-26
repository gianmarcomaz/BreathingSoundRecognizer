# Setup Guide: Unified Neonatal Monitoring App

## Step-by-Step Installation

### Step 1: Install Streamlit and Required Packages

```bash
cd C:\Users\Admin\OneDrive\Desktop\work\HackAudioFeature
pip install streamlit opencv-python sounddevice
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import streamlit, cv2; print('✅ All dependencies installed')"
```

Expected output:
```
✅ All dependencies installed
```

### Step 3: Run the App

**Option A: Using the batch script**
```bash
run_unified_app.bat
```

**Option B: Using Python directly**
```bash
streamlit run unified_neonatal_app.py
```

**Option C: Using Custom Port**
```bash
streamlit run unified_neonatal_app.py --server.port 8502
```

### Step 4: Access the App

The app will automatically open in your browser at:
```
http://localhost:8501
```

## Testing the App

### Breathing Sound Recognition View

1. Click "Breathing Sound Recognition" in the sidebar
2. Click "Start Monitoring" button
3. System will capture 2-second audio windows
4. View real-time metrics:
   - Breathing Rate (bpm)
   - Cry Intensity
   - Distress Score
   - Alert Level

**Note:** If microphone access is denied, the app falls back to synthetic data (demo mode).

### Image & Vitals Analysis View

1. Click "Image & Vitals Analysis" in the sidebar
2. Upload a baby photo (PNG, JPG, JPEG)
3. System will:
   - Preprocess image (auto-white-balance, gamma correction)
   - Segment skin region using K-means
   - Extract color statistics (L*, a*, b*)
4. Enter vital signs:
   - Respiratory Rate (0-150 bpm)
   - Heart Rate (0-300 bpm)
   - Temperature (°C)
   - SpO₂ (%)
   - Capillary Refill status
5. Click "Analyze Vitals"
6. View Risk of Resuscitation (RoR) score:
   - **LOW (<30)**: Continue monitoring
   - **MODERATE (30-60)**: Monitor closely
   - **HIGH (>60)**: Seek urgent evaluation
7. Download JSON report

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit
```

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python
```

### Issue: Microphone not working

**Solution:**
- Grant browser permissions for microphone
- App will use synthetic data if microphone is unavailable
- Check system audio settings

### Issue: Image processing fails

**Solution:**
- Ensure photo has good lighting
- Photo should show baby's face or chest clearly
- Try different angles

### Issue: APIs not connecting

**Solution:**
- Create `.env` file with API keys (see next section)
- App runs in demo mode without APIs
- Check API key validity

## Optional: Configure API Keys

Create a `.env` file in the project root:

```env
# Phoenix/Arize
PHOENIX_API_KEY=your_key_here
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com

# Lava Payments
LAVA_SECRET_KEY=your_key_here
LAVA_CONNECTION_SECRET=your_secret_here

# LiveKit
LIVEKIT_API_KEY=your_key_here
LIVEKIT_API_SECRET=your_secret_here

# Vapi
VAPI_SECRET_TOKEN=your_token_here
VAPI_ASSISTANT_ID=your_id_here
```

If APIs are not configured, the app will still work in demo mode.

## App Structure

```
unified_neonatal_app.py
├── Breathing Sound Recognition View
│   ├── Real-time audio capture
│   ├── Audio analysis (breathing, cry, distress)
│   ├── Medical condition assessment
│   └── Test condition buttons
│
└── Image & Vitals Analysis View
    ├── Image upload
    ├── Skin segmentation
    ├── Color analysis
    ├── Condition probabilities
    ├── Vitals entry
    └── RoR calculation
```

## Features Summary

### Breathing Sound Recognition
- ✅ Real-time microphone capture (or synthetic fallback)
- ✅ Breathing rate analysis (bpm)
- ✅ Cry intensity and frequency detection
- ✅ Distress score calculation
- ✅ Medical condition assessment
- ✅ WebRTC VAD support
- ✅ Ultra-low latency (<100ms)

### Image & Vitals Analysis
- ✅ Image preprocessing (auto-WB, gamma, MSRCR)
- ✅ Skin segmentation (K-means in Lab space)
- ✅ Color statistics (L*, a*, b*, chroma)
- ✅ Jaundice probability (yellow tint)
- ✅ Cyanosis probability (blue tint)
- ✅ Pallor/Asphyxia probability (low saturation)
- ✅ Vital signs risk assessment
- ✅ Combined RoR score calculation
- ✅ JSON report generation

## Next Steps

1. **Install dependencies** (if not already done)
2. **Run the app** using one of the methods above
3. **Test features** in both views
4. **Optional**: Configure API keys for full functionality

## Development Notes

- The app uses existing `medical_analysis.py` for audio processing
- Image processing is implemented in `unified_neonatal_app.py`
- All sponsor API integrations are optional
- Robust error handling throughout
- Falls back gracefully to demo mode when APIs unavailable

## Support

For issues:
1. Check this setup guide
2. Review UNIFIED_APP_README.md for detailed usage
3. Check console for error messages
4. Verify dependencies are installed correctly
