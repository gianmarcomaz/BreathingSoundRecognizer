# Unified Neonatal Monitoring Streamlit App

## Overview

This is a unified Streamlit application combining two major neonatal health monitoring views:
1. **Breathing Sound Recognition** - Real-time audio analysis for respiratory monitoring
2. **Image & Vitals Analysis** - Image-based color analysis + vitals risk assessment

## Features

### 🫁 Breathing Sound Recognition
- Real-time microphone input capture (or synthetic data fallback)
- Advanced audio preprocessing with noise reduction
- Breathing rate analysis (breaths per minute)
- Cry intensity and frequency detection
- Distress score calculation
- Medical condition assessment (asphyxia, jaundice, cyanosis)
- WebRTC Voice Activity Detection (VAD)
- Ultra-low latency analysis (<100ms)

### 🖼️ Image & Vitals Analysis
- Baby photo upload and preprocessing (auto-white-balance, gamma correction)
- Skin segmentation using K-means clustering in Lab color space
- Color statistics extraction (L*, a*, b*, chroma)
- Condition probability assessment:
  - Jaundice (yellow tint detection)
  - Cyanosis (blue tint detection)
  - Pallor/Asphyxia (low saturation indicators)
- Vital signs entry and validation
- Risk of Resuscitation (RoR) score calculation
- JSON report generation

## Installation

### 1. Navigate to Project Directory
```bash
cd HackAudioFeature
```

### 2. Install Additional Dependencies
The requirements.txt has been updated with Streamlit and OpenCV. Install them:
```bash
pip install streamlit opencv-python sounddevice
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Running the App

### Quick Start
```bash
streamlit run unified_neonatal_app.py
```

The app will open in your browser at `http://localhost:8501`

### Navigation
- Use the sidebar to switch between views
- **Breathing Sound Recognition**: Click "Start Monitoring" to begin real-time analysis
- **Image & Vitals Analysis**: Upload a baby photo and enter vital signs

## Configuration

### API Keys (Optional)
Create a `.env` file in the project root with:

```env
# Phoenix/Arize (for telemetry)
PHOENIX_API_KEY=your_phoenix_api_key
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com

# Lava Payments (for usage tracking)
LAVA_SECRET_KEY=your_lava_secret_key
LAVA_CONNECTION_SECRET=your_connection_secret

# LiveKit (for voice chat)
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret

# Vapi (for automated calls)
VAPI_SECRET_TOKEN=your_vapi_secret_token
VAPI_ASSISTANT_ID=your_assistant_id
```

If APIs are not configured, the app will run in demo mode with synthetic data.

## Usage

### Breathing Sound Recognition View

1. **Start Monitoring**
   - Click "Start Monitoring" button
   - Microphone will capture 2-second audio windows
   - Metrics update in real-time

2. **Interpret Results**
   - **Breathing Rate**: Normal range 30-60 bpm
   - **Distress Score**: Higher = more distressed
   - **Alert Level**: normal/warning/critical/emergency
   - **Cry Intensity/Frequency**: For jaundice detection

3. **Test Conditions**
   - Use test buttons to simulate different conditions
   - Results show how system responds to healthy vs. asphyxia patterns

### Image & Vitals Analysis View

1. **Upload Photo**
   - Upload a baby's face or chest photo
   - Image is automatically preprocessed
   - Skin region is isolated and analyzed

2. **View Color Statistics**
   - **L***: Lightness (high = pale skin)
   - **a***: Green-red axis (high = yellow)
   - **b***: Blue-yellow axis (high = yellow, low = blue)
   - **C***: Chroma (saturation)

3. **Condition Probabilities**
   - Jaundice probability based on yellow tint
   - Cyanosis probability based on blue tint
   - Pallor/Asphyxia based on low saturation

4. **Enter Vitals**
   - Fill in respiratory rate, heart rate, temperature, SpO₂
   - Select capillary refill status
   - Check clinical indicators (retractions, lethargy)

5. **Calculate Risk**
   - RoR (Risk of Resuscitation) score is calculated
   - **LOW (<30)**: Continue routine monitoring
   - **MODERATE (30-60)**: Monitor closely
   - **HIGH (>60)**: Seek urgent evaluation

6. **Download Report**
   - JSON report contains all metrics and recommendations

## Technical Details

### Audio Processing Pipeline
1. **Capture**: Microphone input (2-sec windows) or synthetic data
2. **Preprocessing**: 
   - DC removal
   - High-pass filtering (50Hz cutoff)
   - Median filtering for noise reduction
3. **VAD**: WebRTC voice activity detection
4. **Analysis**:
   - Bandpass filtering for breathing (0.1-2 Hz)
   - FFT for cry frequency analysis
   - Peak detection for breathing rate
5. **Assessment**: Medical condition classification
6. **Output**: Metrics + recommendations

### Image Processing Pipeline
1. **Preprocessing**: 
   - Auto-white-balance (gray world)
   - Gamma correction
   - MSRCR tone mapping
2. **Segmentation**: 
   - K-means clustering in Lab color space
   - Morphological operations
3. **Color Analysis**:
   - L*, a*, b* extraction
   - Chroma and hue calculation
4. **Probability Calculation**:
   - Jaundice: high b* value
   - Cyanosis: low b* value
   - Pallor: low saturation
5. **RoR Calculation**: 
   - Combines image probabilities (40%) + vitals risk (60%)

## Error Handling

- **Missing microphone**: Falls back to synthetic data
- **API not configured**: Runs in demo mode
- **Image processing fails**: Shows error message
- **Invalid audio**: Returns safe default metrics
- **No skin detected**: Displays error and continues

## Testing

### Test Audio Patterns
- Healthy pattern: 40-50 bpm regular breathing
- Asphyxia pattern: 10-20 bpm irregular breathing
- Jaundice: weak monotone cry
- Cyanosis: rapid shallow breathing

### Test Conditions
Use the test buttons in the Breathing Sound Recognition view to simulate different medical conditions.

## Sponsor API Integration

### Phoenix/Arize
- Telemetry and observability
- Medical metrics logging
- Dashboard for monitoring

### Lava Payments
- Usage-based billing for LLM calls
- Medical data processing tracking

### LiveKit
- Real-time voice chat capability
- WebRTC audio streaming

### Vapi
- Automated emergency calls
- Voice AI assistant integration

## Troubleshooting

### Microphone Not Working
- Grant browser permissions
- Use synthetic data for testing
- Check system audio settings

### Image Processing Fails
- Ensure good lighting in photo
- Photo should show baby's face or chest
- Try different angles

### APIs Not Connecting
- Check `.env` file configuration
- Verify API keys are valid
- App will run in demo mode without APIs

## Development

### File Structure
```
HackAudioFeature/
├── unified_neonatal_app.py    # Main Streamlit app
├── medical_analysis.py         # Audio analysis functions
├── medical_audio_datasets.py   # Synthetic audio generation
├── requirements.txt            # Dependencies
└── .env                        # API keys (create this)
```

### Key Functions

**Audio Analysis:**
- `analyze_neonatal_audio()` - Main audio processing
- `preprocess_real_world_audio()` - Noise reduction
- `preprocess_audio_with_vad()` - Voice activity detection

**Image Analysis:**
- `preprocess_image()` - Image enhancement
- `segment_skin_lab_kmeans()` - Skin segmentation
- `extract_color_statistics()` - Color analysis
- `calculate_condition_probabilities()` - Medical assessment
- `calculate_vitals_risk()` - Vitals-based risk
- `calculate_final_ror()` - Combined RoR score

## License

This project is part of a hackathon submission integrating multiple sponsor technologies for neonatal health monitoring.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs for error messages
3. Ensure all dependencies are installed
4. Verify API keys are correctly configured
