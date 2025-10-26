# Quick Start - Live Testing with Actual Child

## 🚀 3-Step Live Test

### Step 1: Start the Server
```bash
cd C:\Users\Admin\OneDrive\Desktop\work\HackAudioFeature
python app.py
```

### Step 2: Open Dashboard
```
http://localhost:5000
```

### Step 3: Click "Start Monitoring"
1. Allow microphone access
2. Position near child (1-3 feet away)
3. Watch real-time results

## ✅ What's Now Enhanced for Real Child Monitoring

- ✅ Real-world audio preprocessing (noise reduction)
- ✅ High-pass filtering (removes background noise)
- ✅ DC offset removal (accurate frequency analysis)
- ✅ Gentle compression (enhances weak breathing)
- ✅ Adaptive denoising (removes impulse noise)
- ✅ Voice activity detection (better signal quality)
- ✅ Extended validation (ensures sample adequacy)

## 📊 What You'll See

```
Breathing Rate: X bpm (normal: 30-60)
Breathing Pattern: regular/irregular/shallow
Cry Quality: strong_clear/weak_monotone/etc
Alert Level: normal/watch/warning/critical/emergency
Signal Quality: excellent/good/fair/poor
VAD Activity: 0.0-1.0 (higher = more activity)
```

## 🎯 Expected Results - Healthy Child

- **Breathing Rate**: 30-60 bpm
- **Pattern**: regular
- **Alert**: normal
- **Quality**: good or excellent
- **VAD**: 0.5-1.0

## ⚠️ When to Get Medical Help

Get emergency help if:
- Breathing rate < 20 or > 80 bpm
- Alert level = "critical" or "emergency"
- Breathing pattern = "absent" or "gasping"
- Child appears in distress

## 📱 Positioning Tips

- **Distance**: 1-3 feet from child
- **Height**: Level with chest
- **Environment**: Quiet
- **Duration**: 30+ seconds for accuracy

## 🔧 Troubleshooting

**No detection?**
→ Check mic permissions, move closer, try test buttons

**Poor quality?**
→ Reduce noise, improve positioning, check mic input

**False alerts?**
→ Wait 30s for averaging, ensure quiet environment

## 📖 Full Guide
See `LIVE_TESTING_GUIDE.md` for complete instructions

## ⚡ Quick Test First
Before live testing, verify with test audio:
1. Open dashboard
2. Click "Start Monitoring"  
3. Use 🔊 test audio buttons
4. Verify readings are reasonable
5. Then proceed to live testing

---

**Ready to monitor! Start with `python app.py`**

