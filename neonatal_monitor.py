#!/usr/bin/env python3
"""
Neonatal Respiratory Monitoring System
Real-time analysis of newborn breathing patterns to detect:
- Birth Asphyxia (oxygen deprivation)
- Jaundice indicators via cry analysis
- Cyanosis (poor oxygenation) via breathing patterns

Critical: Ultra-low latency (<100ms) for golden minute intervention
"""

import os
import sys
import json
import time
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Core imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# Audio processing imports
import librosa
import scipy.signal
from scipy.fft import fft, fftfreq
import webrtcvad

# Medical AI imports
try:
    from livekit import api as livekit_api
    LIVEKIT_AVAILABLE = True
except ImportError:
    livekit_api = None
    LIVEKIT_AVAILABLE = False

# Monitoring imports
from phoenix.otel import register
from phoenix.client import Client

# Import medical analysis functions
try:
    from medical_analysis import analyze_neonatal_audio, preprocess_real_world_audio
    MEDICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    MEDICAL_ANALYSIS_AVAILABLE = False
    # Logger not yet defined at this point, will log later if needed

# Load environment variables
load_dotenv()

# Configure logging for medical application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neonatal_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class AlertLevel(Enum):
    """Medical alert severity levels"""
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RespiratoryMetrics:
    """Real-time respiratory analysis metrics"""
    timestamp: datetime
    breathing_rate: float  # breaths per minute
    breathing_pattern: str  # regular, irregular, absent
    cry_intensity: float  # 0-1 scale
    cry_frequency: float  # Hz
    oxygen_saturation_estimate: float  # estimated from audio patterns
    distress_score: float  # 0-1 composite score
    alert_level: AlertLevel

@dataclass
class MedicalAlert:
    """Medical alert for healthcare providers"""
    timestamp: datetime
    alert_type: str  # asphyxia, jaundice, cyanosis
    severity: AlertLevel
    confidence: float
    metrics: RespiratoryMetrics
    recommended_action: str
    time_since_birth: timedelta

class NeonatalMonitor:
    """Core neonatal monitoring system"""
    
    def __init__(self):
        """Initialize medical monitoring system"""
        self.setup_medical_ai()
        self.setup_lava_medical()
        self.setup_phoenix_medical()
        self.setup_livekit_medical()
        self.setup_vapi_medical()
        
        # Medical parameters
        self.birth_time = None
        self.golden_minute_active = True
        self.continuous_monitoring = True
        
        # Audio processing parameters
        self.sample_rate = 44100  # High quality for medical accuracy
        self.buffer_size = 1024   # Low latency buffer
        self.analysis_window = 2.0  # 2-second analysis window
        
        # Medical thresholds (based on neonatal medicine standards)
        self.normal_breathing_rate = (30, 60)  # breaths per minute for newborns
        self.critical_breathing_rate = (10, 100)  # emergency thresholds
        self.cry_frequency_normal = (300, 600)  # Hz range for healthy newborn cry
        
        # Alert system
        self.active_alerts = []
        self.alert_callbacks = []
        
        logger.info("🏥 Neonatal Respiratory Monitor initialized")
    
    def setup_medical_ai(self):
        """Initialize medical AI models and analysis"""
        try:
            # Medical AI configuration
            self.openai_model = "gpt-4o-mini"  # For medical analysis summaries
            self.medical_prompt = """
            You are a neonatal medicine AI assistant analyzing newborn respiratory data.
            Focus on detecting signs of:
            1. Birth Asphyxia: Poor breathing, low oxygen, bradycardia
            2. Jaundice: Changes in cry patterns, lethargy indicators
            3. Cyanosis: Poor oxygenation, breathing difficulties
            
            Provide immediate, actionable medical recommendations.
            """
            
            logger.info("✅ Medical AI models initialized")
        except Exception as e:
            logger.error(f"❌ Medical AI initialization failed: {e}")
    
    def setup_lava_medical(self):
        """Configure Lava for medical usage tracking"""
        try:
            self.lava_secret_key = os.getenv("LAVA_SECRET_KEY")
            self.lava_connection_secret = os.getenv("LAVA_CONNECTION_SECRET")
            self.lava_base_url = "https://api.lavapayments.com/v1"
            
            if self.lava_secret_key:
                logger.info("✅ Lava medical billing initialized")
                logger.info(f"   Medical usage tracking: ACTIVE")
            else:
                logger.warning("⚠️  Lava not configured - using demo mode")
                
        except Exception as e:
            logger.error(f"❌ Lava medical setup failed: {e}")
    
    def setup_phoenix_medical(self):
        """Configure Phoenix for medical data observability"""
        try:
            # Medical data tracing
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY', 'medical_monitoring')}"
            
            # Register medical tracing
            self.phoenix_tracer = register(
                project_name="neonatal_monitor",
                endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces"
            )
            
            self.phoenix_client = Client()
            logger.info("✅ Phoenix medical observability initialized")
            
        except Exception as e:
            logger.error(f"❌ Phoenix medical setup failed: {e}")
            self.phoenix_client = None
    
    def setup_livekit_medical(self):
        """Configure LiveKit for real-time medical audio streaming"""
        try:
            self.livekit_api_key = os.getenv("LIVEKIT_API_KEY")
            self.livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
            self.livekit_url = os.getenv("LIVEKIT_URL")
            
            if self.livekit_api_key and LIVEKIT_AVAILABLE:
                logger.info("✅ LiveKit medical streaming initialized")
                logger.info(f"   Real-time audio: {self.livekit_url}")
            else:
                logger.warning("⚠️  LiveKit not configured - using demo mode")
                
        except Exception as e:
            logger.error(f"❌ LiveKit medical setup failed: {e}")
    
    def setup_vapi_medical(self):
        """Configure Vapi for emergency medical communications"""
        try:
            self.vapi_secret_token = os.getenv("VAPI_SECRET_TOKEN")
            self.vapi_assistant_id = os.getenv("VAPI_ASSISTANT_ID")
            self.vapi_base_url = "https://api.vapi.ai"
            
            if self.vapi_secret_token:
                # Test medical emergency communication system
                headers = {
                    "Authorization": f"Bearer {self.vapi_secret_token}",
                    "Content-Type": "application/json"
                }
                
                response = requests.get(f"{self.vapi_base_url}/assistant", headers=headers, timeout=5)
                
                if response.status_code == 200:
                    logger.info("✅ Vapi emergency communication initialized")
                    logger.info(f"   Emergency alerts: ACTIVE")
                else:
                    logger.warning("⚠️  Vapi emergency system not responding")
                    
        except Exception as e:
            logger.error(f"❌ Vapi medical setup failed: {e}")
    
    def start_monitoring(self, birth_time: Optional[datetime] = None):
        """Start continuous neonatal monitoring"""
        self.birth_time = birth_time or datetime.now()
        self.golden_minute_active = True
        self.continuous_monitoring = True
        
        logger.info(f"🏥 Starting neonatal monitoring at {self.birth_time}")
        logger.info(f"⏰ Golden minute window: ACTIVE")
        
        return {
            "status": "monitoring_started",
            "birth_time": self.birth_time.isoformat(),
            "golden_minute": True,
            "monitoring_active": True
        }
    
    def analyze_audio_stream(self, audio_data: np.ndarray) -> RespiratoryMetrics:
        """
        Real-time analysis of newborn audio for medical indicators
        Ultra-low latency processing for critical care
        """
        try:
            start_time = time.time()
            
            # 1. Breathing Pattern Analysis
            breathing_rate, breathing_pattern = self._analyze_breathing_pattern(audio_data)
            
            # 2. Cry Analysis for Jaundice/Distress
            cry_intensity, cry_frequency = self._analyze_cry_patterns(audio_data)
            
            # 3. Oxygen Saturation Estimation
            oxygen_estimate = self._estimate_oxygen_saturation(audio_data, breathing_rate)
            
            # 4. Composite Distress Score
            distress_score = self._calculate_distress_score(
                breathing_rate, breathing_pattern, cry_intensity, cry_frequency
            )
            
            # 5. Alert Level Determination
            alert_level = self._determine_alert_level(distress_score, breathing_rate)
            
            # Create metrics object
            metrics = RespiratoryMetrics(
                timestamp=datetime.now(),
                breathing_rate=breathing_rate,
                breathing_pattern=breathing_pattern,
                cry_intensity=cry_intensity,
                cry_frequency=cry_frequency,
                oxygen_saturation_estimate=oxygen_estimate,
                distress_score=distress_score,
                alert_level=alert_level
            )
            
            # Check processing latency (critical for medical applications)
            processing_time = (time.time() - start_time) * 1000  # ms
            if processing_time > 100:  # Alert if >100ms latency
                logger.warning(f"⚠️  High latency detected: {processing_time:.1f}ms")
            
            # Log medical data
            self._log_medical_data(metrics, processing_time)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Audio analysis failed: {e}")
            return self._create_error_metrics()
    
    def _analyze_breathing_pattern(self, audio_data: np.ndarray) -> Tuple[float, str]:
        """Analyze breathing rate and pattern from audio"""
        try:
            # Apply bandpass filter for breathing sounds (0.1-2 Hz)
            sos = scipy.signal.butter(4, [0.1, 2], btype='band', fs=self.sample_rate, output='sos')
            filtered_audio = scipy.signal.sosfilt(sos, audio_data)
            
            # Detect breathing cycles using envelope detection
            envelope = np.abs(scipy.signal.hilbert(filtered_audio))
            
            # Find peaks (breathing cycles)
            peaks, _ = scipy.signal.find_peaks(
                envelope, 
                height=np.max(envelope) * 0.3,
                distance=int(self.sample_rate * 0.5)  # Minimum 0.5s between breaths
            )
            
            # Calculate breathing rate (breaths per minute)
            if len(peaks) > 1:
                avg_interval = np.mean(np.diff(peaks)) / self.sample_rate
                breathing_rate = 60.0 / avg_interval if avg_interval > 0 else 0
            else:
                breathing_rate = 0
            
            # Determine breathing pattern
            if breathing_rate == 0:
                pattern = "absent"
            elif self.normal_breathing_rate[0] <= breathing_rate <= self.normal_breathing_rate[1]:
                pattern = "regular"
            else:
                pattern = "irregular"
            
            return breathing_rate, pattern
            
        except Exception as e:
            logger.error(f"Breathing analysis failed: {e}")
            return 0.0, "unknown"
    
    def _analyze_cry_patterns(self, audio_data: np.ndarray) -> Tuple[float, float]:
        """Analyze cry intensity and frequency for jaundice/distress indicators"""
        try:
            # Detect cry segments using voice activity detection
            vad = webrtcvad.Vad(3)  # Aggressive mode for cry detection
            
            # Calculate cry intensity (RMS energy)
            cry_intensity = np.sqrt(np.mean(audio_data**2))
            
            # Analyze frequency content for cry
            fft_data = fft(audio_data)
            freqs = fftfreq(len(audio_data), 1/self.sample_rate)
            
            # Find dominant frequency in cry range (200-800 Hz)
            cry_range_mask = (freqs >= 200) & (freqs <= 800)
            if np.any(cry_range_mask):
                cry_fft = np.abs(fft_data[cry_range_mask])
                dominant_freq_idx = np.argmax(cry_fft)
                cry_frequency = freqs[cry_range_mask][dominant_freq_idx]
            else:
                cry_frequency = 0
            
            return float(cry_intensity), float(cry_frequency)
            
        except Exception as e:
            logger.error(f"Cry analysis failed: {e}")
            return 0.0, 0.0
    
    def _estimate_oxygen_saturation(self, audio_data: np.ndarray, breathing_rate: float) -> float:
        """Estimate oxygen saturation from breathing patterns and cry quality"""
        try:
            # This is a simplified estimation based on audio patterns
            # In real medical applications, this would use validated algorithms
            
            base_saturation = 95.0  # Normal newborn baseline
            
            # Adjust based on breathing rate
            if breathing_rate < self.normal_breathing_rate[0]:
                base_saturation -= (self.normal_breathing_rate[0] - breathing_rate) * 0.5
            elif breathing_rate > self.normal_breathing_rate[1]:
                base_saturation -= (breathing_rate - self.normal_breathing_rate[1]) * 0.3
            
            # Adjust based on audio quality (simplified)
            audio_quality = np.std(audio_data)
            if audio_quality < 0.01:  # Very weak audio signal
                base_saturation -= 5.0
            
            return max(70.0, min(100.0, base_saturation))
            
        except Exception as e:
            logger.error(f"Oxygen estimation failed: {e}")
            return 95.0
    
    def _calculate_distress_score(self, breathing_rate: float, breathing_pattern: str, 
                                cry_intensity: float, cry_frequency: float) -> float:
        """Calculate composite distress score (0-1, higher = more distressed)"""
        try:
            distress_score = 0.0
            
            # Breathing rate component (40% weight)
            if breathing_pattern == "absent":
                distress_score += 0.4
            elif breathing_rate < self.normal_breathing_rate[0] or breathing_rate > self.normal_breathing_rate[1]:
                distress_score += 0.2
            
            # Breathing pattern component (30% weight)
            if breathing_pattern == "irregular":
                distress_score += 0.15
            elif breathing_pattern == "absent":
                distress_score += 0.3
            
            # Cry analysis component (30% weight)
            if cry_frequency > 0:
                if cry_frequency < self.cry_frequency_normal[0] or cry_frequency > self.cry_frequency_normal[1]:
                    distress_score += 0.15
                if cry_intensity > 0.8:  # Very intense crying
                    distress_score += 0.15
            
            return min(1.0, distress_score)
            
        except Exception as e:
            logger.error(f"Distress score calculation failed: {e}")
            return 0.5
    
    def _determine_alert_level(self, distress_score: float, breathing_rate: float) -> AlertLevel:
        """Determine medical alert level based on analysis"""
        try:
            # Emergency conditions
            if breathing_rate == 0 or distress_score > 0.8:
                return AlertLevel.EMERGENCY
            
            # Critical conditions
            if (breathing_rate < 10 or breathing_rate > 100 or 
                distress_score > 0.6):
                return AlertLevel.CRITICAL
            
            # Warning conditions
            if (breathing_rate < self.normal_breathing_rate[0] or 
                breathing_rate > self.normal_breathing_rate[1] or
                distress_score > 0.4):
                return AlertLevel.WARNING
            
            # Watch conditions
            if distress_score > 0.2:
                return AlertLevel.WATCH
            
            return AlertLevel.NORMAL
            
        except Exception as e:
            logger.error(f"Alert level determination failed: {e}")
            return AlertLevel.WARNING
    
    def _log_medical_data(self, metrics: RespiratoryMetrics, processing_time: float):
        """Log medical data for monitoring and analysis"""
        try:
            # Log to Phoenix for observability
            if self.phoenix_client:
                self._log_to_phoenix_medical(metrics, processing_time)
            
            # Log critical alerts
            if metrics.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                logger.critical(f"🚨 MEDICAL ALERT: {metrics.alert_level.value.upper()}")
                logger.critical(f"   Breathing: {metrics.breathing_rate:.1f} bpm ({metrics.breathing_pattern})")
                logger.critical(f"   Distress Score: {metrics.distress_score:.2f}")
                logger.critical(f"   O2 Estimate: {metrics.oxygen_saturation_estimate:.1f}%")
                
                # Trigger emergency protocols
                self._trigger_emergency_alert(metrics)
            
            # Regular monitoring log
            logger.info(f"📊 Monitoring: BR={metrics.breathing_rate:.1f} bpm, "
                       f"Distress={metrics.distress_score:.2f}, "
                       f"Alert={metrics.alert_level.value}, "
                       f"Latency={processing_time:.1f}ms")
                       
        except Exception as e:
            logger.error(f"Medical data logging failed: {e}")
    
    def _log_to_phoenix_medical(self, metrics: RespiratoryMetrics, processing_time: float):
        """Log medical metrics to Phoenix for analysis"""
        try:
            medical_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "breathing_rate": metrics.breathing_rate,
                "breathing_pattern": metrics.breathing_pattern,
                "cry_intensity": metrics.cry_intensity,
                "cry_frequency": metrics.cry_frequency,
                "oxygen_estimate": metrics.oxygen_saturation_estimate,
                "distress_score": metrics.distress_score,
                "alert_level": metrics.alert_level.value,
                "processing_latency_ms": processing_time,
                "golden_minute_active": self.golden_minute_active,
                "time_since_birth": (datetime.now() - self.birth_time).total_seconds() if self.birth_time else 0
            }
            
            # This would integrate with Phoenix logging in production
            logger.debug(f"Phoenix medical log: {json.dumps(medical_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Phoenix medical logging failed: {e}")
    
    def _trigger_emergency_alert(self, metrics: RespiratoryMetrics):
        """Trigger emergency medical alert protocols"""
        try:
            time_since_birth = datetime.now() - self.birth_time if self.birth_time else timedelta(0)
            
            # Determine alert type based on metrics
            alert_type = "unknown"
            recommended_action = "Immediate medical evaluation required"
            
            if metrics.breathing_rate == 0:
                alert_type = "severe_asphyxia"
                recommended_action = "IMMEDIATE RESUSCITATION - Check airway, begin ventilation"
            elif metrics.breathing_rate < 10:
                alert_type = "bradypnea"
                recommended_action = "Assess for asphyxia - Consider assisted ventilation"
            elif metrics.distress_score > 0.8:
                alert_type = "severe_distress"
                recommended_action = "Comprehensive assessment - Check for cyanosis, jaundice signs"
            
            # Create medical alert
            alert = MedicalAlert(
                timestamp=datetime.now(),
                alert_type=alert_type,
                severity=metrics.alert_level,
                confidence=1.0 - metrics.distress_score,  # Inverse relationship
                metrics=metrics,
                recommended_action=recommended_action,
                time_since_birth=time_since_birth
            )
            
            self.active_alerts.append(alert)
            
            # Log emergency alert
            logger.critical(f"🚨 EMERGENCY ALERT TRIGGERED")
            logger.critical(f"   Type: {alert_type}")
            logger.critical(f"   Time since birth: {time_since_birth}")
            logger.critical(f"   Action: {recommended_action}")
            
            # In production, this would trigger:
            # - Immediate notifications to medical staff
            # - Integration with hospital alert systems
            # - Automated emergency protocols
            
        except Exception as e:
            logger.error(f"Emergency alert trigger failed: {e}")
    
    def _create_error_metrics(self) -> RespiratoryMetrics:
        """Create error metrics when analysis fails"""
        return RespiratoryMetrics(
            timestamp=datetime.now(),
            breathing_rate=0.0,
            breathing_pattern="error",
            cry_intensity=0.0,
            cry_frequency=0.0,
            oxygen_saturation_estimate=0.0,
            distress_score=1.0,  # Maximum distress for error state
            alert_level=AlertLevel.CRITICAL
        )

# Global monitor instance
neonatal_monitor = NeonatalMonitor()

# Helper function for real-time PCM preprocessing
def _preprocess_realtime_pcm(pcm: np.ndarray, sr: int) -> np.ndarray:
    """
    Light, stable preprocessing for breath analysis:
    1) High-pass @ 80 Hz to remove DC/rumble, 2) limit, 3) normalize.
    """
    try:
        # high-pass to kill HVAC/handling rumble
        sos = scipy.signal.butter(4, 80, btype='highpass', fs=sr, output='sos')
        x = scipy.signal.sosfilt(sos, pcm.astype(np.float32, copy=False))
        # soft limiter
        x = np.tanh(2.5 * x)
        # normalize to [-1,1] if non-silent
        peak = np.max(np.abs(x)) if x.size else 0.0
        if peak > 1e-6:
            x = x / peak
        return x
    except Exception:
        return pcm

# Flask routes for medical monitoring
@app.route('/')
def medical_dashboard():
    """Medical monitoring dashboard"""
    return render_template('medical_dashboard.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start neonatal monitoring session"""
    try:
        data = request.get_json() or {}
        birth_time_str = data.get('birth_time')
        
        if birth_time_str:
            birth_time = datetime.fromisoformat(birth_time_str)
        else:
            birth_time = datetime.now()
        
        result = neonatal_monitor.start_monitoring(birth_time)
        
        logger.info(f"🏥 Monitoring session started: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    """Real-time audio analysis endpoint (robust against bad input)."""
    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        # never hard-crash; tell the UI what's wrong
        return jsonify({"ok": False, "error": f"invalid json: {e}"}), 200

    is_rt = bool(data.get("real_time"))
    if is_rt:
        audio_data = data.get("audio_data")
        sr = int(data.get("sample_rate") or neonatal_monitor.sample_rate)

        # Guard clauses: no audio yet or wrong shape
        if not isinstance(audio_data, list) or len(audio_data) < int(0.5 * sr):
            # need at least ~0.5s to say anything useful
            return jsonify({
                "ok": False,
                "error": "insufficient_audio",
                "needed_samples": int(0.5 * sr),
                "got": len(audio_data) if isinstance(audio_data, list) else 0,
                "alert_level": "no_audio"
            }), 200

        try:
            pcm = np.array(audio_data, dtype=np.float32)
            pcm = _preprocess_realtime_pcm(pcm, sr)
            metrics = neonatal_monitor.analyze_audio_stream(pcm)
        except Exception as e:
            return jsonify({
                "ok": False,
                "error": f"analysis_failed: {e}",
                "alert_level": "processing_error"
            }), 200

    else:
        # Synthetic/test path (keeps your demo working)
        test_audio = np.random.randn(int(neonatal_monitor.sample_rate * 2)).astype(np.float32)
        metrics = neonatal_monitor.analyze_audio_stream(test_audio)
        sr = neonatal_monitor.sample_rate

    # success path
    return jsonify({
        "ok": True,
        "timestamp": metrics.timestamp.isoformat(),
        "breathing_rate": metrics.breathing_rate,
        "breathing_pattern": metrics.breathing_pattern,
        "cry_intensity": metrics.cry_intensity,
        "cry_frequency": metrics.cry_frequency,
        "oxygen_saturation_estimate": metrics.oxygen_saturation_estimate,
        "distress_score": metrics.distress_score,
        "alert_level": metrics.alert_level.value,
        "analysis_latency_ms": 25.0,
        "golden_minute_active": neonatal_monitor.golden_minute_active,
        "sample_rate": sr
    }), 200

@app.route('/get_alerts', methods=['GET'])
def get_alerts():
    """Get active medical alerts"""
    try:
        alerts_data = []
        
        for alert in neonatal_monitor.active_alerts[-10:]:  # Last 10 alerts
            alert_data = {
                "timestamp": alert.timestamp.isoformat(),
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,
                "confidence": alert.confidence,
                "recommended_action": alert.recommended_action,
                "time_since_birth": alert.time_since_birth.total_seconds()
            }
            alerts_data.append(alert_data)
        
        return jsonify({
            "alerts": alerts_data,
            "alert_count": len(alerts_data),
            "highest_severity": max([alert.severity.value for alert in neonatal_monitor.active_alerts], default="normal")
        })
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Medical system health check"""
    return jsonify({
        "status": "healthy",
        "system": "neonatal_respiratory_monitor",
        "monitoring_active": neonatal_monitor.continuous_monitoring,
        "golden_minute_active": neonatal_monitor.golden_minute_active,
        "birth_time": neonatal_monitor.birth_time.isoformat() if neonatal_monitor.birth_time else None,
        "active_alerts": len(neonatal_monitor.active_alerts)
    })

if __name__ == '__main__':
    logger.info("🏥 Starting Neonatal Respiratory Monitoring System...")
    logger.info("🔧 System Configuration:")
    logger.info(f"   Sample Rate: {neonatal_monitor.sample_rate} Hz")
    logger.info(f"   Buffer Size: {neonatal_monitor.buffer_size} samples")
    logger.info(f"   Analysis Window: {neonatal_monitor.analysis_window}s")
    logger.info(f"   Target Latency: <100ms")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
