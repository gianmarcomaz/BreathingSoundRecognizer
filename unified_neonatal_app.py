#!/usr/bin/env python3
"""
Unified Neonatal Monitoring Streamlit App
Combining Breathing Sound Recognition + Image & Vitals Analysis
"""

import os
import json
import logging
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Optional
import streamlit as st
from dotenv import load_dotenv
import requests

# Audio processing imports
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Microphone input will be disabled.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import scipy.signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Neonatal Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import medical analysis functions
from medical_analysis import (
    analyze_neonatal_audio,
    preprocess_real_world_audio,
    create_error_response
)

# Initialize session state
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'api_status' not in st.session_state:
    st.session_state.api_status = {}

# ============================================================================
# SPONSOR API INITIALIZATION
# ============================================================================

def initialize_sponsor_apis():
    """Initialize and validate all sponsor APIs"""
    status = {
        'lava': False,
        'phoenix': False,
        'livekit': False,
        'vapi': False
    }
    
    # Phoenix/Arize
    try:
        phoenix_api_key = os.getenv("PHOENIX_API_KEY")
        phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        
        if phoenix_api_key and phoenix_endpoint:
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"
            logger.info("✅ Phoenix initialized")
            status['phoenix'] = True
        else:
            logger.warning("⚠️ Phoenix not configured")
    except Exception as e:
        logger.error(f"❌ Phoenix initialization failed: {e}")
    
    # Lava Payments
    try:
        if os.getenv("LAVA_SECRET_KEY"):
            logger.info("✅ Lava initialized")
            status['lava'] = True
        else:
            logger.warning("⚠️ Lava not configured")
    except Exception as e:
        logger.error(f"❌ Lava initialization failed: {e}")
    
    # LiveKit
    try:
        if os.getenv("LIVEKIT_API_KEY") and os.getenv("LIVEKIT_API_SECRET"):
            logger.info("✅ LiveKit initialized")
            status['livekit'] = True
        else:
            logger.warning("⚠️ LiveKit not configured")
    except Exception as e:
        logger.error(f"❌ LiveKit initialization failed: {e}")
    
    # Vapi
    try:
        if os.getenv("VAPI_SECRET_TOKEN") and os.getenv("VAPI_SECRET_TOKEN") != "your_vapi_secret_token_here":
            logger.info("✅ Vapi initialized")
            status['vapi'] = True
        else:
            logger.warning("⚠️ Vapi not configured")
    except Exception as e:
        logger.error(f"❌ Vapi initialization failed: {e}")
    
    return status

# Initialize APIs at startup
if 'api_status' not in st.session_state or not st.session_state.api_status:
    st.session_state.api_status = initialize_sponsor_apis()

# ============================================================================
# IMAGE & VITALS ANALYSIS
# ============================================================================

def preprocess_image(image):
    """Preprocess image: auto-white-balance, gamma correction, MSRCR"""
    try:
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Auto-white-balance using gray world
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        gray_mean = np.mean(gray)
        
        # Split channels and apply white balance
        b, g, r = cv2.split(processed)
        
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)
        
        # Apply gray world assumption
        if b_mean > 0:
            b = b * (gray_mean / b_mean)
        if g_mean > 0:
            g = g * (gray_mean / g_mean)
        if r_mean > 0:
            r = r * (gray_mean / r_mean)
        
        # Merge channels back
        processed = cv2.merge([b, g, r])
        
        # Clip values to valid range
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        # Gamma correction
        gamma = 1.2
        lookup = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed = cv2.LUT(processed, lookup)
        
        # Simple MSRCR (Multi-Scale Retinex with Color Restoration) - simplified
        image_float = processed.astype(np.float32) / 255.0
        intensity = np.mean(image_float, axis=2)
        
        # Simulate MSRCR with tone mapping
        enhanced = image_float + 0.1 * (intensity[:, :, np.newaxis] - image_float)
        processed = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        return processed
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return image

def segment_skin_lab_kmeans(image):
    """K-means skin segmentation in Lab color space"""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_reshaped = lab.reshape((-1, 3))
        lab_reshaped = np.float32(lab_reshaped)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 3
        _, labels, centers = cv2.kmeans(lab_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find skin cluster (typically the one with highest L* and moderate a*)
        skin_cluster = np.argmax([centers[i][0] for i in range(k)])
        
        mask = labels.flatten() == skin_cluster
        mask = mask.reshape(image.shape[:2]).astype(np.uint8) * 255
        
        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        segmented = cv2.bitwise_and(image, mask_rgb)
        
        return mask, segmented
    except Exception as e:
        logger.error(f"Skin segmentation failed: {e}")
        return None, image

def extract_color_statistics(image, mask):
    """Extract L*, a*, b* color statistics from skin region"""
    try:
        if mask is None:
            return {}
        
        # Convert to Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract skin pixels only
        skin_pixels = lab[mask > 0]
        
        if len(skin_pixels) == 0:
            return {}
        
        # Calculate statistics
        l_star = np.mean(skin_pixels[:, 0])  # Lightness
        a_star = np.mean(skin_pixels[:, 1])    # Green-red axis
        b_star = np.mean(skin_pixels[:, 2])    # Blue-yellow axis
        
        # Chroma and hue
        chroma = np.sqrt(a_star**2 + b_star**2)
        hue = np.arctan2(b_star, a_star) * 180 / np.pi
        
        return {
            'L': float(l_star),
            'a': float(a_star),
            'b': float(b_star),
            'C': float(chroma),
            'h': float(hue),
            'skin_pixel_count': int(np.sum(mask > 0))
        }
    except Exception as e:
        logger.error(f"Color extraction failed: {e}")
        return {}

def calculate_condition_probabilities(color_stats):
    """Calculate jaundice, cyanosis, pallor probabilities from color stats"""
    try:
        jaundice_prob = 0.0
        cyanosis_prob = 0.0
        pallor_prob = 0.0
        
        if not color_stats:
            return jaundice_prob, cyanosis_prob, pallor_prob
        
        b_star = color_stats.get('b', 0)
        l_star = color_stats.get('L', 0)
        chroma = color_stats.get('C', 0)
        
        # Jaundice (yellow tint): high b* value
        if b_star > 10:  # Yellow tint
            jaundice_prob = min(100, (b_star - 10) * 5)
        
        # Cyanosis (blue tint): low b* value
        if b_star < -5:  # Blue tint
            cyanosis_prob = min(100, abs(b_star + 5) * 5)
        
        # Pallor/Asphyxia (low saturation/lightness)
        if l_star < 60 or chroma < 15:  # Low saturation
            pallor_prob = min(100, 50 - chroma)
        
        return jaundice_prob, cyanosis_prob, pallor_prob
    except Exception as e:
        logger.error(f"Probability calculation failed: {e}")
        return 0.0, 0.0, 0.0

def calculate_vitals_risk(vitals):
    """Calculate Risk of Resuscitation (RoR) from vitals"""
    try:
        risk_score = 0
        
        # Respiratory rate
        rr = vitals.get('respiratory_rate', 30)
        if rr < 20 or rr > 80:
            risk_score += 40
        elif rr < 30 or rr > 60:
            risk_score += 20
        
        # Heart rate
        hr = vitals.get('heart_rate', 120)
        if hr < 60 or hr > 200:
            risk_score += 30
        elif hr < 100 or hr > 160:
            risk_score += 15
        
        # Temperature
        temp = vitals.get('temperature', 37.0)
        if temp < 36.0 or temp > 38.0:
            risk_score += 20
        
        # SpO2
        spo2 = vitals.get('spo2', 95)
        if spo2 < 85:
            risk_score += 50
        elif spo2 < 90:
            risk_score += 25
        
        # Capillary refill
        if vitals.get('capillary_refill', 'normal') != 'normal':
            risk_score += 15
        
        # Additional indicators
        if vitals.get('retractions', False):
            risk_score += 20
        if vitals.get('lethargy', False):
            risk_score += 15
        
        return min(100, risk_score)
    except Exception as e:
        logger.error(f"Vitals risk calculation failed: {e}")
        return 0

def calculate_final_ror(image_probabilities, vitals_risk):
    """Calculate final Risk of Resuscitation by combining image and vitals"""
    try:
        # Combine image-based probabilities
        image_risk = (image_probabilities['jaundice'] * 0.3 +
                     image_probabilities['cyanosis'] * 0.4 +
                     image_probabilities['pallor'] * 0.3)
        
        # Combine with vitals risk
        combined_ror = (image_risk * 0.4) + (vitals_risk * 0.6)
        
        return min(100, combined_ror)
    except Exception as e:
        logger.error(f"Final RoR calculation failed: {e}")
        return 0

def generate_medical_advice(color_stats, image_probs, vitals=None, final_ror=None):
    """Generate medical advice based on analysis results"""
    advice_items = []
    
    # Analyze jaundice
    if image_probs.get('jaundice', 0) > 50:
        advice_items.append({
            'condition': 'Jaundice',
            'priority': 'high',
            'advice': [
                "Monitor bilirubin levels closely",
                "Ensure adequate feeding (breastfeeding or formula every 2-3 hours)",
                "Expose to indirect sunlight if recommended by healthcare provider",
                "Consider phototherapy if bilirubin levels are rising",
                "Watch for signs of lethargy, poor feeding, or arching of the back"
            ]
        })
    elif image_probs.get('jaundice', 0) > 20:
        advice_items.append({
            'condition': 'Mild Jaundice',
            'priority': 'medium',
            'advice': [
                "Continue monitoring skin color",
                "Ensure frequent feeding to help clear bilirubin",
                "Monitor for worsening symptoms",
                "Consult healthcare provider if yellowing increases"
            ]
        })
    
    # Analyze cyanosis
    if image_probs.get('cyanosis', 0) > 50:
        advice_items.append({
            'condition': 'Cyanosis',
            'priority': 'critical',
            'advice': [
                "🚨 SEEK IMMEDIATE MEDICAL ATTENTION",
                "Check oxygen saturation immediately",
                "Ensure clear airway - check for obstruction",
                "Administer oxygen if available and trained to do so",
                "Monitor respiratory rate and effort",
                "Prepare for potential CPR if breathing stops"
            ]
        })
    elif image_probs.get('cyanosis', 0) > 20:
        advice_items.append({
            'condition': 'Possible Cyanosis',
            'priority': 'high',
            'advice': [
                "Monitor breathing and oxygen levels closely",
                "Check for signs of respiratory distress",
                "Ensure proper positioning to maintain airway",
                "Seek medical evaluation if not improving"
            ]
        })
    
    # Analyze pallor
    if image_probs.get('pallor', 0) > 50:
        advice_items.append({
            'condition': 'Pallor/Asphyxia',
            'priority': 'critical',
            'advice': [
                "🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED",
                "Check for responsiveness and breathing",
                "If not breathing, begin CPR immediately",
                "Call emergency services",
                "Keep baby warm",
                "Check for signs of circulation"
            ]
        })
    elif image_probs.get('pallor', 0) > 20:
        advice_items.append({
            'condition': 'Possible Pallor',
            'priority': 'high',
            'advice': [
                "Monitor closely for changes in condition",
                "Check temperature and ensure baby is warm",
                "Assess feeding and responsiveness",
                "Seek immediate evaluation if baby becomes unresponsive"
            ]
        })
    
    # General advice if no major concerns
    if not advice_items:
        advice_items.append({
            'condition': 'Normal Appearance',
            'priority': 'low',
            'advice': [
                "Continue routine monitoring",
                "Monitor feeding, sleeping, and breathing patterns",
                "Watch for any changes in skin color",
                "Maintain regular check-ups with healthcare provider"
            ]
        })
    
    # Add vitals-based advice if provided
    if vitals:
        vitals_advice = []
        
        # Respiratory rate
        rr = vitals.get('respiratory_rate', 30)
        if rr < 20 or rr > 80:
            vitals_advice.append("⚠️ Abnormal respiratory rate - seek immediate medical attention")
        elif rr < 30 or rr > 60:
            vitals_advice.append("Monitor breathing patterns closely")
        
        # Heart rate
        hr = vitals.get('heart_rate', 120)
        if hr < 60 or hr > 200:
            vitals_advice.append("⚠️ Abnormal heart rate - seek medical attention")
        elif hr < 100 or hr > 160:
            vitals_advice.append("Monitor heart rate periodically")
        
        # SpO2
        spo2 = vitals.get('spo2', 95)
        if spo2 < 85:
            vitals_advice.append("🚨 CRITICAL: Low oxygen saturation - seek immediate medical attention")
        elif spo2 < 90:
            vitals_advice.append("⚠️ Low oxygen saturation - monitor closely")
        
        # Temperature
        temp = vitals.get('temperature', 37.0)
        if temp < 36.0 or temp > 38.0:
            vitals_advice.append("⚠️ Abnormal temperature - check for signs of infection or hypothermia")
        
        if vitals_advice:
            advice_items.insert(0, {
                'condition': 'Vital Signs',
                'priority': 'high',
                'advice': vitals_advice
            })
    
    # Add overall RoR recommendation
    if final_ror is not None:
        if final_ror >= 60:
            overall_advice = {
                'condition': 'High Risk Assessment',
                'priority': 'critical',
                'advice': [
                    "🚨 HIGH RISK - Seek immediate medical evaluation",
                    "Do not delay seeking medical attention",
                    "Monitor continuously until medical help arrives",
                    "Prepare emergency contact information"
                ]
            }
            advice_items.insert(0, overall_advice)
        elif final_ror >= 30:
            overall_advice = {
                'condition': 'Moderate Risk Assessment',
                'priority': 'medium',
                'advice': [
                    "Monitor baby closely for any changes",
                    "Consider consulting healthcare provider within 24 hours",
                    "Be prepared to seek immediate care if symptoms worsen"
                ]
            }
            advice_items.insert(0, overall_advice)
    
    return advice_items

# ============================================================================
# AUDIO PROCESSING HELPERS
# ============================================================================

def generate_synthetic_audio(duration=2.0, condition='healthy'):
    """Generate synthetic audio for testing"""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    if condition == 'healthy':
        # Healthy breathing (40-50 bpm = 0.67-0.83 Hz)
        breathing_rate = np.random.uniform(40, 50)
        breathing_freq = breathing_rate / 60.0
        audio = np.sin(2 * np.pi * breathing_freq * t) * 0.3
        audio += np.random.normal(0, 0.05, len(audio))
    elif condition == 'asphyxia':
        # Irregular breathing (10-20 bpm)
        breathing_rate = np.random.uniform(10, 20)
        breathing_freq = breathing_rate / 60.0
        audio = np.sin(2 * np.pi * breathing_freq * t) * 0.2
        audio += 0.1 * np.sin(2 * np.pi * breathing_freq * 2 * t)
        audio += np.random.normal(0, 0.05, len(audio))
    else:
        audio = np.random.normal(0, 0.1, len(audio))
    
    return audio

def capture_audio(duration=2.0):
    """Capture audio from microphone"""
    if not SOUNDDEVICE_AVAILABLE:
        return generate_synthetic_audio(duration, 'healthy')
    
    try:
        sample_rate = 44100
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        return audio.flatten()
    except Exception as e:
        logger.warning(f"Microphone capture failed: {e}")
        return generate_synthetic_audio(duration, 'healthy')

# ============================================================================
# STREAMLIT PAGES
# ============================================================================

def breathing_sound_recognition():
    """Breathing Sound Recognition View"""
    st.header("🫁 Breathing Sound Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Real-time Audio Analysis")
        
        # Monitor controls
        if st.button("Start Monitoring" if not st.session_state.monitoring_active else "Stop Monitoring"):
            st.session_state.monitoring_active = not st.session_state.monitoring_active
            st.rerun()
        
        if st.session_state.monitoring_active:
            st.success("🔴 Monitoring Active")
            
            # Capture and analyze audio
            with st.spinner("Capturing audio..."):
                audio_data = capture_audio(duration=2.0)
                
                # Analyze audio
                metrics = analyze_neonatal_audio(audio_data)
            
            # Display metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric(
                    "Breathing Rate",
                    f"{metrics['breathing_rate']:.1f} bpm",
                    delta=f"Pattern: {metrics['breathing_pattern']}"
                )
            
            with col_b:
                st.metric(
                    "Cry Intensity",
                    f"{metrics['cry_intensity']:.2f} RMS"
                )
            
            with col_c:
                st.metric(
                    "Distress Score",
                    f"{metrics['distress_score']:.2f} (0-1)"
                )
            
            with col_d:
                alert_color = {
                    'normal': 'normal',
                    'watch': 'off',
                    'warning': 'normal',
                    'critical': 'inverse',
                    'emergency': 'inverse'
                }.get(metrics['alert_level'], 'off')
                
                st.metric(
                    "Alert Level",
                    metrics['alert_level'].upper()
                )
            
            # Additional metrics
            with st.expander("Detailed Analysis"):
                col_x, col_y = st.columns(2)
                
                with col_x:
                    st.write(f"**Cry Frequency:** {metrics['cry_frequency']:.1f} Hz")
                    st.write(f"**Cry Quality:** {metrics['cry_quality']}")
                    st.write(f"**Signal Quality:** {metrics['signal_quality']}")
                
                with col_y:
                    st.write(f"**O₂ Estimate:** {metrics['oxygen_saturation_estimate']:.1f}%")
                    st.write(f"**Condition:** {metrics['medical_condition']}")
                    st.write(f"**VAD Activity:** {metrics['vad_activity']:.2f}")
            
            # Clinical recommendations
            st.info(f"💡 **Recommendation:** {metrics['clinical_recommendations']}")
            
            # Simulated data warning
            if not SOUNDDEVICE_AVAILABLE or metrics.get('data_source') == 'synthetic':
                st.warning("⚠️ Simulated data used - configure microphone for real-time analysis")
    
    with col2:
        st.subheader("🧪 Diagnostic Test Cases")
        st.caption("Play audio samples for different conditions")
        
        # Store audio samples in session state
        if 'test_audio_samples' not in st.session_state:
            st.session_state.test_audio_samples = {}
        
        # Define diagnostic conditions with descriptions
        diagnostics = {
            'healthy': {
                'name': '✅ Healthy Baby',
                'icon': '✅',
                'breathing_rate': '30-60 bpm',
                'pattern': 'Regular, smooth breaths',
                'cry': 'Normal frequency (350-450 Hz), moderate intensity',
                'description': 'Normal breathing pattern with regular rhythm, healthy cry patterns',
                'severity': 'normal'
            },
            'asphyxia': {
                'name': '🚨 Asphyxia',
                'icon': '🚨',
                'breathing_rate': '5-25 bpm',
                'pattern': 'Irregular, gasping breaths',
                'cry': 'Weak cry (250-300 Hz), very low intensity',
                'description': 'Inadequate oxygen delivery - irregular breathing, weak or absent cry',
                'severity': 'critical'
            },
            'jaundice': {
                'name': '🟡 Jaundice',
                'icon': '🟡',
                'breathing_rate': '30-60 bpm',
                'pattern': 'Normal breathing',
                'cry': 'Weak, monotone cry (220-280 Hz), low intensity',
                'description': 'High bilirubin levels - lethargy, weak monotone cry, normal breathing',
                'severity': 'moderate'
            },
            'cyanosis': {
                'name': '🔵 Cyanosis',
                'icon': '🔵',
                'breathing_rate': '70-110 bpm',
                'pattern': 'Rapid, shallow, labored',
                'cry': 'High-pitched, shrill cry (450-500 Hz)',
                'description': 'Low oxygen saturation - rapid breathing, high-pitched distressed cry',
                'severity': 'critical'
            }
        }
        
        # Generate and display test cases
        for condition_key, info in diagnostics.items():
            st.markdown("---")
            
            # Create columns for button and info button
            btn_col, info_col = st.columns([3, 1])
            
            with btn_col:
                if st.button(f"{info['icon']} {info['name']}", key=f"test_{condition_key}"):
                    with st.spinner(f"Generating {info['name']} audio..."):
                        # Generate audio
                        audio = generate_synthetic_audio(duration=3.0, condition=condition_key)
                        # Store in session state
                        st.session_state.test_audio_samples[condition_key] = audio
                        # Analyze
                        st.session_state[f'metrics_{condition_key}'] = analyze_neonatal_audio(audio)
                        # Auto-play audio by setting flag
                        st.session_state[f'play_{condition_key}'] = True
                        st.rerun()
            
            with info_col:
                if st.button("ℹ️", key=f"info_{condition_key}"):
                    st.session_state[f'show_info_{condition_key}'] = True
            
            # Show info popup
            if st.session_state.get(f'show_info_{condition_key}', False):
                with st.expander(f"📋 {info['name']} - Condition Details", expanded=True):
                    st.write(f"**Severity:** {info['severity'].upper()}")
                    st.write(f"**Breathing Rate:** {info['breathing_rate']}")
                    st.write(f"**Breathing Pattern:** {info['pattern']}")
                    st.write(f"**Cry Characteristics:** {info['cry']}")
                    st.markdown(f"**Description:** {info['description']}")
                    
                    # Add clinical details
                    st.markdown("### Clinical Significance")
                    if condition_key == 'healthy':
                        st.success("✅ Normal physiological patterns - continue routine monitoring")
                    elif condition_key == 'asphyxia':
                        st.error("🚨 CRITICAL: Immediate medical intervention required - administer oxygen, ensure airway")
                    elif condition_key == 'jaundice':
                        st.warning("⚠️ Monitor bilirubin levels, ensure adequate feeding")
                    elif condition_key == 'cyanosis':
                        st.error("🚨 CRITICAL: Low oxygen - check SpO2, consider supplemental oxygen")
            
            # Play audio if generated
            if condition_key in st.session_state.test_audio_samples:
                audio = st.session_state.test_audio_samples[condition_key]
                
                # Display metrics
                if f'metrics_{condition_key}' in st.session_state:
                    metrics = st.session_state[f'metrics_{condition_key}']
                    
                    # Metrics display
                    st.caption(f"**Breathing Rate:** {metrics['breathing_rate']:.1f} bpm")
                    st.caption(f"**Pattern:** {metrics['breathing_pattern']}")
                    st.caption(f"**Alert Level:** {metrics['alert_level'].upper()}")
                    
                    # Recommendation
                    if metrics['alert_level'] in ['critical', 'emergency']:
                        st.error(f"🚨 {metrics['clinical_recommendations']}")
                    elif metrics['alert_level'] == 'warning':
                        st.warning(f"⚠️ {metrics['clinical_recommendations']}")
                    else:
                        st.success(f"✅ {metrics['clinical_recommendations']}")
                
                # Audio player
                st.audio(audio, sample_rate=44100, format='audio/wav')
        
        st.markdown("---")
        st.subheader("API Status")
        
        for api, status in st.session_state.api_status.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {api.upper()}")

def image_vitals_analysis():
    """Image & Vitals Analysis View"""
    st.header("🖼️ Image & Vitals Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Baby Photo")
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            with st.spinner("Preprocessing image..."):
                processed_image = preprocess_image(image.copy())
            
            # Segment skin
            with st.spinner("Segmenting skin region..."):
                mask, segmented = segment_skin_lab_kmeans(image.copy())
            
            if mask is not None:
                # Extract color statistics
                color_stats = extract_color_statistics(image, mask)
                
                if color_stats:
                    # Display images
                    st.image(processed_image, caption="Preprocessed Image", use_container_width=True)
                    
                    # Create overlay mask for display
                    if len(mask.shape) == 2:
                        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    else:
                        mask_display = mask.copy()
                    
                    # Ensure mask_display is the same shape and dtype as image
                    if mask_display.shape != image.shape:
                        mask_display = cv2.resize(mask_display, (image.shape[1], image.shape[0]))
                    
                    # Normalize mask to 0-255 range and ensure uint8 dtype
                    mask_normalized = (mask_display.astype(np.float32) * 0.3).astype(np.uint8)
                    
                    # Create overlay with proper type handling
                    overlay = cv2.addWeighted(image.astype(np.float32), 0.7, 
                                             mask_normalized.astype(np.float32), 0.3, 0).astype(np.uint8)
                    
                    st.image(overlay, caption="Skin Overlay", use_container_width=True)
                    
                    # Display color statistics
                    st.subheader("Color Statistics")
                    col_l, col_a, col_b, col_c = st.columns(4)
                    
                    with col_l:
                        st.metric("L*", f"{color_stats['L']:.1f}")
                    with col_a:
                        st.metric("a*", f"{color_stats['a']:.1f}")
                    with col_b:
                        st.metric("b*", f"{color_stats['b']:.1f}")
                    with col_c:
                        st.metric("C*", f"{color_stats['C']:.1f}")
                    
                    # Calculate probabilities
                    jaundice_prob, cyanosis_prob, pallor_prob = calculate_condition_probabilities(color_stats)
                    
                    st.subheader("Condition Probabilities")
                    col_j, col_cy, col_p = st.columns(3)
                    
                    with col_j:
                        st.metric("Jaundice", f"{jaundice_prob:.1f}%")
                    
                    with col_cy:
                        st.metric("Cyanosis", f"{cyanosis_prob:.1f}%")
                    
                    with col_p:
                        st.metric("Pallor/Asphyxia", f"{pallor_prob:.1f}%")
                    
                    # Store for RoR calculation
                    st.session_state.image_probs = {
                        'jaundice': jaundice_prob,
                        'cyanosis': cyanosis_prob,
                        'pallor': pallor_prob
                    }
                    
                    # Generate and display medical advice
                    st.subheader("💡 Medical Advice & Recommendations")
                    
                    # Generate advice based on image analysis
                    advice_items = generate_medical_advice(
                        color_stats, 
                        st.session_state.image_probs
                    )
                    
                    # Display each advice item
                    for item in advice_items:
                        priority = item.get('priority', 'low')
                        condition = item.get('condition', 'Unknown')
                        advice_list = item.get('advice', [])
                        
                        # Choose display style based on priority
                        if priority == 'critical':
                            with st.expander(f"🚨 {condition} - CRITICAL", expanded=True):
                                for adv in advice_list:
                                    st.write(f"• {adv}")
                        elif priority == 'high':
                            with st.expander(f"⚠️ {condition} - HIGH PRIORITY", expanded=True):
                                for adv in advice_list:
                                    st.write(f"• {adv}")
                        elif priority == 'medium':
                            with st.expander(f"📋 {condition} - MONITOR"):
                                for adv in advice_list:
                                    st.write(f"• {adv}")
                        else:
                            with st.expander(f"✅ {condition}"):
                                for adv in advice_list:
                                    st.write(f"• {adv}")
                else:
                    st.error("Could not isolate skin region")
            else:
                st.error("Skin segmentation failed")
    
    with col2:
        st.subheader("Vital Signs Entry")
        
        # Vital signs form
        with st.form("vitals_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                respiratory_rate = st.number_input("Respiratory Rate", min_value=0, max_value=150, value=30, step=1)
                heart_rate = st.number_input("Heart Rate", min_value=0, max_value=300, value=120, step=1)
                temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=42.0, value=37.0, step=0.1)
            
            with col_b:
                spo2 = st.number_input("SpO₂ (%)", min_value=0, max_value=100, value=95, step=1)
                capillary_refill = st.selectbox("Capillary Refill", ["normal", "delayed", "absent"])
                age = st.number_input("Age (hours)", min_value=0, max_value=168, value=0, step=1)
            
            st.markdown("---")
            retractions = st.checkbox("Retractions present")
            lethargy = st.checkbox("Lethargy present")
            feeding = st.selectbox("Feeding Status", ["normal", "reduced", "absent"])
            
            submitted = st.form_submit_button("Analyze Vitals")
            
            if submitted:
                # Prepare vitals dict
                vitals = {
                    'respiratory_rate': respiratory_rate,
                    'heart_rate': heart_rate,
                    'temperature': temperature,
                    'spo2': spo2,
                    'capillary_refill': capillary_refill,
                    'age': age,
                    'retractions': retractions,
                    'lethargy': lethargy,
                    'feeding': feeding
                }
                
                # Calculate RoR
                vitals_risk = calculate_vitals_risk(vitals)
                
                # Combine with image probabilities if available
                if 'image_probs' in st.session_state:
                    final_ror = calculate_final_ror(st.session_state.image_probs, vitals_risk)
                else:
                    final_ror = vitals_risk
                
                # Display RoR
                st.subheader("Risk of Resuscitation (RoR)")
                
                if final_ror < 30:
                    ror_level = "LOW"
                    ror_color = "normal"
                elif final_ror < 60:
                    ror_level = "MODERATE"
                    ror_color = "normal"
                else:
                    ror_level = "HIGH"
                    ror_color = "inverse"
                
                st.metric("RoR Score", f"{final_ror:.0f}/100", delta=ror_level, delta_color=ror_color)
                
                # Recommendation
                if ror_level == "HIGH":
                    st.error(f"🚨 HIGH RoR ({final_ror:.0f}/100) - Seek urgent evaluation")
                elif ror_level == "MODERATE":
                    st.warning(f"⚠️ MODERATE RoR ({final_ror:.0f}/100) - Monitor closely")
                else:
                    st.success(f"✅ LOW RoR ({final_ror:.0f}/100) - Continue routine monitoring")
                
                # Generate comprehensive medical advice based on vitals and image analysis
                st.markdown("---")
                st.subheader("💡 Comprehensive Medical Advice")
                
                # Prepare image probabilities if available
                image_probs = st.session_state.image_probs if 'image_probs' in st.session_state else {}
                
                # Generate comprehensive advice
                comprehensive_advice = generate_medical_advice(
                    color_stats={},
                    image_probs=image_probs,
                    vitals=vitals,
                    final_ror=final_ror
                )
                
                # Display each advice item
                for item in comprehensive_advice:
                    priority = item.get('priority', 'low')
                    condition = item.get('condition', 'Unknown')
                    advice_list = item.get('advice', [])
                    
                    # Choose display style based on priority
                    if priority == 'critical':
                        with st.expander(f"🚨 {condition} - CRITICAL", expanded=True):
                            for adv in advice_list:
                                st.write(f"• {adv}")
                    elif priority == 'high':
                        with st.expander(f"⚠️ {condition} - HIGH PRIORITY", expanded=True):
                            for adv in advice_list:
                                st.write(f"• {adv}")
                    elif priority == 'medium':
                        with st.expander(f"📋 {condition} - MONITOR"):
                            for adv in advice_list:
                                st.write(f"• {adv}")
                    else:
                        with st.expander(f"✅ {condition}"):
                            for adv in advice_list:
                                st.write(f"• {adv}")
                
                # Download report
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'vitals': vitals,
                    'vitals_risk': vitals_risk,
                    'image_probabilities': st.session_state.image_probs if 'image_probs' in st.session_state else {},
                    'ror_score': final_ror,
                    'ror_level': ror_level,
                    'medical_advice': comprehensive_advice
                }
                
                st.download_button(
                    "Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #1e1e2e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #00ff88;
    }
    .main-header {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        border: 3px solid #00ff88;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #00ff88; text-align: center; margin: 0;">🏥 Unified Neonatal Monitoring</h1>
        <p style="color: white; text-align: center; margin-top: 0.5rem;">Breathing Sound Recognition • Image & Vitals Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Breathing Sound Recognition", "Image & Vitals Analysis"]
    )
    
    # Sidebar API status
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Status")
    for api, status in st.session_state.api_status.items():
        status_icon = "✅" if status else "❌"
        color = "#00ff88" if status else "#ff4757"
        st.sidebar.markdown(f'<span style="color: {color};">{status_icon}</span> {api.upper()}', unsafe_allow_html=True)
    
    # Display selected page
    if page == "Breathing Sound Recognition":
        breathing_sound_recognition()
    elif page == "Image & Vitals Analysis":
        image_vitals_analysis()

if __name__ == "__main__":
    main()
