#!/usr/bin/env python3
"""
Medical Audio Datasets for Neonatal Monitoring
Generates synthetic medical audio patterns for testing
"""

import numpy as np
import random
from datetime import datetime

class MedicalDatasets:
    """Generate synthetic medical audio patterns for testing"""
    
    def __init__(self):
        self.sample_rate = 44100
        
    def generate_medical_audio(self, condition, severity, duration):
        """Generate synthetic medical audio based on condition and severity"""
        try:
            samples = int(duration * self.sample_rate)
            audio = np.zeros(samples)
            
            if condition == 'healthy':
                audio, metadata = self._generate_healthy_audio(samples, severity)
            elif condition == 'asphyxia':
                audio, metadata = self._generate_asphyxia_audio(samples, severity)
            elif condition == 'jaundice':
                audio, metadata = self._generate_jaundice_audio(samples, severity)
            elif condition == 'cyanosis':
                audio, metadata = self._generate_cyanosis_audio(samples, severity)
            else:
                audio, metadata = self._generate_healthy_audio(samples, 'normal')
            
            return audio, metadata
            
        except Exception as e:
            print(f"Error generating medical audio: {e}")
            # Return fallback audio
            samples = int(duration * self.sample_rate)
            audio = np.random.normal(0, 0.1, samples)
            metadata = {
                "condition": "error",
                "severity": "unknown",
                "description": f"Error generating audio: {str(e)}"
            }
            return audio, metadata
    
    def _generate_healthy_audio(self, samples, severity):
        """Generate healthy newborn audio patterns"""
        audio = np.zeros(samples)
        
        # Add regular breathing pattern (0.5-1.5 Hz)
        breathing_freq = 0.8 + random.uniform(-0.2, 0.2)
        breathing_amp = 0.3
        
        # Add normal cry pattern (300-500 Hz)
        cry_freq = 400 + random.uniform(-50, 50)
        cry_amp = 0.4
        
        t = np.linspace(0, samples / self.sample_rate, samples)
        
        # Breathing component
        breathing = breathing_amp * np.sin(2 * np.pi * breathing_freq * t)
        
        # Cry component (intermittent)
        cry_mask = np.random.random(samples) < 0.3  # 30% chance of crying
        cry = cry_amp * np.sin(2 * np.pi * cry_freq * t) * cry_mask
        
        # Add some harmonics for realistic cry
        cry += 0.2 * np.sin(2 * np.pi * cry_freq * 2 * t) * cry_mask
        cry += 0.1 * np.sin(2 * np.pi * cry_freq * 3 * t) * cry_mask
        
        audio = breathing + cry
        
        # Add slight noise
        noise = np.random.normal(0, 0.05, samples)
        audio += noise
        
        metadata = {
            "condition": "healthy",
            "severity": severity,
            "breathing_rate": breathing_freq * 60,
            "cry_frequency": cry_freq,
            "description": "Normal healthy newborn patterns"
        }
        
        return audio, metadata
    
    def _generate_asphyxia_audio(self, samples, severity):
        """Generate asphyxia audio patterns"""
        audio = np.zeros(samples)
        
        t = np.linspace(0, samples / self.sample_rate, samples)
        
        if severity == 'mild':
            # Irregular breathing, weak cry - 15-25 bpm
            base_freq = 0.33  # ~20 bpm base
            breathing_amp = 0.2
            cry_amp = 0.2
            cry_freq = 350
            # Add irregularity with modulation
            modulation = 0.1 * np.sin(2 * np.pi * 0.05 * t)
            breathing = breathing_amp * np.sin(2 * np.pi * base_freq * t + modulation)
        elif severity == 'moderate':
            # Very irregular breathing, intermittent cry - 10-20 bpm
            base_freq = 0.25  # ~15 bpm base
            breathing_amp = 0.15
            cry_amp = 0.15
            cry_freq = 300
            # Add more irregularity
            modulation = 0.2 * np.sin(2 * np.pi * 0.08 * t)
            breathing = breathing_amp * np.sin(2 * np.pi * base_freq * t + modulation)
        else:  # severe
            # Gasping pattern - infrequent, irregular breathing - 5-15 bpm
            # Create gasping pattern with clear pauses between breaths
            cry_amp = 0.05
            cry_freq = 250
            
            # Generate gasping pattern: 10 breaths per 2 seconds = 5 bpm average
            num_gasps = 10  # for 2 seconds of audio
            gasp_duration = int(samples / num_gasps)  # samples per gasp
            
            breathing = np.zeros(samples)
            for i in range(num_gasps):
                gasp_start = i * gasp_duration
                gasp_end = min(gasp_start + gasp_duration, samples)
                if gasp_end > gasp_start:
                    # Create a gasp (inhale and exhale)
                    gasp_length = gasp_end - gasp_start
                    gasp_t = np.linspace(0, 1, gasp_length)
                    
                    # Inhale (stronger) then exhale (weaker)
                    inhale = 0.4 * np.sin(np.pi * gasp_t[:gasp_length//2])
                    exhale = 0.2 * np.sin(np.pi * gasp_t[gasp_length//2:])
                    
                    full_gasp = np.concatenate([inhale, exhale])
                    breathing[gasp_start:gasp_start+len(full_gasp)] = full_gasp[:gasp_end-gasp_start]
            
            breathing_amp = 0.3  # For metadata
        
        # Cry component (weak/intermittent)
        cry_mask = np.random.random(samples) < 0.1  # Only 10% crying
        cry = cry_amp * np.sin(2 * np.pi * cry_freq * t) * cry_mask
        
        audio = breathing + cry
        
        # Add noise
        noise = np.random.normal(0, 0.03, samples)
        audio += noise
        
        # Calculate breathing rate for metadata
        if severity == 'mild':
            breathing_rate = 20
        elif severity == 'moderate':
            breathing_rate = 15
        else:  # severe
            breathing_rate = 10
        
        metadata = {
            "condition": "asphyxia",
            "severity": severity,
            "breathing_rate": breathing_rate,
            "cry_frequency": cry_freq,
            "description": f"{severity.title()} asphyxia patterns - irregular breathing at {breathing_rate} bpm"
        }
        
        return audio, metadata
    
    def _generate_jaundice_audio(self, samples, severity):
        """Generate jaundice audio patterns"""
        audio = np.zeros(samples)
        
        t = np.linspace(0, samples / self.sample_rate, samples)
        
        # Jaundice: weak, monotone cry with normal breathing
        breathing_freq = 0.7  # Normal breathing (~42 bpm)
        breathing_amp = 0.25
        
        if severity == 'mild':
            cry_amp = 0.15
            cry_freq = 280  # Lower frequency
        elif severity == 'moderate':
            cry_amp = 0.1
            cry_freq = 250
        else:  # severe
            cry_amp = 0.05
            cry_freq = 220
        
        # Normal breathing
        breathing = breathing_amp * np.sin(2 * np.pi * breathing_freq * t)
        
        # Weak, monotone cry (no harmonics) - generate continuous but weak
        # Instead of random mask, create smooth fade in/out for crying episodes
        num_cries = 3
        cry_duration = samples // num_cries
        cry = np.zeros(samples)
        
        for i in range(num_cries):
            start_idx = i * cry_duration
            end_idx = min((i + 1) * cry_duration, samples)
            cry_segment = cry_amp * np.sin(2 * np.pi * cry_freq * t[start_idx:end_idx])
            # Apply smooth envelope (fade in/out)
            segment_len = end_idx - start_idx
            fade_in = np.linspace(0, 1, segment_len // 4)
            fade_out = np.linspace(1, 0, segment_len // 4)
            middle = np.ones(segment_len - len(fade_in) - len(fade_out))
            envelope = np.concatenate([fade_in, middle, fade_out])
            cry[start_idx:end_idx] = cry_segment * envelope
        
        audio = breathing + cry
        
        # Add smooth noise
        noise = np.random.normal(0, 0.04, samples)
        audio += noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.9
        
        metadata = {
            "condition": "jaundice",
            "severity": severity,
            "breathing_rate": breathing_freq * 60,
            "cry_frequency": cry_freq,
            "description": f"{severity.title()} jaundice - weak monotone cry, normal breathing at {int(breathing_freq * 60)} bpm"
        }
        
        return audio, metadata
    
    def _generate_cyanosis_audio(self, samples, severity):
        """Generate cyanosis audio patterns"""
        audio = np.zeros(samples)
        
        t = np.linspace(0, samples / self.sample_rate, samples)
        
        if severity == 'mild':
            # Rapid shallow breathing
            breathing_freq = 1.2 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
            breathing_amp = 0.2
            cry_amp = 0.3
            cry_freq = 450
        else:  # severe
            # Very rapid, labored breathing
            breathing_freq = 1.8 + 0.4 * np.sin(2 * np.pi * 0.2 * t)
            breathing_amp = 0.15
            cry_amp = 0.2
            cry_freq = 500
        
        # Rapid breathing component
        breathing = breathing_amp * np.sin(2 * np.pi * breathing_freq * t)
        
        # High-pitched cry
        cry_mask = np.random.random(samples) < 0.2
        cry = cry_amp * np.sin(2 * np.pi * cry_freq * t) * cry_mask
        
        # Add harmonics for shrill cry
        cry += 0.3 * np.sin(2 * np.pi * cry_freq * 2 * t) * cry_mask
        
        audio = breathing + cry
        
        # Add noise
        noise = np.random.normal(0, 0.04, samples)
        audio += noise
        
        metadata = {
            "condition": "cyanosis",
                    "severity": severity,
            "breathing_rate": np.mean(breathing_freq) * 60,
            "cry_frequency": cry_freq,
            "description": f"{severity.title()} cyanosis - rapid breathing, high-pitched cry"
        }
        
        return audio, metadata

# Create global instance
medical_datasets = MedicalDatasets()