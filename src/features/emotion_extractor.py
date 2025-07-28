"""
Enhanced emotion feature extractor with fallback strategies.

This module provides robust emotion feature extraction with multiple backends:
1. emotion2vec (primary) - for comprehensive emotion understanding
2. OpenSMILE/eGeMAPS (fallback) - for lightweight emotion features
3. Basic prosodic features (minimal fallback) - for basic emotion approximation
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import librosa
from pathlib import Path
import pickle
import time

logger = logging.getLogger(__name__)

# Emotion category mappings
EMOTION2VEC_LABELS = {
    0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 
    4: "neutral", 5: "other", 6: "sad", 7: "surprised", 8: "unknown"
}

# Blendshape mapping for emotions (focusing on non-mouth expressions)
EMOTION_TO_BLENDSHAPE_MAPPING = {
    "angry": {
        "browDownLeft": 0.8, "browDownRight": 0.8,
        "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
        "noseSneerLeft": 0.4, "noseSneerRight": 0.4,
    },
    "happy": {
        "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3,
        "cheekSquintLeft": 0.7, "cheekSquintRight": 0.7,
        "browOuterUpLeft": 0.2, "browOuterUpRight": 0.2,
    },
    "sad": {
        "browInnerUp": 0.7,
        "eyeSquintLeft": 0.4, "eyeSquintRight": 0.4,
    },
    "surprised": {
        "browInnerUp": 0.5, "browOuterUpLeft": 0.8, "browOuterUpRight": 0.8,
        "eyeWideLeft": 0.9, "eyeWideRight": 0.9,
    },
    "fearful": {
        "browInnerUp": 0.9, "browOuterUpLeft": 0.6, "browOuterUpRight": 0.6,
        "eyeWideLeft": 0.7, "eyeWideRight": 0.7,
    },
    "disgusted": {
        "browDownLeft": 0.5, "browDownRight": 0.5,
        "noseSneerLeft": 0.8, "noseSneerRight": 0.8,
        "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
    },
    "neutral": {},
    "other": {},
    "unknown": {}
}


class EmotionExtractor:
    """
    Robust emotion feature extractor with multiple backends and fallback strategies.
    """
    
    def __init__(
        self,
        backend: str = "emotion2vec",
        model_name: str = "iic/emotion2vec_plus_large",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        enable_caching: bool = True,
        batch_size: int = 4,
        sample_rate: int = 16000,
    ):
        """
        Initialize emotion extractor.
        
        Args:
            backend: Primary backend ("emotion2vec", "opensmile", "basic")
            model_name: emotion2vec model name
            device: Computation device
            cache_dir: Directory for caching features
            enable_caching: Whether to cache extracted features
            batch_size: Batch size for processing
            sample_rate: Expected sample rate
        """
        self.backend = backend
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "emotion_cache"
        self.enable_caching = enable_caching
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        
        # Initialize cache directory
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize extractors
        self.emotion2vec_model = None
        self.opensmile_extractor = None
        self.fallback_level = 0  # 0: emotion2vec, 1: opensmile, 2: basic
        
        # Statistics
        self.extraction_stats = {
            "total_calls": 0,
            "total_processed": 0,  # Add missing field
            "success_rate": 1.0,   # Add missing field
            "cache_hits": 0,
            "emotion2vec_calls": 0,
            "fallback_calls": 0,
            "avg_processing_time": 0.0
        }
        
        self._initialize_backend()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _initialize_backend(self):
        """Initialize the primary backend with fallback strategy."""
        if self.backend == "emotion2vec":
            self._initialize_emotion2vec()
        elif self.backend == "opensmile":
            self._initialize_opensmile()
        else:  # basic
            self.fallback_level = 2
            logger.info("Using basic prosodic features for emotion approximation")
    
    def _initialize_emotion2vec(self):
        """Initialize emotion2vec with proper error handling."""
        try:
            # Try importing FunASR
            from funasr import AutoModel
            
            logger.info(f"Initializing emotion2vec model: {self.model_name}")
            
            # Initialize model with proper hub specification
            self.emotion2vec_model = AutoModel(
                model=self.model_name,
                hub="ms",  # Use ModelScope hub for better stability
                device=str(self.device),
                cache_dir=str(self.cache_dir / "models")
            )
            
            # Test the model with dummy data
            dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second
            test_result = self.emotion2vec_model.generate(
                dummy_audio,
                granularity="utterance",
                extract_embedding=True
            )
            
            logger.info("emotion2vec initialized successfully")
            logger.info(f"Test result structure: {list(test_result[0].keys()) if test_result else 'No result'}")
            
        except ImportError as e:
            logger.warning(f"FunASR not available: {e}")
            logger.info("Falling back to OpenSMILE backend")
            self.fallback_level = 1
            self._initialize_opensmile()
            
        except Exception as e:
            logger.error(f"Failed to initialize emotion2vec: {e}")
            logger.info("Falling back to OpenSMILE backend")
            self.fallback_level = 1
            self._initialize_opensmile()
    
    def _initialize_opensmile(self):
        """Initialize OpenSMILE extractor as fallback."""
        try:
            from .opensmile_extractor import OpenSMILEeGeMAPSExtractor
            
            # Create advanced OpenSMILE extractor with sliding window
            self.opensmile_extractor = OpenSMILEeGeMAPSExtractor(
                sample_rate=self.sample_rate,
                context_window=20.0,  # 20s context (longer than mel's 8.5s)
                update_interval=0.3,  # 300ms updates (real-time)
                feature_set="eGeMAPSv02",
                feature_level="Functionals",
                enable_caching=self.enable_caching,
            )
            
            logger.info("OpenSMILE eGeMAPS sliding window extractor initialized successfully")
            logger.info(f"  Context window: 20.0s (vs mel's 8.5s)")
            logger.info(f"  Update interval: 300ms")
            logger.info(f"  Feature dimension: {self.opensmile_extractor.feature_dim}")
            
        except ImportError as e:
            logger.warning(f"OpenSMILE not available: {e}")
            logger.info("Falling back to basic prosodic features")
            self.fallback_level = 2
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenSMILE: {e}")
            self.fallback_level = 2
    
    def extract_features(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        return_embeddings: bool = True,
        return_predictions: bool = True,
    ) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """
        Extract emotion features with fallback handling.
        
        Args:
            audio: Audio data (B, T) or (T,)
            return_embeddings: Whether to return feature embeddings
            return_predictions: Whether to return emotion predictions
            
        Returns:
            Dictionary containing:
            - embeddings: Feature embeddings (B, T_emotion, dim) if requested
            - predictions: Emotion category predictions if requested
            - blendshape_weights: Mapped blendshape weights
            - metadata: Processing metadata
        """
        start_time = time.time()
        self.extraction_stats["total_calls"] += 1
        
        # Convert to numpy
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        
        # Handle batch dimension
        if audio.ndim == 1:
            audio = audio[None, :]  # Add batch dimension
        
        batch_size = audio.shape[0]
        
        # Initialize monitoring
        try:
            from ..utils.emotion_monitor import get_monitor
            monitor = get_monitor()
            processing_id = monitor.log_processing_start(
                audio_shape=audio.shape,
                backend=self._get_backend_name(),
                config={
                    "backend": self.backend,
                    "model_name": self.model_name,
                    "fallback_level": self.fallback_level,
                    "batch_size": batch_size
                }
            )
        except Exception as e:
            logger.debug(f"Monitoring initialization failed: {e}")
            processing_id = None
            monitor = None
        
        results = {
            "embeddings": [],
            "predictions": [],
            "blendshape_weights": [],
            "metadata": {
                "backend_used": self._get_backend_name(),
                "processing_time": 0.0,
                "cache_used": False,
                "processing_id": processing_id
            }
        }
        
        # Process each sample in batch
        for i in range(batch_size):
            sample_audio = audio[i]
            
            # Check cache first
            if self.enable_caching:
                cached_result = self._load_from_cache(sample_audio)
                if cached_result is not None:
                    self.extraction_stats["cache_hits"] += 1
                    results["embeddings"].append(cached_result["embeddings"])
                    results["predictions"].append(cached_result["predictions"])
                    results["blendshape_weights"].append(cached_result["blendshape_weights"])
                    results["metadata"]["cache_used"] = True
                    continue
            
            # Extract features based on current backend
            if self.fallback_level == 0:  # emotion2vec
                sample_result = self._extract_emotion2vec(sample_audio)
            elif self.fallback_level == 1:  # opensmile
                sample_result = self._extract_opensmile(sample_audio)
            else:  # basic
                sample_result = self._extract_basic(sample_audio)
            
            # Cache results
            if self.enable_caching and sample_result is not None:
                self._save_to_cache(sample_audio, sample_result)
            
            # Append to batch results
            if sample_result is not None:
                results["embeddings"].append(sample_result["embeddings"])
                results["predictions"].append(sample_result["predictions"])
                results["blendshape_weights"].append(sample_result["blendshape_weights"])
        
        # Convert lists to arrays/tensors
        if results["embeddings"]:
            if return_embeddings:
                results["embeddings"] = np.stack(results["embeddings"])
            else:
                del results["embeddings"]
            
            if return_predictions:
                # Keep as list of dictionaries for predictions
                pass
            else:
                del results["predictions"]
            
            results["blendshape_weights"] = np.stack(results["blendshape_weights"])
        
        # Update statistics
        processing_time = time.time() - start_time
        results["metadata"]["processing_time"] = processing_time
        self.extraction_stats["total_processed"] += batch_size
        success = len(results.get("embeddings", [])) > 0
        if success:
            self.extraction_stats["success_rate"] = (
                (self.extraction_stats["success_rate"] * (self.extraction_stats["total_calls"] - 1) + 1.0) 
                / self.extraction_stats["total_calls"]
            )
        else:
            self.extraction_stats["success_rate"] = (
                (self.extraction_stats["success_rate"] * (self.extraction_stats["total_calls"] - 1) + 0.0) 
                / self.extraction_stats["total_calls"]
            )
        self.extraction_stats["avg_processing_time"] = (
            (self.extraction_stats["avg_processing_time"] * (self.extraction_stats["total_calls"] - 1) + processing_time) 
            / self.extraction_stats["total_calls"]
        )
        
        # Log monitoring completion
        if monitor and processing_id:
            try:
                monitor.log_processing_end(
                    processing_id=processing_id,
                    success=len(results.get("embeddings", [])) > 0,
                    results={
                        "backend": self._get_backend_name(),
                        "predictions": results.get("predictions", [{}])[0] if results.get("predictions") else {},
                        "blendshape_weights": results.get("blendshape_weights", np.array([]))[0] if len(results.get("blendshape_weights", [])) > 0 else np.array([]),
                        "embedding_shape": results["embeddings"].shape if hasattr(results.get("embeddings", []), 'shape') else None,
                        "cache_used": results["metadata"]["cache_used"]
                    }
                )
            except Exception as e:
                logger.debug(f"Monitoring completion failed: {e}")
        
        return results
    
    def _extract_emotion2vec(self, audio: np.ndarray) -> Optional[Dict]:
        """Extract features using emotion2vec."""
        if self.emotion2vec_model is None:
            return None
        
        try:
            self.extraction_stats["emotion2vec_calls"] += 1
            
            # Ensure correct sample rate
            if len(audio) / self.sample_rate > 0.1:  # Minimum 0.1 seconds
                # Process with emotion2vec
                result = self.emotion2vec_model.generate(
                    audio.astype(np.float32),
                    granularity="utterance",  # Use utterance level for overall emotion
                    extract_embedding=True
                )
                
                if result and len(result) > 0:
                    # Extract embeddings and predictions
                    embeddings = result[0].get("feats", np.random.randn(1024))  # Default dimension
                    
                    # Get emotion predictions if available
                    if "labels" in result[0]:
                        emotion_labels = result[0]["labels"]
                        predictions = {EMOTION2VEC_LABELS.get(i, f"emotion_{i}"): float(prob) 
                                     for i, prob in enumerate(emotion_labels)}
                    else:
                        # Generate dummy predictions for testing
                        predictions = {label: 0.1 for label in EMOTION2VEC_LABELS.values()}
                        predictions["neutral"] = 0.6  # Default to neutral
                    
                    # Map to blendshape weights
                    blendshape_weights = self._emotion_to_blendshapes(predictions)
                    
                    return {
                        "embeddings": embeddings.reshape(1, -1) if embeddings.ndim == 1 else embeddings,
                        "predictions": predictions,
                        "blendshape_weights": blendshape_weights
                    }
                    
        except Exception as e:
            logger.warning(f"emotion2vec extraction failed: {e}")
            # Fall back to next level
            self.fallback_level = 1
            return self._extract_opensmile(audio)
        
        return None
    
    def _extract_opensmile(self, audio: np.ndarray) -> Optional[Dict]:
        """Extract features using OpenSMILE with sliding window."""
        if self.opensmile_extractor is None:
            return None
        
        try:
            self.extraction_stats["fallback_calls"] += 1
            
            # Process audio through sliding window extractor
            # This handles long-term context automatically
            features = self.opensmile_extractor.process_audio_batch(audio)
            
            if features is None or features.size == 0:
                logger.warning("OpenSMILE returned empty features")
                return None
            
            # Features shape: (T_features, feature_dim) or (feature_dim,)
            if features.ndim == 1:
                features = features[None, :]  # Add time dimension
            
            # Use the most recent feature vector for utterance-level processing
            embeddings = features[-1]  # Most recent features
            
            # Enhanced emotion mapping using eGeMAPS features
            predictions = self._egemaps_to_emotion(embeddings)
            
            # Map to blendshapes with long-term context
            blendshape_weights = self._emotion_to_blendshapes(predictions)
            
            # Enhanced blendshape mapping using eGeMAPS features directly
            enhanced_blendshapes = self._egemaps_to_blendshapes(embeddings)
            blendshape_weights = 0.7 * blendshape_weights + 0.3 * enhanced_blendshapes
            
            return {
                "embeddings": embeddings.reshape(1, -1),
                "predictions": predictions,
                "blendshape_weights": blendshape_weights,
                "context_stats": self.opensmile_extractor.get_stats()
            }
            
        except Exception as e:
            logger.warning(f"OpenSMILE extraction failed: {e}")
            self.fallback_level = 2
            return self._extract_basic(audio)
        
        return None
    
    def _extract_basic(self, audio: np.ndarray) -> Optional[Dict]:
        """Extract basic prosodic features as minimal fallback."""
        try:
            self.extraction_stats["fallback_calls"] += 1
            
            # Basic prosodic analysis
            # Energy
            energy = np.mean(audio ** 2)
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Spectral centroid
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            ))
            
            # F0 estimation
            f0 = librosa.yin(audio, fmin=50, fmax=400, sr=self.sample_rate)
            f0_mean = np.nanmean(f0)
            f0_std = np.nanstd(f0)
            
            # Create basic embeddings
            embeddings = np.array([
                energy, zcr, spectral_centroid, f0_mean, f0_std,
                np.mean(audio), np.std(audio), np.max(audio), np.min(audio)
            ])
            
            # Simple heuristic emotion classification
            predictions = self._basic_emotion_heuristic(energy, zcr, f0_mean, f0_std)
            
            # Map to blendshapes
            blendshape_weights = self._emotion_to_blendshapes(predictions)
            
            return {
                "embeddings": embeddings.reshape(1, -1),
                "predictions": predictions,
                "blendshape_weights": blendshape_weights
            }
            
        except Exception as e:
            logger.error(f"Basic extraction failed: {e}")
            return None
    
    def _prosodic_to_emotion(self, features) -> Dict[str, float]:
        """Convert OpenSMILE features to emotion predictions (heuristic)."""
        # This is a simplified mapping - in practice, use a trained classifier
        predictions = {"neutral": 0.7}
        
        try:
            # Example heuristic based on some eGeMAPS features
            f0_mean = features.get('F0semitoneFrom27.5Hz_sma3nz_amean', [0]).iloc[0]
            f0_std = features.get('F0semitoneFrom27.5Hz_sma3nz_stddevNorm', [0]).iloc[0]
            loudness = features.get('loudness_sma3_amean', [0]).iloc[0]
            
            if f0_mean > 200 and f0_std > 10:  # High pitch, variable
                predictions = {"surprised": 0.6, "happy": 0.3, "neutral": 0.1}
            elif f0_std > 15:  # High variability
                predictions = {"angry": 0.5, "neutral": 0.5}
            elif loudness < -30:  # Very quiet
                predictions = {"sad": 0.6, "neutral": 0.4}
            
        except Exception:
            pass
        
        return predictions
    
    def _egemaps_to_emotion(self, features: np.ndarray) -> Dict[str, float]:
        """Convert eGeMAPS features to emotion predictions using advanced heuristics."""
        predictions = {"neutral": 0.5}
        
        try:
            # eGeMAPS has ~88 features, we'll use key ones for emotion detection
            # This is a more sophisticated mapping than basic prosodic features
            
            # Assuming eGeMAPS feature order (simplified subset)
            feature_dict = {}
            if len(features) >= 10:  # Basic safety check
                # Extract key prosodic indicators (approximate indices)
                f0_mean = features[0] if len(features) > 0 else 0
                f0_std = features[1] if len(features) > 1 else 0
                loudness_mean = features[10] if len(features) > 10 else 0
                loudness_std = features[11] if len(features) > 11 else 0
                jitter = features[20] if len(features) > 20 else 0
                shimmer = features[21] if len(features) > 21 else 0
                hnr = features[30] if len(features) > 30 else 0  # Harmonics-to-Noise Ratio
                
                # Advanced emotion heuristics based on eGeMAPS literature
                # High arousal emotions (anger, joy, surprise)
                arousal_score = 0.0
                if f0_std > 20 and loudness_std > 5:  # High pitch and energy variability
                    arousal_score += 0.4
                if jitter > 0.005 or shimmer > 0.05:  # Voice quality changes
                    arousal_score += 0.3
                
                # Valence indicators (positive vs negative)
                valence_score = 0.0
                if f0_mean > 150 and hnr > 10:  # Higher pitch, cleaner voice (positive)
                    valence_score += 0.4
                elif f0_mean < 100 and hnr < 5:  # Lower pitch, rougher voice (negative)
                    valence_score -= 0.4
                
                # Map to specific emotions
                if arousal_score > 0.5 and valence_score > 0.2:
                    predictions = {"happy": 0.6, "surprised": 0.2, "neutral": 0.2}
                elif arousal_score > 0.5 and valence_score < -0.2:
                    predictions = {"angry": 0.5, "fearful": 0.3, "neutral": 0.2}
                elif arousal_score < 0.2 and valence_score < -0.2:
                    predictions = {"sad": 0.6, "neutral": 0.4}
                elif arousal_score > 0.3 and abs(valence_score) < 0.2:
                    predictions = {"surprised": 0.5, "neutral": 0.5}
                else:
                    predictions = {"neutral": 0.8, "other": 0.2}
                    
        except Exception as e:
            logger.debug(f"eGeMAPS emotion mapping failed: {e}")
            predictions = {"neutral": 0.8, "other": 0.2}
        
        return predictions
    
    def _egemaps_to_blendshapes(self, features: np.ndarray) -> np.ndarray:
        """Direct mapping from eGeMAPS features to blendshapes for long-term context."""
        from ..model.dual_stream_attention import ARKIT_BLENDSHAPES, EXPRESSION_INDICES
        
        blendshape_weights = np.zeros(len(ARKIT_BLENDSHAPES))
        
        try:
            if len(features) >= 10:
                # Extract key features
                f0_mean = features[0] if len(features) > 0 else 0
                f0_std = features[1] if len(features) > 1 else 0
                loudness_mean = features[10] if len(features) > 10 else 0
                energy_std = features[11] if len(features) > 11 else 0
                
                # Normalize features (rough normalization)
                f0_norm = np.clip((f0_mean - 100) / 100, -1, 1)  # Normalize around 100-200 Hz
                f0_var_norm = np.clip(f0_std / 50, 0, 1)  # Variability indicator
                energy_norm = np.clip((loudness_mean + 30) / 30, 0, 1)  # Normalize loudness
                energy_var_norm = np.clip(energy_std / 10, 0, 1)  # Energy variability
                
                # Map to expression blendshapes (avoid mouth region)
                # Brow movements (related to pitch and energy patterns)
                if 'browInnerUp' in ARKIT_BLENDSHAPES:
                    idx = ARKIT_BLENDSHAPES.index('browInnerUp')
                    blendshape_weights[idx] = max(0, f0_var_norm * 0.6)  # Surprise/concern
                
                if 'browDownLeft' in ARKIT_BLENDSHAPES and 'browDownRight' in ARKIT_BLENDSHAPES:
                    brow_down_left = ARKIT_BLENDSHAPES.index('browDownLeft')
                    brow_down_right = ARKIT_BLENDSHAPES.index('browDownRight')
                    brow_intensity = max(0, (1 - energy_norm) * f0_var_norm * 0.5)  # Anger/concentration
                    blendshape_weights[brow_down_left] = brow_intensity
                    blendshape_weights[brow_down_right] = brow_intensity
                
                # Eye expressions (related to overall emotional state)
                if 'eyeWideLeft' in ARKIT_BLENDSHAPES and 'eyeWideRight' in ARKIT_BLENDSHAPES:
                    eye_wide_left = ARKIT_BLENDSHAPES.index('eyeWideLeft')
                    eye_wide_right = ARKIT_BLENDSHAPES.index('eyeWideRight')
                    eye_intensity = max(0, f0_var_norm * energy_var_norm * 0.4)  # Surprise
                    blendshape_weights[eye_wide_left] = eye_intensity
                    blendshape_weights[eye_wide_right] = eye_intensity
                
                # Cheek expressions (related to positive emotions)
                if 'cheekSquintLeft' in ARKIT_BLENDSHAPES and 'cheekSquintRight' in ARKIT_BLENDSHAPES:
                    cheek_left = ARKIT_BLENDSHAPES.index('cheekSquintLeft')
                    cheek_right = ARKIT_BLENDSHAPES.index('cheekSquintRight')
                    cheek_intensity = max(0, energy_norm * (1 - f0_var_norm) * 0.3)  # Contentment
                    blendshape_weights[cheek_left] = cheek_intensity
                    blendshape_weights[cheek_right] = cheek_intensity
                    
        except Exception as e:
            logger.debug(f"eGeMAPS to blendshapes mapping failed: {e}")
        
        return np.clip(blendshape_weights, 0, 1)
    
    def _basic_emotion_heuristic(
        self, energy: float, zcr: float, f0_mean: float, f0_std: float
    ) -> Dict[str, float]:
        """Basic emotion classification heuristic."""
        predictions = {"neutral": 0.8, "other": 0.2}
        
        try:
            # Very simple heuristics
            if energy > 0.1 and f0_std > 50:  # High energy, variable pitch
                predictions = {"angry": 0.6, "neutral": 0.4}
            elif energy > 0.05 and f0_mean > 200:  # Moderate energy, high pitch
                predictions = {"happy": 0.5, "surprised": 0.3, "neutral": 0.2}
            elif energy < 0.01:  # Low energy
                predictions = {"sad": 0.6, "neutral": 0.4}
            
        except Exception:
            pass
        
        return predictions
    
    def _emotion_to_blendshapes(self, predictions: Dict[str, float]) -> np.ndarray:
        """Map emotion predictions to blendshape weights."""
        from ..model.dual_stream_attention import ARKIT_BLENDSHAPES, EXPRESSION_INDICES
        
        # Initialize blendshape weights
        blendshape_weights = np.zeros(len(ARKIT_BLENDSHAPES))
        
        # Apply emotion mapping
        for emotion, confidence in predictions.items():
            if emotion in EMOTION_TO_BLENDSHAPE_MAPPING:
                emotion_mapping = EMOTION_TO_BLENDSHAPE_MAPPING[emotion]
                
                for blendshape_name, weight in emotion_mapping.items():
                    if blendshape_name in ARKIT_BLENDSHAPES:
                        idx = ARKIT_BLENDSHAPES.index(blendshape_name)
                        blendshape_weights[idx] += confidence * weight
        
        # Normalize to [0, 1] range
        blendshape_weights = np.clip(blendshape_weights, 0, 1)
        
        return blendshape_weights
    
    def _get_backend_name(self) -> str:
        """Get current backend name."""
        if self.fallback_level == 0:
            return "emotion2vec"
        elif self.fallback_level == 1:
            return "opensmile"
        else:
            return "basic"
    
    def _load_from_cache(self, audio: np.ndarray) -> Optional[Dict]:
        """Load cached features."""
        if not self.enable_caching:
            return None
        
        try:
            # Create hash for audio data
            audio_hash = hash(audio.tobytes())
            cache_file = self.cache_dir / f"emotion_{audio_hash}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        
        return None
    
    def _save_to_cache(self, audio: np.ndarray, result: Dict):
        """Save features to cache."""
        if not self.enable_caching:
            return
        
        try:
            audio_hash = hash(audio.tobytes())
            cache_file = self.cache_dir / f"emotion_{audio_hash}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.debug(f"Failed to cache results: {e}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get extraction statistics."""
        return self.extraction_stats.copy()
    
    def reset_statistics(self):
        """Reset extraction statistics."""
        self.extraction_stats = {
            "total_calls": 0,
            "total_processed": 0,  # Add missing field
            "success_rate": 1.0,   # Add missing field
            "cache_hits": 0,
            "emotion2vec_calls": 0,
            "fallback_calls": 0,
            "avg_processing_time": 0.0
        }


def create_emotion_extractor(config: Dict) -> EmotionExtractor:
    """Create emotion extractor from configuration."""
    return EmotionExtractor(
        backend=config.get("backend", "emotion2vec"),
        model_name=config.get("model_name", "iic/emotion2vec_plus_large"),
        device=config.get("device", "auto"),
        cache_dir=config.get("cache_dir"),
        enable_caching=config.get("enable_caching", True),
        batch_size=config.get("batch_size", 4),
        sample_rate=config.get("sample_rate", 16000),
    )