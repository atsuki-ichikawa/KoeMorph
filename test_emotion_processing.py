#!/usr/bin/env python3
"""
Test script for emotion processing in KoeMorph dual-stream architecture.

This script tests the emotion extraction pipeline with fallback strategies
and provides detailed diagnostics about the processing performance.
"""

import sys
import logging
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.features.emotion_extractor import EmotionExtractor
from src.model.simplified_dual_stream_model import SimplifiedDualStreamModel
from src.utils.emotion_monitor import get_monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_emotion_extractor():
    """Test the EmotionExtractor with different backends."""
    print("\n" + "="*50)
    print("TESTING EMOTION EXTRACTOR")
    print("="*50)
    
    # Generate dummy audio (3 seconds at 16kHz)
    sample_rate = 16000
    duration = 3.0
    audio_length = int(sample_rate * duration)
    
    # Create realistic-looking audio with some variation
    audio = np.random.randn(2, audio_length) * 0.1  # Batch of 2 samples
    
    # Test different backends
    backends = ["emotion2vec", "opensmile", "basic"]
    
    for backend in backends:
        print(f"\n--- Testing {backend} backend ---")
        
        try:
            # Create extractor
            extractor = EmotionExtractor(
                backend=backend,
                device="cpu",  # Use CPU for testing
                enable_caching=False,  # Disable caching for testing
                batch_size=2,
                sample_rate=sample_rate
            )
            
            print(f"✓ Initialized {backend} extractor")
            print(f"  Backend used: {extractor._get_backend_name()}")
            print(f"  Fallback level: {extractor.fallback_level}")
            
            # Extract features
            results = extractor.extract_features(
                audio,
                return_embeddings=True,
                return_predictions=True
            )
            
            print(f"✓ Feature extraction successful")
            embeddings = results.get('embeddings')
            if embeddings is not None:
                if hasattr(embeddings, 'shape'):
                    print(f"  Embeddings shape: {embeddings.shape}")
                else:
                    print(f"  Embeddings length: {len(embeddings)} (list)")
            else:
                print(f"  Embeddings: None")
            print(f"  Processing time: {results['metadata']['processing_time']:.3f}s")
            print(f"  Backend used: {results['metadata']['backend_used']}")
            
            if 'predictions' in results and results['predictions']:
                print(f"  Sample predictions: {list(results['predictions'][0].keys())}")
            
            if 'blendshape_weights' in results:
                print(f"  Blendshape weights shape: {results['blendshape_weights'].shape}")
                active_blendshapes = np.sum(results['blendshape_weights'] > 0.1)
                print(f"  Active blendshapes: {active_blendshapes}/52")
            
        except Exception as e:
            print(f"✗ {backend} backend failed: {e}")
            import traceback
            traceback.print_exc()


def test_dual_stream_model():
    """Test the SimplifiedDualStreamModel."""
    print("\n" + "="*50)
    print("TESTING DUAL-STREAM MODEL")
    print("="*50)
    
    # Model configuration
    emotion_config = {
        "backend": "basic",  # Use basic backend for reliable testing
        "device": "cpu",
        "enable_caching": False,
        "sample_rate": 16000
    }
    
    try:
        # Create model
        model = SimplifiedDualStreamModel(
            d_model=128,  # Smaller for testing
            num_heads=4,
            num_blendshapes=52,
            sample_rate=16000,
            target_fps=30,
            mel_sequence_length=128,  # Smaller for testing
            emotion_config=emotion_config,
            device="cpu"
        )
        
        print("✓ Model initialized successfully")
        print(f"  Model info: {model.get_model_info()}")
        
        # Generate test audio (2 seconds)
        batch_size = 2
        audio_length = 16000 * 2  # 2 seconds
        test_audio = torch.randn(batch_size, audio_length) * 0.1
        
        print(f"\n--- Testing forward pass ---")
        print(f"Input audio shape: {test_audio.shape}")
        
        # Forward pass
        with torch.no_grad():
            results = model(test_audio, return_attention=True)
        
        print("✓ Forward pass successful")
        print(f"  Output blendshapes shape: {results['blendshapes'].shape}")
        print(f"  Blendshape range: [{results['blendshapes'].min():.3f}, {results['blendshapes'].max():.3f}]")
        
        if 'emotion_backend' in results:
            print(f"  Emotion backend used: {results['emotion_backend']}")
            print(f"  Emotion processing time: {results['emotion_processing_time']:.3f}s")
        
        # Check for attention weights
        if 'mel_attention_weights' in results:
            print(f"  Mel attention shape: {results['mel_attention_weights'].shape}")
        if 'emotion_attention_weights' in results:
            print(f"  Emotion attention shape: {results['emotion_attention_weights'].shape}")
        
        # Test model info
        model_info = model.get_model_info()
        print(f"\n--- Model Information ---")
        for key, value in model_info.items():
            if key != 'emotion_extraction_stats':  # Skip complex nested dict
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ Dual-stream model test failed: {e}")
        import traceback
        traceback.print_exc()


def test_monitoring():
    """Test the emotion processing monitoring system."""
    print("\n" + "="*50)
    print("TESTING MONITORING SYSTEM")
    print("="*50)
    
    try:
        # Get monitor instance
        monitor = get_monitor()
        
        print("✓ Monitor initialized")
        
        # Simulate some processing
        for i in range(3):
            processing_id = monitor.log_processing_start(
                audio_shape=(1, 16000),
                backend="test_backend",
                config={"test": True}
            )
            
            # Simulate processing time
            import time
            time.sleep(0.1)
            
            monitor.log_processing_end(
                processing_id=processing_id,
                success=True,
                results={
                    "backend": "test_backend",
                    "predictions": {"happy": 0.7, "neutral": 0.3},
                    "blendshape_weights": np.random.rand(52) * 0.5,
                    "cache_used": False
                }
            )
        
        # Get statistics
        stats = monitor.get_statistics()
        print("✓ Monitoring statistics generated")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Backend usage: {stats['backend_usage']}")
        
        # Generate report
        report = monitor.generate_report()
        print("✓ Monitoring report generated")
        print(f"  Report length: {len(report)} characters")
        
    except Exception as e:
        print(f"✗ Monitoring test failed: {e}")
        import traceback
        traceback.print_exc()


def run_all_tests():
    """Run all tests."""
    print("KOEMORPH EMOTION PROCESSING TEST SUITE")
    print("=" * 60)
    
    try:
        test_emotion_extractor()
        test_dual_stream_model()
        test_monitoring()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED")
        print("Check the logs above for detailed results.")
        print("If any test failed, check the error messages and traceback.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()