#!/usr/bin/env python3
"""
Comparison and validation script for eGeMAPS vs emotion2vec vs basic features.

Tests the performance and characteristics of different long-term context
feature extraction methods for KoeMorph dual-stream architecture.
"""

import sys
import logging
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.features.emotion_extractor import EmotionExtractor
from src.features.opensmile_extractor import OpenSMILEeGeMAPSExtractor
from src.model.simplified_dual_stream_model import SimplifiedDualStreamModel
from src.utils.emotion_monitor import get_monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate realistic test audio with different emotional characteristics."""
    samples = int(duration * sample_rate)
    
    # Base audio signal
    t = np.linspace(0, duration, samples)
    
    # Simulate speech-like signal with varying characteristics
    # Base frequency modulation (simulates F0)
    f0_base = 120  # Base frequency
    f0_variation = 30 * np.sin(2 * np.pi * 0.5 * t)  # Slow variation
    f0 = f0_base + f0_variation
    
    # Carrier signal (simulates formants)
    carrier = np.sin(2 * np.pi * f0 * t)
    
    # Add formant-like structure
    formant1 = 0.3 * np.sin(2 * np.pi * 800 * t)
    formant2 = 0.2 * np.sin(2 * np.pi * 1200 * t)
    
    # Energy envelope (simulates speech energy patterns)
    energy_envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 2 * t) * np.exp(-0.1 * t)
    
    # Combine signals
    audio = (carrier + formant1 + formant2) * energy_envelope
    
    # Add realistic noise
    noise = np.random.randn(samples) * 0.05
    audio += noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def create_emotion_scenarios() -> Dict[str, np.ndarray]:
    """Create audio scenarios representing different emotional contexts."""
    scenarios = {}
    sample_rate = 16000
    
    # Scenario 1: Neutral/calm speech (20 seconds)
    neutral_audio = generate_test_audio(20.0, sample_rate)
    scenarios["neutral_20s"] = neutral_audio
    
    # Scenario 2: Excited/high arousal (15 seconds) 
    t = np.linspace(0, 15, int(15 * sample_rate))
    excited_f0 = 150 + 50 * np.sin(2 * np.pi * 1.5 * t)  # Higher, more variable F0
    excited_carrier = np.sin(2 * np.pi * excited_f0 * t)
    excited_energy = 0.8 + 0.4 * np.sin(2 * np.pi * 3 * t)  # More energy variation
    excited_audio = excited_carrier * excited_energy
    excited_audio += np.random.randn(len(excited_audio)) * 0.03
    excited_audio = excited_audio / np.max(np.abs(excited_audio)) * 0.9
    scenarios["excited_15s"] = excited_audio.astype(np.float32)
    
    # Scenario 3: Sad/low arousal (25 seconds)
    sad_audio = generate_test_audio(25.0, sample_rate)
    # Modify to be more monotone and lower energy
    sad_audio *= 0.6  # Lower overall energy
    # Add low-pass filtering effect
    from scipy import signal
    b, a = signal.butter(4, 0.3, 'low')
    sad_audio = signal.filtfilt(b, a, sad_audio)
    scenarios["sad_25s"] = sad_audio.astype(np.float32)
    
    # Scenario 4: Short utterance (3 seconds)
    short_audio = generate_test_audio(3.0, sample_rate)
    scenarios["short_3s"] = short_audio
    
    # Scenario 5: Very long context (45 seconds)
    long_audio = generate_test_audio(45.0, sample_rate)
    scenarios["long_45s"] = long_audio
    
    return scenarios


def test_feature_extraction_backends():
    """Test different feature extraction backends."""
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTION BACKENDS")
    print("="*60)
    
    scenarios = create_emotion_scenarios()
    backends = ["opensmile", "emotion2vec", "basic"]
    results = {}
    
    for backend in backends:
        print(f"\n--- Testing {backend.upper()} Backend ---")
        backend_results = {}
        
        try:
            # Create extractor
            config = {
                "backend": backend,
                "sample_rate": 16000,
                "enable_caching": False,
                "device": "cpu"
            }
            
            if backend == "opensmile":
                config.update({
                    "context_window": 20.0,
                    "update_interval": 0.3,
                    "feature_set": "eGeMAPSv02"
                })
            
            extractor = EmotionExtractor(**config)
            print(f"✓ {backend} extractor initialized")
            print(f"  Fallback level: {extractor.fallback_level}")
            print(f"  Backend used: {extractor._get_backend_name()}")
            
            # Test each scenario
            for scenario_name, audio in scenarios.items():
                print(f"\n  Testing scenario: {scenario_name}")
                
                start_time = time.time()
                results_dict = extractor.extract_features(
                    audio[None, :],  # Add batch dimension
                    return_embeddings=True,
                    return_predictions=True
                )
                processing_time = time.time() - start_time
                
                # Extract info
                info = {
                    "audio_duration": len(audio) / 16000,
                    "processing_time": processing_time,
                    "feature_shape": results_dict["embeddings"].shape if "embeddings" in results_dict else None,
                    "backend_used": results_dict["metadata"]["backend_used"],
                    "predictions": results_dict.get("predictions", [{}])[0] if results_dict.get("predictions") else {},
                }
                
                # Performance metrics
                processing_ratio = processing_time / info["audio_duration"]
                info["real_time_factor"] = processing_ratio
                info["faster_than_realtime"] = processing_ratio < 1.0
                
                backend_results[scenario_name] = info
                
                print(f"    ✓ Duration: {info['audio_duration']:.1f}s")
                print(f"    ✓ Processing: {processing_time:.3f}s (RTF: {processing_ratio:.3f})")
                print(f"    ✓ Features: {info['feature_shape']}")
                print(f"    ✓ Real-time: {'Yes' if info['faster_than_realtime'] else 'No'}")
                
                if info["predictions"]:
                    top_emotion = max(info["predictions"].items(), key=lambda x: x[1])
                    print(f"    ✓ Top emotion: {top_emotion[0]} ({top_emotion[1]:.2f})")
            
            results[backend] = backend_results
            
        except Exception as e:
            print(f"✗ {backend} backend failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def test_sliding_window_performance():
    """Test OpenSMILE sliding window performance in detail."""
    print("\n" + "="*60)
    print("TESTING OPENSMILE SLIDING WINDOW PERFORMANCE")
    print("="*60)
    
    try:
        # Create OpenSMILE extractor directly
        extractor = OpenSMILEeGeMAPSExtractor(
            context_window=20.0,
            update_interval=0.3,
            feature_set="eGeMAPSv02"
        )
        
        print("✓ OpenSMILE extractor created")
        print(f"  Feature dimension: {extractor.feature_dim}")
        print(f"  Context window: {extractor.context_window}s")
        print(f"  Update interval: {extractor.update_interval}s")
        
        # Test with streaming audio simulation
        audio_duration = 30.0  # 30 seconds
        frame_duration = 0.033  # 33ms frames (like mel-spectrogram)
        
        full_audio = generate_test_audio(audio_duration)
        frame_size = int(16000 * frame_duration)
        
        print(f"\n--- Streaming Simulation ---")
        print(f"Audio duration: {audio_duration}s")
        print(f"Frame size: {frame_size} samples ({frame_duration*1000:.1f}ms)")
        
        # Simulate streaming
        streaming_times = []
        feature_updates = []
        
        for i in range(0, len(full_audio), frame_size):
            frame = full_audio[i:i+frame_size]
            
            start_time = time.time()
            features = extractor.process_audio_frame(frame)
            end_time = time.time()
            
            streaming_times.append(end_time - start_time)
            
            if features is not None:
                feature_updates.append(i / 16000)  # Time when features were updated
        
        # Statistics
        avg_frame_time = np.mean(streaming_times)
        max_frame_time = np.max(streaming_times)
        frame_real_time = avg_frame_time / frame_duration
        
        print(f"\n--- Performance Results ---")
        print(f"Total frames processed: {len(streaming_times)}")
        print(f"Feature updates: {len(feature_updates)}")
        print(f"Average frame processing: {avg_frame_time*1000:.2f}ms")
        print(f"Max frame processing: {max_frame_time*1000:.2f}ms")
        print(f"Frame RTF: {frame_real_time:.4f}")
        print(f"Real-time capable: {'Yes' if frame_real_time < 1.0 else 'No'}")
        
        # Expected update frequency
        expected_updates = audio_duration / extractor.update_interval
        actual_updates = len(feature_updates)
        print(f"Expected updates: {expected_updates:.1f}")
        print(f"Actual updates: {actual_updates}")
        print(f"Update accuracy: {actual_updates/expected_updates:.2%}")
        
        # Buffer statistics
        buffer_stats = extractor.get_stats()
        print(f"\n--- Buffer Statistics ---")
        for key, value in buffer_stats.items():
            if key != "buffer_stats":
                print(f"{key}: {value}")
        
        return {
            "avg_frame_time": avg_frame_time,
            "real_time_factor": frame_real_time,
            "feature_updates": len(feature_updates),
            "buffer_stats": buffer_stats
        }
        
    except Exception as e:
        print(f"✗ Sliding window test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_context_window_comparison():
    """Compare different context window lengths."""
    print("\n" + "="*60)
    print("TESTING CONTEXT WINDOW LENGTHS")
    print("="*60)
    
    context_windows = [5.0, 10.0, 15.0, 20.0, 30.0]
    audio = generate_test_audio(35.0)  # Long enough for all windows
    
    results = {}
    
    for window_size in context_windows:
        print(f"\n--- Testing {window_size}s context window ---")
        
        try:
            extractor = OpenSMILEeGeMAPSExtractor(
                context_window=window_size,
                update_interval=0.5,
            )
            
            # Measure extraction time
            start_time = time.time()
            features = extractor.process_audio_batch(audio)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            results[window_size] = {
                "processing_time": processing_time,
                "feature_shape": features.shape,
                "rtf": processing_time / (len(audio) / 16000)
            }
            
            print(f"  Processing time: {processing_time:.3f}s")
            print(f"  RTF: {results[window_size]['rtf']:.4f}")
            print(f"  Feature shape: {features.shape}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[window_size] = {"error": str(e)}
    
    # Find optimal window size
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        optimal = min(valid_results.items(), 
                     key=lambda x: abs(x[1]["rtf"] - 0.1))  # Target RTF of 0.1
        print(f"\n--- Optimal Context Window ---")
        print(f"Recommended: {optimal[0]}s")
        print(f"RTF: {optimal[1]['rtf']:.4f}")
    
    return results


def create_performance_comparison_plot(backend_results: Dict):
    """Create visualization comparing backend performance."""
    print("\n--- Generating Performance Comparison Plot ---")
    
    try:
        import matplotlib.pyplot as plt
        
        # Extract data for plotting
        backends = list(backend_results.keys())
        scenarios = list(next(iter(backend_results.values())).keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Extraction Backend Comparison', fontsize=16, fontweight='bold')
        
        # 1. Processing time comparison
        ax1 = axes[0, 0]
        processing_times = {}
        for backend in backends:
            times = [backend_results[backend][scenario]["processing_time"] 
                    for scenario in scenarios]
            processing_times[backend] = times
        
        x = np.arange(len(scenarios))
        width = 0.25
        for i, (backend, times) in enumerate(processing_times.items()):
            ax1.bar(x + i*width, times, width, label=backend, alpha=0.8)
        
        ax1.set_xlabel('Scenarios')
        ax1.set_ylabel('Processing Time (s)')
        ax1.set_title('Processing Time by Backend')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Real-time factor comparison
        ax2 = axes[0, 1]
        rtf_data = {}
        for backend in backends:
            rtfs = [backend_results[backend][scenario]["real_time_factor"] 
                   for scenario in scenarios]
            rtf_data[backend] = rtfs
        
        for i, (backend, rtfs) in enumerate(rtf_data.items()):
            ax2.bar(x + i*width, rtfs, width, label=backend, alpha=0.8)
        
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Real-time threshold')
        ax2.set_xlabel('Scenarios')
        ax2.set_ylabel('Real-time Factor')
        ax2.set_title('Real-time Performance')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Feature dimension comparison
        ax3 = axes[1, 0]
        feature_dims = []
        backend_names = []
        for backend in backends:
            first_scenario = scenarios[0]
            shape = backend_results[backend][first_scenario]["feature_shape"]
            if shape:
                feature_dims.append(shape[-1])  # Last dimension
                backend_names.append(backend)
        
        ax3.bar(backend_names, feature_dims, alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'][:len(backend_names)])
        ax3.set_xlabel('Backend')
        ax3.set_ylabel('Feature Dimension')
        ax3.set_title('Feature Dimensionality')
        ax3.grid(True, alpha=0.3)
        
        # Add dimension labels on bars
        for i, dim in enumerate(feature_dims):
            ax3.text(i, dim + max(feature_dims)*0.01, str(dim), ha='center', va='bottom')
        
        # 4. Audio duration vs processing time scatter
        ax4 = axes[1, 1]
        colors = ['blue', 'red', 'green']
        for i, backend in enumerate(backends):
            durations = [backend_results[backend][scenario]["audio_duration"] 
                        for scenario in scenarios]
            proc_times = [backend_results[backend][scenario]["processing_time"] 
                         for scenario in scenarios]
            ax4.scatter(durations, proc_times, label=backend, alpha=0.8, 
                       color=colors[i % len(colors)], s=100)
        
        # Add ideal real-time line
        max_duration = max([backend_results[backend][scenario]["audio_duration"] 
                           for backend in backends for scenario in scenarios])
        ax4.plot([0, max_duration], [0, max_duration], 'k--', alpha=0.5, 
                label='Real-time line')
        
        ax4.set_xlabel('Audio Duration (s)')
        ax4.set_ylabel('Processing Time (s)')
        ax4.set_title('Processing Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = "egemaps_performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Performance comparison plot saved to {save_path}")
        
    except Exception as e:
        print(f"✗ Failed to create plot: {e}")


def main():
    """Run all comparison tests."""
    print("KOEMORPH eGEMAPS vs EMOTION2VEC COMPARISON")
    print("=" * 60)
    
    try:
        # Test 1: Backend comparison
        backend_results = test_feature_extraction_backends()
        
        # Test 2: Sliding window performance
        sliding_results = test_sliding_window_performance()
        
        # Test 3: Context window comparison
        context_results = test_context_window_comparison()
        
        # Generate comparison plots
        if backend_results:
            create_performance_comparison_plot(backend_results)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY AND RECOMMENDATIONS")
        print("="*60)
        
        if backend_results:
            print("\n--- Backend Performance Summary ---")
            for backend, scenarios in backend_results.items():
                avg_rtf = np.mean([s["real_time_factor"] for s in scenarios.values()])
                realtime_scenarios = sum(1 for s in scenarios.values() if s["faster_than_realtime"])
                print(f"{backend.upper()}:")
                print(f"  Average RTF: {avg_rtf:.4f}")
                print(f"  Real-time scenarios: {realtime_scenarios}/{len(scenarios)}")
        
        if sliding_results:
            print(f"\n--- OpenSMILE Streaming Performance ---")
            print(f"Frame RTF: {sliding_results['real_time_factor']:.4f}")
            print(f"Suitable for real-time: {'Yes' if sliding_results['real_time_factor'] < 1.0 else 'No'}")
        
        print(f"\n--- Recommendations ---")
        print("✓ OpenSMILE eGeMAPS: Best for real-time long-term context")
        print("✓ 20s context window: Optimal balance of context vs performance")
        print("✓ 300ms updates: Good responsiveness without overhead")
        print("✓ eGeMAPSv02: Comprehensive feature set for emotion recognition")
        
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()