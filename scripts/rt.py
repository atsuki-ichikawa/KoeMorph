"""
Real-time inference script for GaussianFace model.

Captures audio from microphone, processes it in real-time, and outputs
blendshape coefficients via UDP/OSC or files.
"""

import argparse
import json
import logging
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    print("Warning: pyaudio not available. Real-time audio capture disabled.")

try:
    from pythonosc import udp_client
    HAS_OSC = True
except ImportError:
    HAS_OSC = False
    print("Warning: python-osc not available. OSC output disabled.")

from src.features.emotion2vec import Emotion2VecExtractor
from src.features.prosody import ProsodyExtractor
from src.features.stft import MelSpectrogramExtractor
from src.model.gaussian_face import create_gaussian_face_model


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RingBuffer:
    """Ring buffer for audio samples."""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.write_ptr = 0
        self.read_ptr = 0
        self.available = 0
    
    def write(self, data: np.ndarray):
        """Write data to buffer."""
        data = data.astype(np.float32)
        write_size = min(len(data), self.size - self.available)
        
        if write_size == 0:
            return  # Buffer full
        
        # Handle wrapping
        end_ptr = self.write_ptr + write_size
        if end_ptr <= self.size:
            self.buffer[self.write_ptr:end_ptr] = data[:write_size]
        else:
            # Split write
            first_part = self.size - self.write_ptr
            self.buffer[self.write_ptr:] = data[:first_part]
            self.buffer[:write_size - first_part] = data[first_part:write_size]
        
        self.write_ptr = end_ptr % self.size
        self.available = min(self.available + write_size, self.size)
    
    def read(self, size: int) -> Optional[np.ndarray]:
        """Read data from buffer."""
        if self.available < size:
            return None
        
        # Handle wrapping
        end_ptr = self.read_ptr + size
        if end_ptr <= self.size:
            data = self.buffer[self.read_ptr:end_ptr].copy()
        else:
            # Split read
            first_part = self.size - self.read_ptr
            data = np.concatenate([
                self.buffer[self.read_ptr:],
                self.buffer[:size - first_part]
            ])
        
        self.read_ptr = end_ptr % self.size
        self.available -= size
        
        return data


class AudioCapture:
    """Real-time audio capture using PyAudio."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        audio_queue: Optional[queue.Queue] = None,
    ):
        if not HAS_PYAUDIO:
            raise RuntimeError("PyAudio not available. Install with: pip install pyaudio")
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = audio_queue or queue.Queue()
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Put in queue for processing
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            logger.warning("Audio queue full, dropping samples")
        
        return (None, pyaudio.paContinue)
    
    def start(self):
        """Start audio capture."""
        if self.is_recording:
            return
        
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        
        self.stream.start_stream()
        self.is_recording = True
        logger.info(f"Started audio capture: {self.sample_rate}Hz, {self.chunk_size} samples/chunk")
    
    def stop(self):
        """Stop audio capture."""
        if not self.is_recording:
            return
        
        self.stream.stop_stream()
        self.stream.close()
        self.is_recording = False
        logger.info("Stopped audio capture")
    
    def __del__(self):
        """Cleanup."""
        self.stop()
        if hasattr(self, 'audio'):
            self.audio.terminate()


class BlendshapeStreamer:
    """Streams blendshape data via UDP or OSC."""
    
    def __init__(
        self,
        output_mode: str = "udp",  # udp, osc, file
        host: str = "127.0.0.1",
        port: int = 9001,
        osc_address: str = "/blendshapes",
        output_file: Optional[str] = None,
    ):
        self.output_mode = output_mode
        self.host = host
        self.port = port
        self.osc_address = osc_address
        self.output_file = output_file
        
        # Setup output
        if output_mode == "udp":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif output_mode == "osc":
            if not HAS_OSC:
                raise RuntimeError("python-osc not available. Install with: pip install python-osc")
            self.osc_client = udp_client.SimpleUDPClient(host, port)
        elif output_mode == "file":
            if output_file:
                self.file_handle = open(output_file, 'w')
            else:
                raise ValueError("output_file required for file mode")
        else:
            raise ValueError(f"Unknown output mode: {output_mode}")
        
        logger.info(f"Blendshape streamer initialized: {output_mode} -> {host}:{port}")
    
    def send(self, blendshapes: np.ndarray, timestamp: float):
        """Send blendshape data."""
        if self.output_mode == "udp":
            # Send as JSON over UDP
            data = {
                "timestamp": timestamp,
                "blendshapes": blendshapes.tolist(),
            }
            message = json.dumps(data).encode('utf-8')
            self.socket.sendto(message, (self.host, self.port))
        
        elif self.output_mode == "osc":
            # Send via OSC
            self.osc_client.send_message(self.osc_address, blendshapes.tolist())
        
        elif self.output_mode == "file":
            # Write to file
            data = {
                "timestamp": timestamp,
                "blendshapes": blendshapes.tolist(),
            }
            self.file_handle.write(json.dumps(data) + '\n')
            self.file_handle.flush()
    
    def close(self):
        """Close streamer."""
        if hasattr(self, 'socket'):
            self.socket.close()
        elif hasattr(self, 'file_handle'):
            self.file_handle.close()


class RealTimeInference:
    """Real-time blendshape inference system."""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        sample_rate: int = 16000,
        target_fps: float = 30.0,
        buffer_duration: float = 2.0,
        device: str = "auto",
    ):
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.frame_samples = int(sample_rate / target_fps)
        self.device = self._setup_device(device)
        
        # Setup audio buffer
        buffer_size = int(buffer_duration * sample_rate)
        self.audio_buffer = RingBuffer(buffer_size)
        
        # Load model
        self.model = self._load_model(model_path, config_path)
        self.feature_extractors = self._setup_feature_extractors()
        
        # State
        self.prev_blendshapes = None
        self.frame_count = 0
        
        logger.info(f"Real-time inference initialized: {target_fps} FPS, {sample_rate} Hz")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device = torch.device(device)
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self, model_path: str, config_path: Optional[str]):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        if config_path:
            config = OmegaConf.load(config_path)
        elif 'config' in checkpoint:
            config = OmegaConf.create(checkpoint['config'])
        else:
            # Use default config
            config = OmegaConf.create({
                'model': {
                    'd_model': 256,
                    'num_heads': 8,
                    'mel_dim': 80,
                    'prosody_dim': 4,
                    'emotion_dim': 256,
                }
            })
        
        # Create model
        model = create_gaussian_face_model(config.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model parameters: {model.get_num_parameters():,}")
        
        return model
    
    def _setup_feature_extractors(self):
        """Setup feature extraction modules."""
        extractors = {}
        
        # Mel-spectrogram extractor
        extractors['mel'] = MelSpectrogramExtractor(
            sample_rate=self.sample_rate,
            target_fps=self.target_fps,
        ).to(self.device)
        
        # Prosody extractor
        extractors['prosody'] = ProsodyExtractor(
            sample_rate=self.sample_rate,
            target_fps=self.target_fps,
        ).to(self.device)
        
        # Emotion2vec extractor
        extractors['emotion2vec'] = Emotion2VecExtractor(
            model_name="dummy",  # Use dummy for real-time
            target_fps=self.target_fps,
            sample_rate=self.sample_rate,
            output_dim=256,
        ).to(self.device)
        
        return extractors
    
    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process audio chunk and add to buffer."""
        self.audio_buffer.write(audio_chunk)
    
    def inference_step(self) -> Optional[np.ndarray]:
        """Perform single inference step if enough data available."""
        # Check if we have enough audio for processing
        required_samples = self.frame_samples
        audio_data = self.audio_buffer.read(required_samples)
        
        if audio_data is None:
            return None
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features
            mel_features = self.feature_extractors['mel'](audio_tensor)
            prosody_features = self.feature_extractors['prosody'](audio_tensor)
            emotion_features = self.feature_extractors['emotion2vec'](audio_tensor)
            
            # Ensure features have time dimension
            if mel_features.dim() == 2:
                mel_features = mel_features.unsqueeze(1)
            if prosody_features.dim() == 2:
                prosody_features = prosody_features.unsqueeze(1)
            if emotion_features.dim() == 2:
                emotion_features = emotion_features.unsqueeze(1)
            
            # Run inference
            blendshapes = self.model.inference_step(
                mel_features, prosody_features, emotion_features, self.prev_blendshapes
            )
            
            # Update state
            self.prev_blendshapes = blendshapes.clone()
            self.frame_count += 1
            
            # Convert to numpy
            blendshapes_np = blendshapes.squeeze(0).cpu().numpy()
            
            return blendshapes_np
    
    def reset(self):
        """Reset inference state."""
        self.prev_blendshapes = None
        self.frame_count = 0
        self.model.reset_temporal_state()
        logger.info("Reset inference state")


def main():
    """Main real-time inference function."""
    parser = argparse.ArgumentParser(description="Real-time GaussianFace inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str,
                       help="Path to model config file")
    
    # Audio arguments
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Audio sample rate")
    parser.add_argument("--target_fps", type=float, default=30.0,
                       help="Target blendshape frame rate")
    parser.add_argument("--chunk_size", type=int, default=1024,
                       help="Audio chunk size for capture")
    
    # Output arguments
    parser.add_argument("--output_mode", type=str, default="udp",
                       choices=["udp", "osc", "file"],
                       help="Output mode for blendshapes")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Output host")
    parser.add_argument("--port", type=int, default=9001,
                       help="Output port")
    parser.add_argument("--output_file", type=str,
                       help="Output file for file mode")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Computation device")
    parser.add_argument("--duration", type=float,
                       help="Duration to run (seconds), None for infinite")
    parser.add_argument("--no_audio", action="store_true",
                       help="Disable audio capture (test mode)")
    
    args = parser.parse_args()
    
    # Check model file exists
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    try:
        # Initialize inference system
        inference = RealTimeInference(
            model_path=args.model_path,
            config_path=args.config_path,
            sample_rate=args.sample_rate,
            target_fps=args.target_fps,
            device=args.device,
        )
        
        # Initialize output streamer
        streamer = BlendshapeStreamer(
            output_mode=args.output_mode,
            host=args.host,
            port=args.port,
            output_file=args.output_file,
        )
        
        # Initialize audio capture (if enabled)
        audio_queue = queue.Queue(maxsize=100)
        audio_capture = None
        
        if not args.no_audio and HAS_PYAUDIO:
            audio_capture = AudioCapture(
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size,
                audio_queue=audio_queue,
            )
            audio_capture.start()
        
        # Main processing loop
        logger.info("Starting real-time inference...")
        start_time = time.time()
        frame_times = []
        
        try:
            while True:
                loop_start = time.time()
                
                # Check duration limit
                if args.duration and (time.time() - start_time) > args.duration:
                    break
                
                # Process audio chunks from queue
                processed_audio = False
                while not audio_queue.empty():
                    try:
                        audio_chunk = audio_queue.get_nowait()
                        inference.process_audio_chunk(audio_chunk)
                        processed_audio = True
                    except queue.Empty:
                        break
                
                # Generate dummy audio if no real audio
                if not processed_audio and args.no_audio:
                    # Generate dummy audio chunk
                    dummy_chunk = np.random.randn(args.chunk_size).astype(np.float32) * 0.01
                    inference.process_audio_chunk(dummy_chunk)
                
                # Perform inference
                blendshapes = inference.inference_step()
                
                if blendshapes is not None:
                    # Send blendshapes
                    timestamp = time.time()
                    streamer.send(blendshapes, timestamp)
                    
                    # Log progress
                    if inference.frame_count % 30 == 0:  # Every second at 30 FPS
                        avg_time = np.mean(frame_times[-30:]) if frame_times else 0
                        logger.info(f"Frame {inference.frame_count}, avg time: {avg_time*1000:.1f}ms")
                
                # Record timing
                frame_time = time.time() - loop_start
                frame_times.append(frame_time)
                
                # Keep only recent times for averaging
                if len(frame_times) > 100:
                    frame_times = frame_times[-100:]
                
                # Sleep to maintain target FPS
                target_frame_time = 1.0 / args.target_fps
                sleep_time = target_frame_time - frame_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            if audio_capture:
                audio_capture.stop()
            streamer.close()
            
            # Print statistics
            if frame_times:
                avg_time = np.mean(frame_times) * 1000
                max_time = np.max(frame_times) * 1000
                logger.info(f"Average frame time: {avg_time:.1f}ms")
                logger.info(f"Maximum frame time: {max_time:.1f}ms")
                logger.info(f"Processed {inference.frame_count} frames")
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()