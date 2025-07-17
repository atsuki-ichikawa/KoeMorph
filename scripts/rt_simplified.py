"""
Real-time inference script for simplified KoeMorph model.

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
import librosa

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

from src.model.simplified_model import SimplifiedKoeMorphModel


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


class AudioFileReader:
    """Audio file reader that simulates real-time processing."""
    
    def __init__(
        self,
        file_path: str,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        audio_queue: Optional[queue.Queue] = None,
    ):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = audio_queue or queue.Queue()
        
        # Load audio file
        self.audio_data, self.original_sr = librosa.load(file_path, sr=sample_rate, mono=True)
        self.current_pos = 0
        self.is_playing = False
        self.playback_thread = None
        
        logger.info(f"Loaded audio file: {file_path}")
        logger.info(f"Duration: {len(self.audio_data) / sample_rate:.2f}s, SR: {sample_rate}Hz")
    
    def _playback_loop(self):
        """Playback loop that simulates real-time audio streaming."""
        chunk_samples = self.chunk_size
        chunk_duration = chunk_samples / self.sample_rate
        
        while self.is_playing and self.current_pos < len(self.audio_data):
            # Get next chunk
            end_pos = min(self.current_pos + chunk_samples, len(self.audio_data))
            chunk = self.audio_data[self.current_pos:end_pos]
            
            # Pad if necessary
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
            
            # Put in queue
            try:
                self.audio_queue.put_nowait(chunk.astype(np.float32))
            except queue.Full:
                logger.warning("Audio queue full, dropping samples")
            
            self.current_pos = end_pos
            
            # Sleep to simulate real-time playback
            time.sleep(chunk_duration)
        
        self.is_playing = False
    
    def start(self):
        """Start audio file playback."""
        if self.is_playing:
            return
        
        self.is_playing = True
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
        logger.info("Started audio file playback")
    
    def stop(self):
        """Stop audio file playback."""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join()
        logger.info("Stopped audio file playback")
    
    def reset(self):
        """Reset playback to beginning."""
        self.current_pos = 0
        logger.info("Reset audio file playback")


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


class SimplifiedRealTimeInference:
    """Real-time blendshape inference system using simplified model."""
    
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        target_fps: float = 30.0,
        buffer_duration: float = 2.0,
        device: str = "auto",
        audio_length: int = 16000,  # 1 second of audio
    ):
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.audio_length = audio_length
        self.device = self._setup_device(device)
        
        # Setup audio buffer
        buffer_size = int(buffer_duration * sample_rate)
        self.audio_buffer = RingBuffer(buffer_size)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # State
        self.frame_count = 0
        
        logger.info(f"Simplified real-time inference initialized: {target_fps} FPS, {sample_rate} Hz")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device = torch.device(device)
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self, model_path: str):
        """Load trained simplified model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create simplified model
        model = SimplifiedKoeMorphModel(
            d_model=256,
            num_blendshapes=52,
            sample_rate=self.sample_rate,
            target_fps=self.target_fps,
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded simplified model from {model_path}")
        logger.info(f"Model parameters: {model.get_num_parameters():,}")
        
        return model
    
    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process audio chunk and add to buffer."""
        self.audio_buffer.write(audio_chunk)
    
    def inference_step(self) -> Optional[np.ndarray]:
        """Perform single inference step if enough data available."""
        # Check if we have enough audio for processing
        audio_data = self.audio_buffer.read(self.audio_length)
        
        if audio_data is None:
            return None
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Run inference
            blendshapes = self.model(audio_tensor)
            
            # Update state
            self.frame_count += 1
            
            # Convert to numpy
            blendshapes_np = blendshapes.squeeze(0).cpu().numpy()
            
            return blendshapes_np
    
    def reset(self):
        """Reset inference state."""
        self.frame_count = 0
        self.model.reset_temporal_state()
        logger.info("Reset inference state")


def main():
    """Main real-time inference function."""
    parser = argparse.ArgumentParser(description="Real-time simplified KoeMorph inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    
    # Audio arguments
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Audio sample rate")
    parser.add_argument("--target_fps", type=float, default=30.0,
                       help="Target blendshape frame rate")
    parser.add_argument("--chunk_size", type=int, default=1024,
                       help="Audio chunk size for capture")
    parser.add_argument("--audio_length", type=int, default=16000,
                       help="Audio length for each inference (samples)")
    parser.add_argument("--input_file", type=str,
                       help="Input audio file for processing (instead of microphone)")
    
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
        inference = SimplifiedRealTimeInference(
            model_path=args.model_path,
            sample_rate=args.sample_rate,
            target_fps=args.target_fps,
            device=args.device,
            audio_length=args.audio_length,
        )
        
        # Initialize output streamer
        streamer = BlendshapeStreamer(
            output_mode=args.output_mode,
            host=args.host,
            port=args.port,
            output_file=args.output_file,
        )
        
        # Initialize audio input (file or microphone)
        audio_queue = queue.Queue(maxsize=100)
        audio_source = None
        
        if args.input_file:
            # Use audio file input
            if not Path(args.input_file).exists():
                logger.error(f"Input file not found: {args.input_file}")
                return
            
            audio_source = AudioFileReader(
                file_path=args.input_file,
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size,
                audio_queue=audio_queue,
            )
            audio_source.start()
            
        elif not args.no_audio and HAS_PYAUDIO:
            # Use microphone input
            audio_source = AudioCapture(
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size,
                audio_queue=audio_queue,
            )
            audio_source.start()
        
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
                if not processed_audio and args.no_audio and not args.input_file:
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
            if audio_source:
                audio_source.stop()
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