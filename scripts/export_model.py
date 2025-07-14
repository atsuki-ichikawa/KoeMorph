"""
Model export script for deployment optimization.

Exports trained KoeMorph model to various formats for deployment:
- TorchScript for C++ inference
- ONNX for cross-platform deployment
- TensorRT for NVIDIA GPU optimization
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.onnx
from omegaconf import OmegaConf

from src.model.gaussian_face import create_koemorph_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.warning("TensorRT not available. TensorRT export disabled.")


class ModelExporter:
    """Export trained KoeMorph model to various formats."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize model exporter.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Load model
        self.model, self.config = self._load_model()
        self.device = next(self.model.parameters()).device
        
        logger.info(f"Loaded model with {self.model.get_num_parameters():,} parameters")
    
    def _load_model(self) -> Tuple[torch.nn.Module, dict]:
        """Load trained model and configuration."""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Load config
        if self.config_path:
            config = OmegaConf.load(self.config_path)
        elif 'config' in checkpoint:
            config = OmegaConf.create(checkpoint['config'])
        else:
            raise ValueError("No configuration found. Provide config_path or ensure checkpoint contains config.")
        
        # Create and load model
        model = create_koemorph_model(config.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Loaded model from {self.model_path}")
        
        return model, config.model
    
    def _create_dummy_inputs(self, batch_size: int = 1, seq_len: int = 1) -> Tuple[torch.Tensor, ...]:
        """Create dummy inputs for model export."""
        mel_features = torch.randn(batch_size, seq_len, self.config.get('mel_dim', 80))
        prosody_features = torch.randn(batch_size, seq_len, self.config.get('prosody_dim', 4))
        emotion_features = torch.randn(batch_size, seq_len, self.config.get('emotion_dim', 256))
        
        return mel_features, prosody_features, emotion_features
    
    def export_torchscript(self, output_path: str, optimization_level: str = "default"):
        """
        Export model to TorchScript format.
        
        Args:
            output_path: Output path for TorchScript model
            optimization_level: Optimization level ("default", "mobile")
        """
        logger.info("Exporting to TorchScript...")
        
        # Create wrapper for inference
        class KoeMorphWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, mel_features, prosody_features, emotion_features):
                output = self.model(
                    mel_features=mel_features,
                    prosody_features=prosody_features,
                    emotion_features=emotion_features,
                    apply_smoothing=True,
                    apply_constraints=True,
                    return_attention=False,
                )
                return output['blendshapes']
        
        wrapper = KoeMorphWrapper(self.model)
        wrapper.eval()
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs()
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, dummy_inputs)
        
        # Optimize if requested
        if optimization_level == "mobile":
            from torch.utils.mobile_optimizer import optimize_for_mobile
            traced_model = optimize_for_mobile(traced_model)
            logger.info("Applied mobile optimizations")
        
        # Save model
        torch.jit.save(traced_model, output_path)
        logger.info(f"TorchScript model saved to {output_path}")
        
        # Verify the exported model
        self._verify_torchscript(output_path, dummy_inputs)
    
    def _verify_torchscript(self, model_path: str, dummy_inputs: Tuple[torch.Tensor, ...]):
        """Verify exported TorchScript model."""
        loaded_model = torch.jit.load(model_path)
        loaded_model.eval()
        
        # Test inference
        with torch.no_grad():
            original_output = self.model(*dummy_inputs, apply_smoothing=False, apply_constraints=False)['blendshapes']
            traced_output = loaded_model(*dummy_inputs)
        
        # Check outputs are close
        max_diff = torch.max(torch.abs(original_output - traced_output)).item()
        logger.info(f"TorchScript verification: max difference = {max_diff:.6f}")
        
        if max_diff > 1e-5:
            logger.warning("Large difference detected in TorchScript export")
    
    def export_onnx(
        self, 
        output_path: str, 
        opset_version: int = 11,
        dynamic_axes: bool = False
    ):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output path for ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic axes for variable sequence lengths
        """
        logger.info("Exporting to ONNX...")
        
        # Create wrapper for clean ONNX export
        class KoeMorphONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, mel_features, prosody_features, emotion_features):
                # Disable temporal smoothing and constraints for ONNX
                output = self.model(
                    mel_features=mel_features,
                    prosody_features=prosody_features,
                    emotion_features=emotion_features,
                    apply_smoothing=False,
                    apply_constraints=False,
                    return_attention=False,
                )
                return output['blendshapes']
        
        wrapper = KoeMorphONNXWrapper(self.model)
        wrapper.eval()
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs()
        
        # Define input/output names
        input_names = ['mel_features', 'prosody_features', 'emotion_features']
        output_names = ['blendshapes']
        
        # Define dynamic axes if requested
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'mel_features': {1: 'sequence_length'},
                'prosody_features': {1: 'sequence_length'},
                'emotion_features': {1: 'sequence_length'},
            }
        
        # Export to ONNX
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
        )
        
        logger.info(f"ONNX model saved to {output_path}")
        
        # Verify ONNX model
        self._verify_onnx(output_path, dummy_inputs)
    
    def _verify_onnx(self, model_path: str, dummy_inputs: Tuple[torch.Tensor, ...]):
        """Verify exported ONNX model."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            logger.warning("ONNX/ONNXRuntime not available for verification")
            return
        
        # Load and check ONNX model
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference with ONNXRuntime
        ort_session = ort.InferenceSession(model_path)
        
        # Prepare inputs
        ort_inputs = {
            'mel_features': dummy_inputs[0].numpy(),
            'prosody_features': dummy_inputs[1].numpy(),
            'emotion_features': dummy_inputs[2].numpy(),
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output = torch.from_numpy(ort_outputs[0])
        
        # Compare with original
        with torch.no_grad():
            original_output = self.model(*dummy_inputs, apply_smoothing=False, apply_constraints=False)['blendshapes']
        
        max_diff = torch.max(torch.abs(original_output - onnx_output)).item()
        logger.info(f"ONNX verification: max difference = {max_diff:.6f}")
        
        if max_diff > 1e-4:
            logger.warning("Large difference detected in ONNX export")
    
    def export_tensorrt(
        self, 
        onnx_path: str, 
        output_path: str,
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30,  # 1GB
        fp16: bool = True,
    ):
        """
        Export ONNX model to TensorRT format.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Output path for TensorRT engine
            max_batch_size: Maximum batch size
            workspace_size: Workspace size in bytes
            fp16: Whether to use FP16 precision
        """
        if not HAS_TENSORRT:
            logger.error("TensorRT not available. Install TensorRT to use this feature.")
            return
        
        logger.info("Exporting to TensorRT...")
        
        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Using FP16 precision")
        else:
            logger.info("Using FP32 precision")
        
        # Build engine
        logger.info("Building TensorRT engine (this may take a while)...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved to {output_path}")
        
        # Cleanup
        del engine
        del builder
        del network
        del parser
    
    def benchmark_model(self, num_runs: int = 100, warmup_runs: int = 10):
        """
        Benchmark the original model performance.
        
        Args:
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
        """
        logger.info(f"Benchmarking model ({num_runs} runs)...")
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs()
        dummy_inputs = [x.to(self.device) for x in dummy_inputs]
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(*dummy_inputs)
        
        # Benchmark
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(*dummy_inputs)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Report statistics
        import numpy as np
        mean_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Mean: {mean_time:.2f} ± {std_time:.2f} ms")
        logger.info(f"  Min: {min_time:.2f} ms")
        logger.info(f"  Max: {max_time:.2f} ms")
        logger.info(f"  Target FPS: {1000/mean_time:.1f}")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export KoeMorph model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="exported_models",
                       help="Output directory for exported models")
    
    # Optional arguments
    parser.add_argument("--config_path", type=str,
                       help="Path to model configuration file")
    parser.add_argument("--formats", nargs="+", 
                       choices=["torchscript", "onnx", "tensorrt", "all"],
                       default=["torchscript", "onnx"],
                       help="Export formats")
    
    # Format-specific options
    parser.add_argument("--mobile_optimize", action="store_true",
                       help="Optimize TorchScript for mobile")
    parser.add_argument("--onnx_opset", type=int, default=11,
                       help="ONNX opset version")
    parser.add_argument("--dynamic_axes", action="store_true",
                       help="Use dynamic axes in ONNX export")
    parser.add_argument("--tensorrt_fp16", action="store_true",
                       help="Use FP16 precision for TensorRT")
    
    # Other options
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark the original model")
    
    args = parser.parse_args()
    
    # Check model file exists
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize exporter
    try:
        exporter = ModelExporter(args.model_path, args.config_path)
        
        # Benchmark if requested
        if args.benchmark:
            exporter.benchmark_model()
        
        # Handle "all" format
        formats = args.formats
        if "all" in formats:
            formats = ["torchscript", "onnx", "tensorrt"]
        
        # Export to requested formats
        onnx_path = None
        
        if "torchscript" in formats:
            torchscript_path = output_dir / "model.pt"
            optimization = "mobile" if args.mobile_optimize else "default"
            exporter.export_torchscript(str(torchscript_path), optimization)
        
        if "onnx" in formats:
            onnx_path = output_dir / "model.onnx"
            exporter.export_onnx(
                str(onnx_path), 
                opset_version=args.onnx_opset,
                dynamic_axes=args.dynamic_axes
            )
        
        if "tensorrt" in formats:
            if onnx_path is None:
                # Need to export ONNX first
                onnx_path = output_dir / "model.onnx"
                exporter.export_onnx(str(onnx_path))
            
            tensorrt_path = output_dir / "model.trt"
            exporter.export_tensorrt(
                str(onnx_path),
                str(tensorrt_path),
                fp16=args.tensorrt_fp16
            )
        
        logger.info(f"Export completed! Models saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()