"""
Debugging and monitoring utilities for emotion processing in KoeMorph.

Provides comprehensive monitoring, logging, and debugging capabilities
for the dual-stream emotion processing pipeline.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import psutil
import threading


class EmotionProcessingMonitor:
    """
    Monitor for tracking emotion processing performance and debugging issues.
    """
    
    def __init__(
        self,
        log_dir: str = "logs/emotion_monitor",
        max_history: int = 1000,
        enable_plotting: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize emotion processing monitor.
        
        Args:
            log_dir: Directory for saving logs and plots
            max_history: Maximum number of processing records to keep
            enable_plotting: Whether to generate plots
            verbose: Whether to print detailed logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_history = max_history
        self.enable_plotting = enable_plotting
        self.verbose = verbose
        
        # Processing history
        self.processing_history = deque(maxlen=max_history)
        self.backend_usage = defaultdict(int)
        self.error_log = deque(maxlen=100)
        
        # Performance metrics
        self.processing_times = defaultdict(list)
        self.memory_usage = deque(maxlen=max_history)
        self.gpu_usage = deque(maxlen=max_history)
        
        # Emotion statistics
        self.emotion_predictions = defaultdict(list)
        self.blendshape_activations = defaultdict(list)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Thread lock for concurrent access
        self.lock = threading.Lock()
        
        self.logger.info("EmotionProcessingMonitor initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated logger for emotion monitoring."""
        logger = logging.getLogger("emotion_monitor")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # File handler
        log_file = self.log_dir / "emotion_processing.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_processing_start(
        self,
        audio_shape: Tuple[int, ...],
        backend: str,
        config: Dict[str, Any],
    ) -> str:
        """
        Log the start of emotion processing.
        
        Returns:
            Processing ID for tracking
        """
        processing_id = f"proc_{time.time():.6f}"
        
        with self.lock:
            record = {
                "id": processing_id,
                "start_time": time.time(),
                "audio_shape": audio_shape,
                "backend": backend,
                "config": config,
                "system_memory": psutil.virtual_memory().percent,
                "status": "started"
            }
            
            # Add GPU info if available
            if torch.cuda.is_available():
                record["gpu_memory"] = torch.cuda.memory_allocated() / 1024**3  # GB
                record["gpu_utilization"] = self._get_gpu_utilization()
            
            self.processing_history.append(record)
            self.backend_usage[backend] += 1
            
        if self.verbose:
            self.logger.info(
                f"Started processing {processing_id}: {backend} backend, "
                f"audio shape {audio_shape}"
            )
            
        return processing_id
    
    def log_processing_end(
        self,
        processing_id: str,
        success: bool,
        results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Log the end of emotion processing."""
        end_time = time.time()
        
        with self.lock:
            # Find the processing record
            record = None
            for r in reversed(self.processing_history):
                if r["id"] == processing_id:
                    record = r
                    break
            
            if record is None:
                self.logger.warning(f"Processing record {processing_id} not found")
                return
            
            # Update record
            record["end_time"] = end_time
            record["processing_time"] = end_time - record["start_time"]
            record["success"] = success
            record["status"] = "completed" if success else "failed"
            
            if results:
                record["results"] = results
                
                # Extract emotion predictions and blendshape data
                if "predictions" in results:
                    for emotion, confidence in results["predictions"].items():
                        self.emotion_predictions[emotion].append(confidence)
                
                if "blendshape_weights" in results:
                    weights = results["blendshape_weights"]
                    if isinstance(weights, (np.ndarray, torch.Tensor)):
                        for i, weight in enumerate(weights.flatten()):
                            self.blendshape_activations[f"bs_{i}"].append(float(weight))
            
            if error:
                record["error"] = error
                self.error_log.append({
                    "time": end_time,
                    "processing_id": processing_id,
                    "backend": record["backend"],
                    "error": error
                })
            
            # Update performance metrics
            backend = record["backend"]
            self.processing_times[backend].append(record["processing_time"])
            
            # System metrics
            self.memory_usage.append(psutil.virtual_memory().percent)
            if torch.cuda.is_available():
                self.gpu_usage.append(self._get_gpu_utilization())
        
        if self.verbose:
            status = "successfully" if success else "with error"
            self.logger.info(
                f"Completed processing {processing_id} {status} "
                f"in {record['processing_time']:.3f}s"
            )
            
            if error:
                self.logger.error(f"Error in {processing_id}: {error}")
    
    def log_fallback_usage(self, from_backend: str, to_backend: str, reason: str):
        """Log when fallback is used."""
        with self.lock:
            fallback_record = {
                "time": time.time(),
                "from_backend": from_backend,
                "to_backend": to_backend,
                "reason": reason
            }
            
            # Add to error log for tracking
            self.error_log.append(fallback_record)
        
        self.logger.warning(
            f"Fallback from {from_backend} to {to_backend}: {reason}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        with self.lock:
            stats = {
                "total_processed": len(self.processing_history),
                "backend_usage": dict(self.backend_usage),
                "processing_times": {},
                "success_rate": 0.0,
                "error_count": len(self.error_log),
                "system_metrics": {
                    "avg_memory_usage": np.mean(list(self.memory_usage)) if self.memory_usage else 0,
                    "avg_gpu_usage": np.mean(list(self.gpu_usage)) if self.gpu_usage else 0,
                },
                "emotion_stats": {},
                "blendshape_stats": {}
            }
            
            # Processing time statistics
            for backend, times in self.processing_times.items():
                if times:
                    stats["processing_times"][backend] = {
                        "mean": np.mean(times),
                        "median": np.median(times),
                        "std": np.std(times),
                        "min": np.min(times),
                        "max": np.max(times),
                        "count": len(times)
                    }
            
            # Success rate
            successful = sum(1 for r in self.processing_history if r.get("success", False))
            if self.processing_history:
                stats["success_rate"] = successful / len(self.processing_history)
            
            # Emotion statistics
            for emotion, confidences in self.emotion_predictions.items():
                if confidences:
                    stats["emotion_stats"][emotion] = {
                        "mean_confidence": np.mean(confidences),
                        "activation_frequency": np.mean(np.array(confidences) > 0.5),
                        "max_confidence": np.max(confidences)
                    }
            
            # Blendshape statistics
            for bs_name, activations in self.blendshape_activations.items():
                if activations:
                    stats["blendshape_stats"][bs_name] = {
                        "mean_activation": np.mean(activations),
                        "activation_frequency": np.mean(np.array(activations) > 0.1),
                        "max_activation": np.max(activations)
                    }
        
        return stats
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive monitoring report."""
        stats = self.get_statistics()
        
        report_lines = [
            "# Emotion Processing Monitoring Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            f"- Total processed: {stats['total_processed']}",
            f"- Success rate: {stats['success_rate']:.2%}",
            f"- Error count: {stats['error_count']}",
            "",
            "## Backend Usage",
        ]
        
        for backend, count in stats["backend_usage"].items():
            percentage = count / stats["total_processed"] * 100 if stats["total_processed"] > 0 else 0
            report_lines.append(f"- {backend}: {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "## Performance Metrics",
        ])
        
        for backend, times in stats["processing_times"].items():
            report_lines.extend([
                f"### {backend}",
                f"- Mean time: {times['mean']:.3f}s",
                f"- Median time: {times['median']:.3f}s",
                f"- Std deviation: {times['std']:.3f}s",
                f"- Min/Max: {times['min']:.3f}s / {times['max']:.3f}s",
                f"- Sample count: {times['count']}",
                ""
            ])
        
        # System metrics
        report_lines.extend([
            "## System Metrics",
            f"- Average memory usage: {stats['system_metrics']['avg_memory_usage']:.1f}%",
            f"- Average GPU usage: {stats['system_metrics']['avg_gpu_usage']:.1f}%",
            "",
            "## Top Emotions",
        ])
        
        # Sort emotions by activation frequency
        emotion_items = [(emotion, data["activation_frequency"]) 
                        for emotion, data in stats["emotion_stats"].items()]
        emotion_items.sort(key=lambda x: x[1], reverse=True)
        
        for emotion, freq in emotion_items[:5]:
            report_lines.append(f"- {emotion}: {freq:.2%} activation rate")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        if save_path is None:
            save_path = self.log_dir / f"report_{int(time.time())}.md"
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Report saved to {save_path}")
        return report_text
    
    def plot_performance_metrics(self, save_dir: Optional[str] = None):
        """Generate performance visualization plots."""
        if not self.enable_plotting:
            return
        
        save_dir = Path(save_dir) if save_dir else (self.log_dir / "plots")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing time distribution
        if self.processing_times:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Processing time by backend
            ax1 = axes[0, 0]
            backend_times = []
            backend_labels = []
            
            for backend, times in self.processing_times.items():
                if times:
                    backend_times.append(times)
                    backend_labels.append(backend)
            
            if backend_times:
                ax1.boxplot(backend_times, labels=backend_labels)
                ax1.set_title('Processing Time Distribution by Backend')
                ax1.set_ylabel('Time (seconds)')
                ax1.tick_params(axis='x', rotation=45)
            
            # 2. Backend usage pie chart
            ax2 = axes[0, 1]
            if self.backend_usage:
                ax2.pie(self.backend_usage.values(), labels=self.backend_usage.keys(), autopct='%1.1f%%')
                ax2.set_title('Backend Usage Distribution')
            
            # 3. System resource usage over time
            ax3 = axes[1, 0]
            if self.memory_usage:
                ax3.plot(list(self.memory_usage), label='Memory %', alpha=0.7)
            if self.gpu_usage:
                ax3.plot(list(self.gpu_usage), label='GPU %', alpha=0.7)
            ax3.set_title('System Resource Usage Over Time')
            ax3.set_ylabel('Usage %')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Success rate over time
            ax4 = axes[1, 1]
            window_size = 50
            success_rates = []
            
            for i in range(window_size, len(self.processing_history)):
                window = list(self.processing_history)[i-window_size:i]
                successes = sum(1 for r in window if r.get("success", False))
                success_rates.append(successes / window_size)
            
            if success_rates:
                ax4.plot(success_rates)
                ax4.set_title(f'Success Rate (Rolling Window: {window_size})')
                ax4.set_ylabel('Success Rate')
                ax4.set_ylim(0, 1)
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / "performance_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Emotion activation heatmap
        if self.emotion_predictions:
            emotions = list(self.emotion_predictions.keys())
            recent_predictions = []
            
            # Get recent predictions (last 100)
            for emotion in emotions:
                recent = self.emotion_predictions[emotion][-100:] if self.emotion_predictions[emotion] else [0]
                recent_predictions.append(recent)
            
            if recent_predictions:
                # Pad to same length
                max_len = max(len(pred) for pred in recent_predictions)
                padded_predictions = []
                
                for pred in recent_predictions:
                    if len(pred) < max_len:
                        pred = pred + [0] * (max_len - len(pred))
                    padded_predictions.append(pred)
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    padded_predictions,
                    yticklabels=emotions,
                    xticklabels=False,
                    cmap='viridis',
                    cbar_kws={'label': 'Confidence'}
                )
                plt.title('Recent Emotion Predictions Heatmap')
                plt.xlabel('Time Steps (Recent 100)')
                plt.ylabel('Emotions')
                plt.tight_layout()
                plt.savefig(save_dir / "emotion_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        self.logger.info(f"Performance plots saved to {save_dir}")
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            if torch.cuda.is_available():
                # This is a simplified approach - in practice, you might want to use nvidia-ml-py
                return torch.cuda.utilization()
        except:
            pass
        return 0.0
    
    def export_data(self, export_path: Optional[str] = None) -> str:
        """Export monitoring data to JSON."""
        if export_path is None:
            export_path = self.log_dir / f"monitoring_data_{int(time.time())}.json"
        
        with self.lock:
            data = {
                "processing_history": list(self.processing_history),
                "backend_usage": dict(self.backend_usage),
                "error_log": list(self.error_log),
                "statistics": self.get_statistics(),
                "export_timestamp": time.time()
            }
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring data exported to {export_path}")
        return str(export_path)
    
    def reset_metrics(self):
        """Reset all monitoring metrics."""
        with self.lock:
            self.processing_history.clear()
            self.backend_usage.clear()
            self.error_log.clear()
            self.processing_times.clear()
            self.memory_usage.clear()
            self.gpu_usage.clear()
            self.emotion_predictions.clear()
            self.blendshape_activations.clear()
        
        self.logger.info("All monitoring metrics reset")


# Global monitor instance
_global_monitor: Optional[EmotionProcessingMonitor] = None


def get_monitor() -> EmotionProcessingMonitor:
    """Get the global emotion processing monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = EmotionProcessingMonitor()
    return _global_monitor


def initialize_monitor(config: Dict[str, Any]) -> EmotionProcessingMonitor:
    """Initialize the global monitor with configuration."""
    global _global_monitor
    _global_monitor = EmotionProcessingMonitor(**config)
    return _global_monitor