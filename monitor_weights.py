#!/usr/bin/env python3
"""Monitor and visualize weight updates during training."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional
import argparse

class WeightMonitor:
    """Monitor weight updates during training."""
    
    def __init__(self, model: torch.nn.Module, save_dir: str = "weight_logs"):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize weight history
        self.weight_history: Dict[str, List[float]] = {}
        self.gradient_history: Dict[str, List[float]] = {}
        self.iteration_count = 0
        
        # Get initial weights
        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()
                self.weight_history[name] = []
                self.gradient_history[name] = []
        
        print(f"Monitoring {len(self.initial_weights)} parameters")
        
    def log_weights(self, epoch: int, step: int):
        """Log current weights and gradients."""
        self.iteration_count += 1
        
        weight_stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.initial_weights:
                # Calculate weight change from initial
                weight_change = torch.abs(param.data - self.initial_weights[name]).max().item()
                self.weight_history[name].append(weight_change)
                
                # Calculate gradient norm
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    self.gradient_history[name].append(grad_norm)
                else:
                    self.gradient_history[name].append(0.0)
                
                weight_stats[name] = {
                    'weight_change': weight_change,
                    'grad_norm': grad_norm if param.grad is not None else 0.0,
                    'param_norm': param.data.norm().item(),
                    'param_shape': list(param.shape)
                }
        
        # Save stats
        stats_entry = {
            'epoch': epoch,
            'step': step,
            'iteration': self.iteration_count,
            'timestamp': datetime.now().isoformat(),
            'weights': weight_stats
        }
        
        stats_file = self.save_dir / f"weight_stats_epoch_{epoch}.jsonl"
        with open(stats_file, 'a') as f:
            f.write(json.dumps(stats_entry) + '\n')
        
        return weight_stats
    
    def print_weight_summary(self, epoch: int, step: int):
        """Print summary of weight updates."""
        print(f"\n=== Weight Update Summary (Epoch {epoch}, Step {step}) ===")
        
        # Get current stats
        current_stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.initial_weights:
                weight_change = torch.abs(param.data - self.initial_weights[name]).max().item()
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                current_stats[name] = {'weight_change': weight_change, 'grad_norm': grad_norm}
        
        # Sort by weight change
        sorted_params = sorted(current_stats.items(), key=lambda x: x[1]['weight_change'], reverse=True)
        
        print("Top 10 parameters by weight change:")
        for i, (name, stats) in enumerate(sorted_params[:10]):
            print(f"  {i+1:2d}. {name:50s} | Change: {stats['weight_change']:.6f} | Grad: {stats['grad_norm']:.6f}")
        
        # Overall statistics
        all_changes = [stats['weight_change'] for stats in current_stats.values()]
        all_grads = [stats['grad_norm'] for stats in current_stats.values()]
        
        print(f"\nOverall Statistics:")
        print(f"  Total parameters: {len(all_changes)}")
        print(f"  Max weight change: {max(all_changes):.6f}")
        print(f"  Mean weight change: {np.mean(all_changes):.6f}")
        print(f"  Max gradient norm: {max(all_grads):.6f}")
        print(f"  Mean gradient norm: {np.mean(all_grads):.6f}")
        
        # Check for parameters not updating
        not_updating = [name for name, stats in current_stats.items() if stats['weight_change'] < 1e-8]
        if not_updating:
            print(f"\nParameters not updating ({len(not_updating)}):")
            for name in not_updating[:5]:  # Show first 5
                print(f"  - {name}")
            if len(not_updating) > 5:
                print(f"  ... and {len(not_updating) - 5} more")
    
    def plot_weight_evolution(self, save_plots: bool = True):
        """Plot weight evolution over time."""
        if not self.weight_history or not any(self.weight_history.values()):
            print("No weight history to plot")
            return
        
        # Plot top 10 parameters by final weight change
        final_changes = {name: history[-1] if history else 0.0 
                        for name, history in self.weight_history.items()}
        top_params = sorted(final_changes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot weight changes
        ax1.set_title("Weight Changes Over Time (Top 10 Parameters)")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Max Weight Change from Initial")
        ax1.set_yscale('log')
        
        for name, _ in top_params:
            if name in self.weight_history and self.weight_history[name]:
                ax1.plot(self.weight_history[name], label=name.split('.')[-1], linewidth=2)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot gradient norms
        ax2.set_title("Gradient Norms Over Time (Top 10 Parameters)")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Gradient Norm")
        ax2.set_yscale('log')
        
        for name, _ in top_params:
            if name in self.gradient_history and self.gradient_history[name]:
                ax2.plot(self.gradient_history[name], label=name.split('.')[-1], linewidth=2)
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.save_dir / f"weight_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Weight evolution plot saved to: {plot_file}")
        
        plt.show()
    
    def save_final_report(self):
        """Save final training report."""
        report = {
            'total_iterations': self.iteration_count,
            'monitored_parameters': len(self.initial_weights),
            'final_weight_changes': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.initial_weights:
                weight_change = torch.abs(param.data - self.initial_weights[name]).max().item()
                report['final_weight_changes'][name] = weight_change
        
        report_file = self.save_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Final report saved to: {report_file}")

def analyze_weight_logs(log_dir: str):
    """Analyze existing weight logs."""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory {log_path} does not exist")
        return
    
    # Find all weight stats files
    stats_files = list(log_path.glob("weight_stats_epoch_*.jsonl"))
    if not stats_files:
        print("No weight stats files found")
        return
    
    print(f"Found {len(stats_files)} weight stats files")
    
    # Load all data
    all_data = []
    for file in sorted(stats_files):
        with open(file, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
    
    print(f"Loaded {len(all_data)} weight update records")
    
    # Analyze trends
    if all_data:
        print("\nWeight update trends:")
        first_record = all_data[0]
        last_record = all_data[-1]
        
        # Compare first and last
        for param_name in first_record['weights'].keys():
            if param_name in last_record['weights']:
                initial_change = first_record['weights'][param_name]['weight_change']
                final_change = last_record['weights'][param_name]['weight_change']
                print(f"  {param_name}: {initial_change:.6f} → {final_change:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze weight monitoring logs")
    parser.add_argument("--log-dir", default="weight_logs", help="Directory containing weight logs")
    args = parser.parse_args()
    
    analyze_weight_logs(args.log_dir)