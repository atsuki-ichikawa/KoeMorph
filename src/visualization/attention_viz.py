"""
Attention visualization tools for dual-stream KoeMorph model.

Provides visualization for mel-frequency attention patterns and 
emotion2vec attention patterns, showing which frequency bands 
and emotion features contribute to specific blendshapes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..model.dual_stream_attention import (
    MOUTH_INDICES, EXPRESSION_INDICES, ARKIT_BLENDSHAPES
)


class AttentionVisualizer:
    """
    Visualizer for dual-stream attention patterns.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        
        # Frequency band information (for 80 mel channels)
        self.freq_bands = {
            'low': list(range(0, 20)),      # 0-20: Low frequencies
            'mid_low': list(range(20, 40)), # 20-40: Lower-mid
            'mid_high': list(range(40, 60)),# 40-60: Upper-mid
            'high': list(range(60, 80))     # 60-80: High frequencies
        }
        
        # Color scheme for frequency bands
        self.freq_colors = {
            'low': '#FF6B6B',      # Red
            'mid_low': '#4ECDC4',  # Teal
            'mid_high': '#45B7D1', # Blue
            'high': '#96CEB4'      # Green
        }
        
        # Blendshape groupings
        self.mouth_blendshapes = [ARKIT_BLENDSHAPES[i] for i in MOUTH_INDICES]
        self.expression_blendshapes = [ARKIT_BLENDSHAPES[i] for i in EXPRESSION_INDICES]
    
    def visualize_mel_attention(
        self,
        attention_weights: torch.Tensor,
        blendshape_names: Optional[List[str]] = None,
        title: str = "Mel-Frequency Attention Pattern",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize mel-frequency attention patterns.
        
        Args:
            attention_weights: Attention weights (num_mouth_blendshapes, 80)
            blendshape_names: Names of blendshapes (default: mouth blendshapes)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if blendshape_names is None:
            blendshape_names = self.mouth_blendshapes
            
        # Convert to numpy
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Heatmap of full attention matrix
        ax1 = axes[0, 0]
        sns.heatmap(
            attention_weights,
            xticklabels=[f'Mel {i}' for i in range(80)],
            yticklabels=blendshape_names,
            cmap='viridis',
            ax=ax1,
            cbar_kws={'label': 'Attention Weight'}
        )
        ax1.set_title('Full Attention Matrix')
        ax1.set_xlabel('Mel Frequency Channels')
        ax1.set_ylabel('Mouth Blendshapes')
        
        # 2. Frequency band aggregation
        ax2 = axes[0, 1]
        band_attention = {}
        for band_name, indices in self.freq_bands.items():
            band_attention[band_name] = attention_weights[:, indices].mean(axis=1)
        
        # Create stacked bar chart
        bottom = np.zeros(len(blendshape_names))
        for band_name, weights in band_attention.items():
            ax2.bar(
                range(len(blendshape_names)), weights, bottom=bottom,
                label=band_name, color=self.freq_colors[band_name], alpha=0.8
            )
            bottom += weights
        
        ax2.set_title('Attention by Frequency Band')
        ax2.set_xlabel('Mouth Blendshapes')
        ax2.set_ylabel('Average Attention Weight')
        ax2.set_xticks(range(len(blendshape_names)))
        ax2.set_xticklabels(blendshape_names, rotation=45, ha='right')
        ax2.legend()
        
        # 3. Top frequency contributors for each blendshape
        ax3 = axes[1, 0]
        top_freq_per_blendshape = []
        for i, blendshape in enumerate(blendshape_names):
            top_indices = np.argsort(attention_weights[i])[-5:]  # Top 5
            top_freq_per_blendshape.append(top_indices)
        
        # Create scatter plot
        for i, (blendshape, top_indices) in enumerate(zip(blendshape_names, top_freq_per_blendshape)):
            ax3.scatter([i] * len(top_indices), top_indices, 
                       s=attention_weights[i, top_indices] * 1000,
                       alpha=0.7, label=blendshape if i < 5 else "")
        
        ax3.set_title('Top Frequency Contributors')
        ax3.set_xlabel('Blendshape Index')
        ax3.set_ylabel('Mel Frequency Channel')
        ax3.set_xticks(range(len(blendshape_names)))
        ax3.set_xticklabels([f'BS{i}' for i in range(len(blendshape_names))])
        
        # 4. Average attention per frequency channel
        ax4 = axes[1, 1]
        avg_attention = attention_weights.mean(axis=0)
        
        # Color by frequency band
        colors = []
        for i in range(80):
            for band_name, indices in self.freq_bands.items():
                if i in indices:
                    colors.append(self.freq_colors[band_name])
                    break
        
        ax4.bar(range(80), avg_attention, color=colors, alpha=0.8)
        ax4.set_title('Average Attention per Frequency Channel')
        ax4.set_xlabel('Mel Frequency Channel')
        ax4.set_ylabel('Average Attention Weight')
        
        # Add frequency band labels
        for band_name, indices in self.freq_bands.items():
            ax4.axvspan(min(indices), max(indices), alpha=0.2, 
                       color=self.freq_colors[band_name], label=band_name)
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_emotion_attention(
        self,
        attention_weights: torch.Tensor,
        emotion_features: Optional[torch.Tensor] = None,
        blendshape_names: Optional[List[str]] = None,
        title: str = "Emotion2Vec Attention Pattern",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize emotion2vec attention patterns.
        
        Args:
            attention_weights: Attention weights (num_expression_blendshapes, T)
            emotion_features: Emotion features for reference (B, T, emotion_dim)
            blendshape_names: Names of blendshapes (default: expression blendshapes)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if blendshape_names is None:
            blendshape_names = self.expression_blendshapes
            
        # Convert to numpy
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Temporal attention heatmap
        ax1 = axes[0, 0]
        sns.heatmap(
            attention_weights,
            yticklabels=blendshape_names,
            xticklabels=[f'T{i}' for i in range(0, attention_weights.shape[1], 
                                              max(1, attention_weights.shape[1]//10))],
            cmap='plasma',
            ax=ax1,
            cbar_kws={'label': 'Attention Weight'}
        )
        ax1.set_title('Temporal Attention Pattern')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Expression Blendshapes')
        
        # 2. Attention intensity over time
        ax2 = axes[0, 1]
        total_attention = attention_weights.sum(axis=0)
        ax2.plot(total_attention, linewidth=2, color='purple', alpha=0.8)
        ax2.fill_between(range(len(total_attention)), total_attention, alpha=0.3, color='purple')
        ax2.set_title('Total Attention Intensity Over Time')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Total Attention Weight')
        ax2.grid(True, alpha=0.3)
        
        # 3. Peak attention moments
        ax3 = axes[1, 0]
        peak_indices = []
        peak_values = []
        
        for i, blendshape in enumerate(blendshape_names):
            peak_idx = np.argmax(attention_weights[i])
            peak_val = attention_weights[i, peak_idx]
            peak_indices.append(peak_idx)
            peak_values.append(peak_val)
        
        scatter = ax3.scatter(peak_indices, range(len(blendshape_names)), 
                             s=np.array(peak_values) * 1000, 
                             c=peak_values, cmap='plasma', alpha=0.7)
        ax3.set_title('Peak Attention Moments')
        ax3.set_xlabel('Time Step of Peak Attention')
        ax3.set_ylabel('Expression Blendshapes')
        ax3.set_yticks(range(len(blendshape_names)))
        ax3.set_yticklabels([name.replace('Left', 'L').replace('Right', 'R') 
                            for name in blendshape_names], fontsize=8)
        plt.colorbar(scatter, ax=ax3, label='Peak Attention Weight')
        
        # 4. Average attention per blendshape
        ax4 = axes[1, 1]
        avg_attention = attention_weights.mean(axis=1)
        bars = ax4.barh(range(len(blendshape_names)), avg_attention, 
                       color='purple', alpha=0.7)
        ax4.set_title('Average Attention per Blendshape')
        ax4.set_xlabel('Average Attention Weight')
        ax4.set_ylabel('Expression Blendshapes')
        ax4.set_yticks(range(len(blendshape_names)))
        ax4.set_yticklabels([name.replace('Left', 'L').replace('Right', 'R') 
                            for name in blendshape_names], fontsize=8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, avg_attention)):
            ax4.text(val + max(avg_attention) * 0.01, i, f'{val:.3f}', 
                    va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_dual_stream_comparison(
        self,
        mel_attention: torch.Tensor,
        emotion_attention: torch.Tensor,
        blendshapes_pred: torch.Tensor,
        title: str = "Dual-Stream Attention Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare mel and emotion attention patterns side by side.
        
        Args:
            mel_attention: Mel attention weights (num_mouth_blendshapes, 80)
            emotion_attention: Emotion attention weights (num_expression_blendshapes, T)
            blendshapes_pred: Predicted blendshapes (52,)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Convert to numpy
        if torch.is_tensor(mel_attention):
            mel_attention = mel_attention.detach().cpu().numpy()
        if torch.is_tensor(emotion_attention):
            emotion_attention = emotion_attention.detach().cpu().numpy()
        if torch.is_tensor(blendshapes_pred):
            blendshapes_pred = blendshapes_pred.detach().cpu().numpy()
        
        # Top row: Mel stream analysis
        # 1. Mel attention heatmap
        ax1 = axes[0, 0]
        sns.heatmap(
            mel_attention,
            yticklabels=self.mouth_blendshapes,
            xticklabels=False,
            cmap='viridis',
            ax=ax1,
            cbar_kws={'label': 'Attention'}
        )
        ax1.set_title('Mel → Mouth Blendshapes')
        ax1.set_xlabel('Mel Frequency Channels (0-80)')
        
        # 2. Mouth blendshape predictions
        ax2 = axes[0, 1]
        mouth_pred = blendshapes_pred[MOUTH_INDICES]
        bars = ax2.bar(range(len(self.mouth_blendshapes)), mouth_pred, 
                      color='skyblue', alpha=0.8)
        ax2.set_title('Predicted Mouth Blendshapes')
        ax2.set_xlabel('Mouth Blendshapes')
        ax2.set_ylabel('Prediction Value')
        ax2.set_xticks(range(len(self.mouth_blendshapes)))
        ax2.set_xticklabels([name.replace('mouth', 'm').replace('jaw', 'j') 
                            for name in self.mouth_blendshapes], 
                           rotation=45, ha='right', fontsize=8)
        
        # 3. Frequency band contribution to mouth
        ax3 = axes[0, 2]
        band_contrib = {}
        for band_name, indices in self.freq_bands.items():
            band_contrib[band_name] = mel_attention[:, indices].mean()
        
        colors = [self.freq_colors[band] for band in band_contrib.keys()]
        ax3.pie(band_contrib.values(), labels=band_contrib.keys(), 
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Frequency Band Contribution')
        
        # Bottom row: Emotion stream analysis
        # 4. Emotion attention heatmap
        ax4 = axes[1, 0]
        # Downsample emotion attention for visualization if too large
        if emotion_attention.shape[1] > 100:
            step = emotion_attention.shape[1] // 50
            emotion_attention_viz = emotion_attention[:, ::step]
        else:
            emotion_attention_viz = emotion_attention
            
        sns.heatmap(
            emotion_attention_viz,
            yticklabels=[name.replace('Left', 'L').replace('Right', 'R') 
                        for name in self.expression_blendshapes],
            xticklabels=False,
            cmap='plasma',
            ax=ax4,
            cbar_kws={'label': 'Attention'}
        )
        ax4.set_title('Emotion → Expression Blendshapes')
        ax4.set_xlabel('Time Steps')
        
        # 5. Expression blendshape predictions
        ax5 = axes[1, 1]
        expression_pred = blendshapes_pred[EXPRESSION_INDICES]
        bars = ax5.bar(range(len(self.expression_blendshapes)), expression_pred, 
                      color='lightcoral', alpha=0.8)
        ax5.set_title('Predicted Expression Blendshapes')
        ax5.set_xlabel('Expression Blendshapes')
        ax5.set_ylabel('Prediction Value')
        ax5.set_xticks(range(len(self.expression_blendshapes)))
        ax5.set_xticklabels([name.replace('Left', 'L').replace('Right', 'R') 
                            for name in self.expression_blendshapes], 
                           rotation=45, ha='right', fontsize=6)
        
        # 6. Stream specialization analysis
        ax6 = axes[1, 2]
        mouth_avg = np.mean(mouth_pred)
        expression_avg = np.mean(expression_pred)
        
        categories = ['Mouth\n(Mel Stream)', 'Expression\n(Emotion Stream)']
        values = [mouth_avg, expression_avg]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax6.bar(categories, values, color=colors, alpha=0.8)
        ax6.set_title('Stream Specialization')
        ax6.set_ylabel('Average Prediction Value')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_interactive_attention_plot(
        self,
        mel_attention: torch.Tensor,
        emotion_attention: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive plotly visualization of attention patterns.
        
        Args:
            mel_attention: Mel attention weights
            emotion_attention: Emotion attention weights
            audio_features: Original audio features for reference
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        # Convert to numpy
        if torch.is_tensor(mel_attention):
            mel_attention = mel_attention.detach().cpu().numpy()
        if torch.is_tensor(emotion_attention):
            emotion_attention = emotion_attention.detach().cpu().numpy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mel-Frequency Attention', 'Frequency Band Analysis',
                          'Emotion Temporal Attention', 'Attention Summary'),
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # 1. Mel attention heatmap
        fig.add_trace(
            go.Heatmap(
                z=mel_attention,
                x=[f'Mel{i}' for i in range(80)],
                y=self.mouth_blendshapes,
                colorscale='Viridis',
                name='Mel Attention'
            ),
            row=1, col=1
        )
        
        # 2. Frequency band analysis
        band_attention = {}
        for band_name, indices in self.freq_bands.items():
            band_attention[band_name] = mel_attention[:, indices].mean()
        
        fig.add_trace(
            go.Bar(
                x=list(band_attention.keys()),
                y=list(band_attention.values()),
                marker_color=[self.freq_colors[band] for band in band_attention.keys()],
                name='Band Attention'
            ),
            row=1, col=2
        )
        
        # 3. Emotion attention heatmap (downsampled for interactivity)
        if emotion_attention.shape[1] > 100:
            step = emotion_attention.shape[1] // 50
            emotion_viz = emotion_attention[:, ::step]
            time_labels = [f'T{i}' for i in range(0, emotion_attention.shape[1], step)]
        else:
            emotion_viz = emotion_attention
            time_labels = [f'T{i}' for i in range(emotion_attention.shape[1])]
        
        fig.add_trace(
            go.Heatmap(
                z=emotion_viz,
                x=time_labels,
                y=[name[:15] + '...' if len(name) > 15 else name 
                   for name in self.expression_blendshapes],
                colorscale='Plasma',
                name='Emotion Attention'
            ),
            row=2, col=1
        )
        
        # 4. Attention summary scatter
        mel_max = mel_attention.max(axis=1)
        emotion_max = emotion_attention.max(axis=1)
        
        fig.add_trace(
            go.Scatter(
                x=mel_max,
                y=list(range(len(self.mouth_blendshapes))),
                mode='markers',
                marker=dict(size=10, color='blue'),
                name='Mel Max Attention',
                text=self.mouth_blendshapes,
                hovertemplate='<b>%{text}</b><br>Max Attention: %{x:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=emotion_max,
                y=list(range(len(self.expression_blendshapes))),
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Emotion Max Attention',
                text=self.expression_blendshapes,
                hovertemplate='<b>%{text}</b><br>Max Attention: %{x:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Interactive Dual-Stream Attention Analysis',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig


def visualize_training_attention(
    model,
    dataloader,
    device: torch.device,
    num_samples: int = 5,
    save_dir: str = "attention_viz",
) -> None:
    """
    Visualize attention patterns during training.
    
    Args:
        model: Trained dual-stream model
        dataloader: Data loader
        device: Computation device
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    visualizer = AttentionVisualizer()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            audio = batch["audio"].to(device)
            
            # Get model output with attention
            output = model(audio, return_attention=True)
            
            # Visualize for first sample in batch
            if 'mel_attention_weights' in output and 'emotion_attention_weights' in output:
                mel_attn = output['mel_attention_weights'][0]  # First sample
                emotion_attn = output['emotion_attention_weights'][0]
                blendshapes = output['blendshapes'][0]
                
                # Create visualizations
                mel_fig = visualizer.visualize_mel_attention(
                    mel_attn,
                    save_path=f"{save_dir}/mel_attention_sample_{batch_idx}.png"
                )
                plt.close(mel_fig)
                
                emotion_fig = visualizer.visualize_emotion_attention(
                    emotion_attn,
                    save_path=f"{save_dir}/emotion_attention_sample_{batch_idx}.png"
                )
                plt.close(emotion_fig)
                
                comparison_fig = visualizer.visualize_dual_stream_comparison(
                    mel_attn, emotion_attn, blendshapes,
                    save_path=f"{save_dir}/comparison_sample_{batch_idx}.png"
                )
                plt.close(comparison_fig)
                
                # Create interactive plot
                interactive_fig = visualizer.create_interactive_attention_plot(
                    mel_attn, emotion_attn,
                    save_path=f"{save_dir}/interactive_sample_{batch_idx}.html"
                )
            
    print(f"Attention visualizations saved to {save_dir}/")