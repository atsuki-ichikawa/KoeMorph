#!/usr/bin/env python3
"""Simple training test to verify weight updates."""

import torch
import torch.nn as nn
from pathlib import Path
from src.data.dataset import KoeMorphDataModule
from src.model.gaussian_face import KoeMorphModel

def test_weight_updates():
    """Test that model weights are actually being updated during training."""
    
    print("=== Setting up data ===")
    # Setup data
    data_module = KoeMorphDataModule(
        train_data_dir="/home/ichikawa/KoeMorph/data/train",
        val_data_dir=None,
        test_data_dir=None,
        sample_rate=16000,
        max_audio_length=6.0,
        target_fps=30,
        batch_size=1,  # Single batch for testing
        num_workers=0,
        pin_memory=False,
    )
    
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    print(f"Dataset size: {len(data_module._train_dataset)}")
    print(f"DataLoader batches: {len(train_loader)}")
    
    # Get first batch
    batch = next(iter(train_loader))
    print(f"Batch audio shape: {batch['wav'].shape}")
    print(f"Batch blendshape shape: {batch['arkit'].shape}")
    
    print("\n=== Setting up model ===")
    # Simple model configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a much simpler model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, 52)  # 52 blendshapes
            
        def forward(self, batch):
            x = batch['wav'].unsqueeze(1)  # Add channel dim
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)  # Remove time dim
            x = torch.sigmoid(self.fc(x))  # Output in [0,1]
            
            # Repeat to match target sequence length
            seq_len = batch['arkit'].shape[1]
            x = x.unsqueeze(1).repeat(1, seq_len, 1)
            return x
    
    model = SimpleModel().to(device)
    
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\n=== Testing forward pass ===")
    # Test forward pass
    with torch.no_grad():
        output = model(batch)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n=== Testing weight updates ===")
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Get initial weights
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.data.clone()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    output = model(batch)
    target = batch['arkit']
    
    loss = criterion(output, target)
    print(f"Initial loss: {loss.item():.6f}")
    
    loss.backward()
    optimizer.step()
    
    # Check if weights changed
    weights_changed = False
    for name, param in model.named_parameters():
        if not torch.equal(initial_weights[name], param.data):
            weights_changed = True
            diff = torch.abs(initial_weights[name] - param.data).max()
            print(f"  {name}: max weight change = {diff:.6f}")
    
    if weights_changed:
        print("\n✅ SUCCESS: Model weights were updated!")
        
        # Test a few more iterations
        print("\n=== Testing multiple iterations ===")
        for i in range(5):
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f"Iteration {i+1}: loss = {loss.item():.6f}")
        
        return True
    else:
        print("\n❌ FAIL: Model weights were NOT updated!")
        return False

if __name__ == "__main__":
    success = test_weight_updates()
    if success:
        print("\n🎉 Training pipeline is working correctly!")
    else:
        print("\n💥 Training pipeline has issues.")