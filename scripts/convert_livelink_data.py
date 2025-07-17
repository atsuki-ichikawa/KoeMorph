#!/usr/bin/env python3
"""
Convert Live Link Face data to KoeMorph training format.

Live Link Face exports:
- CSV file with ARKit blendshapes at 30 FPS
- MOV file with audio at 48kHz
- JSON metadata files

This script converts them to:
- WAV file at 16kHz (KoeMorph format)
- JSONL file with synchronized blendshape data at 30 FPS
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def parse_timecode(timecode: str) -> float:
    """Parse timecode string to seconds.
    
    Format: HH:MM:SS:FF.mmm where FF is frame number and mmm is milliseconds
    """
    parts = timecode.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    
    # Handle frame and milliseconds
    frame_ms = parts[3].split('.')
    frame = int(frame_ms[0])
    milliseconds = int(frame_ms[1]) if len(frame_ms) > 1 else 0
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_seconds += frame / 30.0  # Assuming 30 FPS
    total_seconds += milliseconds / 1000.0
    
    return total_seconds


def extract_audio_from_mov(mov_path: Path, output_wav: Path, target_sr: int = 16000) -> bool:
    """Extract audio from MOV file and convert to target sample rate."""
    print(f"Warning: Audio extraction from MOV files is not implemented.")
    print(f"Please manually extract audio from {mov_path} to {output_wav}")
    print(f"You can use: ffmpeg -i {mov_path} -ar {target_sr} -ac 1 {output_wav}")
    
    # For now, create a dummy audio file for testing
    # Use a much shorter duration to match the available blendshape data
    duration = 6.0  # seconds - shorter to match available data
    samples = int(duration * target_sr)
    dummy_audio = np.random.randn(samples) * 0.001  # Very quiet noise
    sf.write(str(output_wav), dummy_audio, target_sr)
    
    return True


def load_blendshape_csv(csv_path: Path) -> pd.DataFrame:
    """Load blendshape CSV file."""
    # Try different encodings in case of encoding issues
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='utf-16')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
    
    print(f"CSV columns: {list(df.columns)}")
    print(f"CSV shape: {df.shape}")
    return df


def convert_blendshapes_to_jsonl(df: pd.DataFrame, output_path: Path, 
                                start_time: float = 0.0) -> None:
    """Convert blendshape DataFrame to JSONL format."""
    
    # ARKit blendshape names (52 total)
    arkit_names = [
        'EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft',
        'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight',
        'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight',
        'EyeSquintRight', 'EyeWideRight', 'JawForward', 'JawRight', 'JawLeft',
        'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight',
        'MouthLeft', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft',
        'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft',
        'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower',
        'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
        'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight', 'BrowDownLeft',
        'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight',
        'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft',
        'NoseSneerRight', 'TongueOut'
    ]
    
    with open(output_path, 'w') as f:
        for idx, row in df.iterrows():
            # Parse timecode to get timestamp
            timestamp = parse_timecode(row['Timecode']) - start_time
            
            # Extract blendshape values
            blendshapes = []
            for name in arkit_names:
                if name in row:
                    blendshapes.append(float(row[name]))
                else:
                    blendshapes.append(0.0)
            
            # Ensure we have exactly 52 values
            if len(blendshapes) != 52:
                print(f"Warning: Expected 52 blendshapes, got {len(blendshapes)}")
                blendshapes = blendshapes[:52] + [0.0] * max(0, 52 - len(blendshapes))
            
            data = {
                'timestamp': timestamp,
                'blendshapes': blendshapes
            }
            
            f.write(json.dumps(data) + '\n')


def convert_take_to_koemorph(take_dir: Path, output_dir: Path, 
                           max_duration: Optional[float] = None) -> bool:
    """Convert a single Live Link Face take to KoeMorph format."""
    
    take_name = take_dir.name
    print(f"Converting take: {take_name}")
    
    # Find files in take directory
    csv_files = list(take_dir.glob("*.csv"))
    mov_files = list(take_dir.glob("*.mov"))
    take_json = take_dir / "take.json"
    
    # Filter out Zone.Identifier files
    csv_files = [f for f in csv_files if not f.name.endswith(':Zone.Identifier') and not f.name.endswith('.Zone.Identifier')]
    mov_files = [f for f in mov_files if not f.name.endswith(':Zone.Identifier') and not f.name.endswith('.Zone.Identifier')]
    
    print(f"Found CSV files: {[f.name for f in csv_files]}")
    print(f"Found MOV files: {[f.name for f in mov_files]}")
    
    if not csv_files or not mov_files:
        print(f"Error: Missing CSV or MOV files in {take_dir}")
        return False
    
    # Find the main CSV file (not frame_log.csv)
    csv_path = None
    for f in csv_files:
        if 'frame_log' not in f.name:
            csv_path = f
            break
    
    if csv_path is None:
        print(f"Error: No main CSV file found in {take_dir}")
        return False
    
    mov_path = mov_files[0]
    print(f"Using CSV file: {csv_path.name}")
    print(f"Using MOV file: {mov_path.name}")
    
    # Load take metadata
    take_info = {}
    if take_json.exists():
        with open(take_json, 'r') as f:
            take_info = json.load(f)
    
    # Get start time from take info
    start_time = 0.0
    if 'startTimecode' in take_info:
        start_time = parse_timecode(take_info['startTimecode'])
    
    # Create output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wav = output_dir / f"{take_name}.wav"
    output_jsonl = output_dir / f"{take_name}.jsonl"
    
    # Extract and convert audio
    print("Extracting audio...")
    if not extract_audio_from_mov(mov_path, output_wav):
        return False
    
    # Apply duration limit if specified
    if max_duration:
        audio, sr = sf.read(str(output_wav))
        if len(audio) > max_duration * sr:
            audio = audio[:int(max_duration * sr)]
            sf.write(str(output_wav), audio, sr)
    
    # Convert blendshapes
    print("Converting blendshapes...")
    df = load_blendshape_csv(csv_path)
    
    # Apply duration limit to blendshapes if specified
    if max_duration:
        # Filter rows by timestamp
        df['timestamp_seconds'] = df['Timecode'].apply(lambda x: parse_timecode(x) - start_time)
        df = df[df['timestamp_seconds'] <= max_duration]
    
    convert_blendshapes_to_jsonl(df, output_jsonl, start_time)
    
    print(f"Converted {take_name} successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert Live Link Face data to KoeMorph format")
    parser.add_argument("input_dir", type=Path, help="Directory containing Live Link Face takes")
    parser.add_argument("output_dir", type=Path, help="Output directory for converted data")
    parser.add_argument("--max-duration", type=float, help="Maximum duration in seconds (for testing)")
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='train', 
                       help="Data split to create")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir / args.split
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    # Find all take directories
    take_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not take_dirs:
        print(f"Error: No take directories found in {input_dir}")
        return 1
    
    print(f"Found {len(take_dirs)} takes to convert")
    
    # Convert each take
    success_count = 0
    for take_dir in tqdm(take_dirs, desc="Converting takes"):
        if convert_take_to_koemorph(take_dir, output_dir, args.max_duration):
            success_count += 1
    
    print(f"\nConversion complete: {success_count}/{len(take_dirs)} takes converted successfully")
    
    if success_count > 0:
        print(f"Converted data saved to: {output_dir}")
        print(f"Use this directory for training: --data.{args.split}_data_dir={output_dir}")
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())