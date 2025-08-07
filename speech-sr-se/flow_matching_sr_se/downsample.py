import os
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import argparse

def downsample_audio_file(input_path, output_path, target_sr):
    audio, original_sr = sf.read(input_path)
    if original_sr == target_sr:
        print(f"Skipping {input_path}: already at {target_sr} Hz")
        sf.write(output_path, audio, target_sr)
        return

    factor = gcd(original_sr, target_sr)
    up = target_sr // factor
    down = original_sr // factor

    audio_downsampled = resample_poly(audio, up, down, axis=0)
    sf.write(output_path, audio_downsampled, target_sr)
    print(f"Processed: {input_path} -> {output_path}")

def batch_downsample(input_dir, output_dir, target_sr):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.wav', '.flac', '.aiff', '.aif', '.aifc')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            downsample_audio_file(input_path, output_path, target_sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample all audio files in a directory using scipy.resample_poly.")
    parser.add_argument("input_dir", help="Directory containing input audio files")
    parser.add_argument("output_dir", help="Directory to save downsampled audio files")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sampling rate (e.g., 16000)")

    args = parser.parse_args()
    batch_downsample(args.input_dir, args.output_dir, args.target_sr)
