import os
import glob
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from spafe.features.lfcc import lfcc

warnings.filterwarnings('ignore')

def extract_audio_features(file_path):
    try:
        # 1. Fast loading using soundfile (bypasses Numba JIT overhead)
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1) # downmix to mono
            
        # Standardize sample rate for spafe/librosa consistency
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # Ensure minimum length for STFT (e.g., 2048 samples)
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)))
            
        features = []
        
        # 2. MFCCs (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
        # 3. LFCCs (Linear-frequency cepstral coefficients)
        # Highly effective for capturing synthetic artifacts
        lfccs = lfcc(y, fs=sr, num_ceps=20)
        features.extend(np.mean(lfccs, axis=0))
        features.extend(np.std(lfccs, axis=0))
        
        # 4. Spectral & Energy Features
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        
        for feat in [cent, bw, rolloff, zcr, rms]:
            features.append(np.mean(feat))
            features.append(np.std(feat))
            
        return features
        
    except Exception as e:
        # Return None if the file is corrupt or unreadable
        return None

def main():
    base_dir = '/Users/hardik/Desktop/Mars/Deepfake_Audio/data/for-norm/for-norm'
    
    real_files = glob.glob(os.path.join(base_dir, '*/real/*.wav'))
    fake_files = glob.glob(os.path.join(base_dir, '*/fake/*.wav'))
    
    print(f"Found {len(real_files)} genuine and {len(fake_files)} deepfake files across all splits.")
    print("We will process the entire folder. This takes ~15-20 minutes.")
    
    data = []
    
    print("Extracting features from GENUINE audio...")
    for f in tqdm(real_files, desc="Real"):
        feats = extract_audio_features(f)
        if feats is not None:
            data.append(feats + [0]) # 0 = Genuine
            
    print("Extracting features from DEEPFAKE audio...")
    for f in tqdm(fake_files, desc="Fake"):
        feats = extract_audio_features(f)
        if feats is not None:
            data.append(feats + [1]) # 1 = Deepfake
            
    # Generate column names (90 dimensions total)
    cols = []
    for i in range(1, 21): cols.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
    for i in range(1, 21): cols.extend([f'lfcc_{i}_mean', f'lfcc_{i}_std'])
    for name in ['cent', 'bw', 'rolloff', 'zcr', 'rms']:
        cols.extend([f'{name}_mean', f'{name}_std'])
    cols.append('label')
    
    df = pd.DataFrame(data, columns=cols)
    out_path = '/Users/hardik/Desktop/Mars/Deepfake_Audio/data/extracted_features.csv'
    df.to_csv(out_path, index=False)
    print(f"\nFeature extraction complete! Saved to {out_path}")
    print(f"Total shape: {df.shape}")

if __name__ == '__main__':
    main()
