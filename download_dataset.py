"""
Sequential dataset downloader for the Fake-or-Real (FoR) dataset.
Downloads for-norm (LA norm) first, then all other subsets automatically.
"""

import os
import sys
import subprocess
import zipfile
import time

# ─── Kaggle credentials ────────────────────────────────────────────────────────
os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY']      = 'cab3af2761e387b397589ec38fd6ad8b'
DATASET = 'mohammedabdeldayem/the-fake-or-real-dataset'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def install_kaggle():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return True
    except ImportError:
        print("Installing kaggle...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle', '-q'])
        return True

def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def download_and_extract(api, zip_name, label):
    zip_path = os.path.join(DATA_DIR, zip_name)
    print(f"\n{'='*60}")
    print(f"[{label}] Starting download...")
    print(f"{'='*60}")
    t0 = time.time()
    
    try:
        # Download full zip
        api.dataset_download_files(
            DATASET,
            path=DATA_DIR,
            unzip=False,
            quiet=False
        )
        elapsed = time.time() - t0
        print(f"[{label}] Download finished in {elapsed/60:.1f} minutes.")
    except Exception as e:
        print(f"[{label}] Download error: {e}")
        return False
    
    # Find the zip file
    zip_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.zip')]
    if not zip_files:
        print(f"[{label}] No zip file found after download.")
        return False
    
    zip_path = os.path.join(DATA_DIR, zip_files[0])
    print(f"\n[{label}] Extracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
            # Extract for-norm first (LA norm - required by PS)
            norm_members = [m for m in members if 'for-norm' in m]
            print(f"[{label}] Extracting for-norm ({len(norm_members)} files)...")
            for member in norm_members:
                z.extract(member, DATA_DIR)
            
            print(f"\n✅ for-norm (LA norm) extracted successfully!")
            print(f"   Path: {os.path.join(DATA_DIR, 'for-norm')}")
            print(f"\n[AUTO] Now extracting remaining dataset subsets...")
            
            # Auto-extract everything else without asking
            other_members = [m for m in members if 'for-norm' not in m]
            print(f"[AUTO] Extracting {len(other_members)} remaining files...")
            for i, member in enumerate(other_members):
                if i % 1000 == 0 and i > 0:
                    print(f"[AUTO] Progress: {i}/{len(other_members)} files extracted...")
                z.extract(member, DATA_DIR)
            
            print(f"\n✅ Full dataset extraction complete!")
            
    except Exception as e:
        print(f"Extraction error: {e}")
        return False
    
    # Clean up zip to free disk space
    print(f"\nCleaning up zip file to free disk space...")
    os.remove(zip_path)
    print(f"✅ Done. Dataset ready at: {DATA_DIR}")
    
    # Print dataset structure
    print(f"\nDataset structure:")
    for item in sorted(os.listdir(DATA_DIR)):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            subfolders = os.listdir(item_path)
            print(f"  📁 {item}/  ({len(subfolders)} items)")
    
    return True

def main():
    print("🎙️  Deepfake Audio Detection — Dataset Downloader")
    print("   Dataset: Fake-or-Real (FoR) Dataset")
    print("   Strategy: Download full zip → extract for-norm first → then rest\n")
    
    install_kaggle()
    api = get_api()
    
    success = download_and_extract(api, 'the-fake-or-real-dataset.zip', 'FoR Dataset')
    
    if success:
        print("\n" + "="*60)
        print("🎉 ALL DONE! Dataset fully ready.")
        print("   Next step: Run feature extraction on for-norm/training/")
        print("="*60)
    else:
        print("\n❌ Download failed. Please check your internet connection.")
        sys.exit(1)

if __name__ == '__main__':
    main()
