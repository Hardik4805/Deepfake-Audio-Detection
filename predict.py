import argparse
import joblib
import os
import sys

# Import our feature extractor
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from extract_features import extract_audio_features

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection Inference Script")
    parser.add_argument('--audio', type=str, required=True, help="Path to the audio file (.wav or .mp3)")
    parser.add_argument('--model', type=str, default='models/lgbm_model.pkl', help="Path to the trained LightGBM model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' not found.")
        return
        
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
        
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)
    
    print(f"Extracting acoustic features from {args.audio}...")
    features = extract_audio_features(args.audio)
    
    if features is None:
        print("Error extracting features. Audio may be corrupted.")
        return
        
    # Model expects 2D array
    import numpy as np
    features_2d = np.array(features).reshape(1, -1)
    
    prob = model.predict_proba(features_2d)[0]
    prediction = model.predict(features_2d)[0]
    
    print("\n" + "="*40)
    print("        INFERENCE RESULTS")
    print("="*40)
    if prediction == 1:
        print(f"Prediction: DEEPFAKE (AI-Generated)")
        print(f"Confidence: {prob[1]*100:.2f}%")
    else:
        print(f"Prediction: GENUINE (Human Speech)")
        print(f"Confidence: {prob[0]*100:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()
