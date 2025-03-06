from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
import argparse

# Global variables to store the model and device
model_fair_4 = None
device = None
race_map_4 = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian'
}

def load_model(models_path='fairface/models/'):
    """
    Load the race prediction model
    
    Parameters:
    -----------
    models_path : str
        Path to the directory containing the FairFace models
        
    Returns:
    --------
    None
    """
    global model_fair_4, device
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load 4-race classifier model if not already loaded
    if model_fair_4 is None:
        print("Loading race classification model...")
        model_fair_4 = torchvision.models.resnet34(pretrained=True)
        model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
        model_fair_4.load_state_dict(torch.load(os.path.join(models_path, 'res34_fair_align_multi_4_20190809.pt')))
        model_fair_4 = model_fair_4.to(device)
        model_fair_4.eval()
        print("Model loaded successfully!")

def predict_race(img_path, models_path='fairface/models/'):
    """
    Predict race from a face image using the 4-race classifier
    
    Parameters:
    -----------
    img_path : str
        Path to the face image
    models_path : str
        Path to the directory containing the FairFace models
        
    Returns:
    --------
    dict
        Dictionary containing predicted race label and confidence scores
    """
    global model_fair_4, device
    
    # Load model if not already loaded
    if model_fair_4 is None:
        load_model(models_path)
    
    # Image transformation
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process the image using PIL
    try:
        image = Image.open(img_path).convert('RGB')
        image = trans(image)
        image = image.unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
        image = image.to(device)
    except Exception as e:
        raise Exception(f"Error loading image {img_path}: {str(e)}")
    
    # Predict with 4-race model
    with torch.no_grad():  # No need to track gradients for inference
        outputs = model_fair_4(image)
        outputs = outputs.cpu().numpy()
        outputs = np.squeeze(outputs)
    
    race_outputs = outputs[:4]
    race_scores = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    race_pred = np.argmax(race_scores)
    
    race_label = race_map_4[race_pred]
    
    # Create result dictionary
    result = {
        'race': race_label,
        'race_score': float(race_scores[race_pred]),
        'race_probabilities': {race_map_4[i]: float(race_scores[i]) for i in range(4)}
    }
    
    return result

def predict_race_batch(img_paths, output_csv=None, models_path='fairface/models/'):
    """
    Predict race for a batch of face images using the 4-race classifier
    
    Parameters:
    -----------
    img_paths : list
        List of paths to face images
    output_csv : str, optional
        Path to save results as CSV
    models_path : str
        Path to the directory containing the FairFace models
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing race predictions and scores for all images
    """
    # Make sure model is loaded before batch processing
    if model_fair_4 is None:
        load_model(models_path)
    
    results = []
    total_images = len(img_paths)
    
    for i, img_path in enumerate(img_paths):
        if i % 100 == 0:
            print(f"Processing {i}/{total_images} ({i/total_images*100:.1f}%)")
        
        try:
            result = predict_race(img_path)  # No need to pass models_path again
            result['face_name'] = img_path
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns to put face_name first
    if not df.empty:
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('face_name')))
        df = df[cols]
    
        # Save to CSV if output path is provided
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
    else:
        print("No valid results to save.")
    
    return df

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict race from face images using 4-race classifier')
    parser.add_argument('--image', dest='image_path', help='Path to a single face image')
    parser.add_argument('--csv', dest='input_csv', help='CSV file with column "img_path" containing paths to face images')
    parser.add_argument('--dir', dest='image_dir', help='Directory containing face images')
    parser.add_argument('--output', dest='output_csv', default='race_predictions.csv', help='Output CSV file path')
    parser.add_argument('--models', dest='models_path', default='fairface/models/', help='Path to FairFace models')
    
    args = parser.parse_args()
    
    # Load model at startup
    load_model(args.models_path)
    
    if args.image_path:
        # Single image prediction
        result = predict_race(args.image_path)
        print("\nRace Prediction Results:")
        print(f"Race: {result['race']} (confidence: {result['race_score']*100:.2f}%)")
        print("\nProbabilities:")
        for race, prob in result['race_probabilities'].items():
            print(f"  {race}: {prob*100:.2f}%")
    
    elif args.input_csv:
        # Process images from CSV
        img_paths = pd.read_csv(args.input_csv)['img_path'].tolist()
        predict_race_batch(img_paths, args.output_csv)
    
    elif args.image_dir:
        # Process all images in directory
        valid_extensions = ['.jpg', '.jpeg', '.png']
        img_paths = [os.path.join(args.image_dir, f) 
                    for f in os.listdir(args.image_dir) 
                    if os.path.splitext(f.lower())[1] in valid_extensions]
        predict_race_batch(img_paths, args.output_csv)
    
    else:
        parser.print_help()
