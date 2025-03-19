import os
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.io import read_image

from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from PIL import Image
import numpy as np

# Define the attribute maps
SKIN_TONE_MAP = {
    "black": 0,
    "white": 1,
    "brown": 2,
    "other": 3
}

EYE_COLOR_MAP = {
    "black": 0,
    "blue": 1,
    "green": 2,
    "brown": 3,
    "other": 4,
    "unknown": 5
}

QUALITY_SCORE_MAP = {
    "bad": 0.1,
    "average": 0.5,
    "good": 1.0,
    "not_rate": 0.0
}

GENDER_MAP = {
    "male": 0,
    "female": 1,
    "unknown": 2
}

FACIAL_HAIR_MAP = {
    "yes": 0,
    "no": 1,
    "partial": 2
}

HAS_GLASSES_MAP = {
    "yes": 0,
    "no": 1
}

HAS_NECKLACE_MAP = {
    "yes": 0,
    "no": 1
}

LIP_COLOR_MAP = {
    "red": 0,
    "pink": 1,
    "light pink": 2,
    "dark": 3,
    "darker": 4,
    "other": 5,
    "unknown": 6
}

# Define output sizes for each attribute
ATTRIBUTE_SIZES = {
    "skin_tone": len(SKIN_TONE_MAP),
    "eye_color": len(EYE_COLOR_MAP),
    "quality_score": 1,  # Regression task
    "gender": len(GENDER_MAP),
    "facial_hair": len(FACIAL_HAIR_MAP),
    "has_glasses": len(HAS_GLASSES_MAP),
    "has_necklace": len(HAS_NECKLACE_MAP),
    "lip_color": len(LIP_COLOR_MAP)
}

class FaceAttributeDataset(Dataset):
    def __init__(self, data_json, transform=None):
        """
        Args:
            data_json (str): Path to the JSON file with annotations
            transform (callable, optional): Optional transform to apply to the image
        """
        with open(data_json, 'r') as f:
            data = json.load(f)
        
        self.annotations = []
        for img_path, annotation in data['annotations'].items():
            if annotation.get('is_completed', False):
                self.annotations.append(annotation)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = annotation['image_path']
        
        try:
            # Handle both regular image files and webp
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            image = torch.zeros((3, 224, 224))
            
        # Create labels dictionary
        labels = {
            "skin_tone": SKIN_TONE_MAP.get(annotation.get('skin_tone', 'other'), SKIN_TONE_MAP['other']),
            "eye_color": EYE_COLOR_MAP.get(annotation.get('eye_color', 'unknown'), EYE_COLOR_MAP['unknown']),
            "quality_score": QUALITY_SCORE_MAP.get(annotation.get('quality_score', 'not_rate'), QUALITY_SCORE_MAP['not_rate']),
            "gender": GENDER_MAP.get(annotation.get('gender', 'unknown'), GENDER_MAP['unknown']),
            "facial_hair": FACIAL_HAIR_MAP.get(annotation.get('facial_hair', 'no'), FACIAL_HAIR_MAP['no']),
            "has_glasses": HAS_GLASSES_MAP.get(annotation.get('has_glasses', 'no'), HAS_GLASSES_MAP['no']),
            "has_necklace": HAS_NECKLACE_MAP.get(annotation.get('has_necklace', 'no'), HAS_NECKLACE_MAP['no']),
            "lip_color": LIP_COLOR_MAP.get(annotation.get('lip_color', 'unknown'), LIP_COLOR_MAP['unknown'])
        }
        
        # Convert quality_score to float for regression
        labels["quality_score"] = float(labels["quality_score"])
        
        return image, labels

class AttributeClassifier(nn.Module):
    def __init__(self, attribute_sizes):
        super(AttributeClassifier, self).__init__()
        self.attribute_sizes = attribute_sizes
        
        # Load the ConvNeXt V2 model from Hugging Face (using the classification version)
        self.backbone = ConvNextV2ForImageClassification.from_pretrained(
            "facebook/convnextv2-tiny-22k-224",
            ignore_mismatched_sizes=True,
            num_labels=1  # Temporarily set to 1 - we'll replace the classifier
        )
        
        # Remove the final classification layer
        self.feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Create classifier heads for each attribute
        self.classifiers = nn.ModuleDict({
            attr: nn.Linear(self.feature_dim, output_size if attr != "quality_score" else 1)
            for attr, output_size in attribute_sizes.items()
        })
        
    def forward(self, x):
        # Extract features from the backbone (excluding the classification head)
        features = self.backbone(x).logits
        
        # Apply each attribute classifier
        outputs = {}
        for attr, classifier in self.classifiers.items():
            outputs[attr] = classifier(features)
            
        return outputs

def calculate_loss(outputs, labels, device):
    """Calculate the combined loss for all attributes"""
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    total_loss = 0
    losses = {}
    
    for attr, pred in outputs.items():
        if attr == "quality_score":
            # Regression loss for quality score
            target = torch.tensor([labels[attr]]).to(device).float().view(-1, 1)
            loss = criterion_reg(pred, target)
        else:
            # Classification loss for other attributes
            target = torch.tensor([labels[attr]]).to(device).long()
            loss = criterion_cls(pred, target)
        
        total_loss += loss
        losses[attr] = loss.item()
    
    losses["total"] = total_loss.item()
    return total_loss, losses

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    epoch_losses = {attr: 0.0 for attr in ATTRIBUTE_SIZES.keys()}
    epoch_losses["total"] = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for i, (images, labels_dict) in enumerate(progress_bar):
        images = images.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        batch_loss = 0
        for j in range(images.size(0)):
            loss, losses = calculate_loss({k: v[j:j+1] for k, v in outputs.items()}, 
                                         {k: labels_dict[k][j] for k in labels_dict.keys()}, 
                                         device)
            batch_loss += loss
            
            # Update running losses
            for attr, loss_val in losses.items():
                epoch_losses[attr] += loss_val / len(dataloader)
        
        # Backward pass and optimization
        batch_loss = batch_loss / images.size(0)
        batch_loss.backward()
        optimizer.step()
        
        running_loss += batch_loss.item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1)})
    
    return epoch_losses

def validate(model, dataloader, device):
    model.eval()
    val_losses = {attr: 0.0 for attr in ATTRIBUTE_SIZES.keys()}
    val_losses["total"] = 0.0
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for i, (images, labels_dict) in enumerate(progress_bar):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            for j in range(images.size(0)):
                _, losses = calculate_loss({k: v[j:j+1] for k, v in outputs.items()}, 
                                         {k: labels_dict[k][j] for k in labels_dict.keys()}, 
                                         device)
                
                # Update validation losses
                for attr, loss_val in losses.items():
                    val_losses[attr] += loss_val / len(dataloader)
    
    return val_losses

def evaluate(model, dataloader, device):
    model.eval()
    correct = {attr: 0 for attr in ATTRIBUTE_SIZES.keys() if attr != "quality_score"}
    total = 0
    quality_mse = 0.0
    
    with torch.no_grad():
        for images, labels_dict in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            total += images.size(0)
            
            # Forward pass
            outputs = model(images)
            
            for j in range(images.size(0)):
                for attr, pred in outputs.items():
                    if attr == "quality_score":
                        # Calculate MSE for quality score
                        target = labels_dict[attr][j].item()
                        predicted = pred[j].item()
                        quality_mse += (predicted - target) ** 2
                    else:
                        # Calculate accuracy for classification attributes
                        _, predicted = torch.max(pred[j].unsqueeze(0), 1)
                        target = labels_dict[attr][j].item()
                        correct[attr] += (predicted.item() == target)
    
    # Calculate accuracies and MSE
    accuracies = {attr: (correct[attr] / total) * 100 for attr in correct.keys()}
    quality_mse = quality_mse / total
    
    return accuracies, quality_mse

def save_checkpoint(model, optimizer, epoch, losses, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0, {}
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint.get('losses', {})
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, losses

def main():
    parser = argparse.ArgumentParser(description='Train facial attribute classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to the data JSON file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to load checkpoint from')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset
    dataset = FaceAttributeDataset(args.data, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset (80% train, 10% validation, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = AttributeClassifier(ATTRIBUTE_SIZES)
    model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint:
        start_epoch, _ = load_checkpoint(model, optimizer, args.load_checkpoint, device)
        start_epoch += 1  # Start from the next epoch
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch} train loss: {train_losses['total']:.4f}")
        
        # Validate
        val_losses = validate(model, val_loader, device)
        print(f"Epoch {epoch} validation loss: {val_losses['total']:.4f}")
        
        # Print individual attribute losses
        for attr in ATTRIBUTE_SIZES.keys():
            print(f"  {attr}: train_loss = {train_losses[attr]:.4f}, val_loss = {val_losses[attr]:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_losses, args.checkpoint_dir)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    accuracies, quality_mse = evaluate(model, test_loader, device)
    
    print("\nTest Results:")
    for attr, acc in accuracies.items():
        print(f"  {attr} accuracy: {acc:.2f}%")
    print(f"  quality_score MSE: {quality_mse:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    main()
