import argparse
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from torch.utils.data import Subset
from data_loader import DotsOcrJsonl
from config import stopphrases, prompt, numbers, persian_words, category_units

def fix_persian_text(text: str) -> str:
    """Fix Persian text for proper display."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def main():
    # Set up argument parser with default values
    parser = argparse.ArgumentParser(description='Generate OCR training data')
    
    parser.add_argument('--data_path', type=str, default=" "*1000, 
                       help='Path to data source')
    parser.add_argument('--processor', type=str, default="", 
                       help='Text processor identifier (default: empty)')
    parser.add_argument('--mode', type=str, default="train", 
                       help='Dataset mode (default: train)')
    parser.add_argument('--output_path', type=str, default="generated_images", 
                       help='Output directory (default: generated_images)')
    parser.add_argument('--max_samples', type=int, default=10, 
                       help='Maximum samples to process (default: 10)')
    parser.add_argument('--visualize', default = False, action='store_true', 
                       help='Visualize samples (default: False)')
    parser.add_argument('--save', default = True, action='store_true', 
                       help='Save samples to disk (default: False)')
    parser.add_argument('--shuffle', action='store_true', default=True,
                       help='Shuffle dataset (default: True)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create dataset with the provided arguments
    data_train = DotsOcrJsonl(
        args.data_path,
        stopphrases,
        prompt,
        numbers,
        persian_words,
        category_units,
        args.processor,
        args.mode
    )
    
    # Create subset
    train_ds = Subset(data_train, list(range(len(data_train))))
    
    # Get indices and shuffle if requested
    indices = list(range(len(train_ds)))
    if args.shuffle:
        random.shuffle(indices)
    
    # Limit to max_samples
    indices = indices[:args.max_samples]
    
    # Visualization
    if args.visualize:
        print(f"Visualizing {len(indices)} samples...")
        for i, ind in enumerate(indices):
            item = train_ds[ind]
            
            plt.imshow(np.array(item['image']))
            plt.title(f"Sample {i+1}/{len(indices)}")
            plt.axis('off')
            plt.show()
            
            print(fix_persian_text(item["answer"]))
            print("_" * 100)
    
    # Save to disk
    if args.save:
        print(f"Saving {len(indices)} samples to {args.output_path}...")
        
        images_dir = os.path.join(args.output_path, "images")
        labels_dir = os.path.join(args.output_path, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for i, ind in enumerate(indices):
            item = train_ds[ind]
            
            # Save image
            plt.imsave(os.path.join(images_dir, f"{i}.png"), np.array(item["image"]))
            
            # Save label
            with open(os.path.join(labels_dir, f"{i}.txt"), "w", encoding="utf-8") as f:
                f.write(item["answer"])
        
        print(f"Saved {len(indices)} samples successfully!")

if __name__ == "__main__":
    main()
