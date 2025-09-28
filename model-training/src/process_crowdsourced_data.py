# model-training/src/process_crowdsourced_data.py
import os
import shutil
from pathlib import Path
import cv2

def organize_crowdsourced_data():
    """Process and organize data from friends"""
    
    source_dir = Path("model-training/data/raw/crowdsourced")
    processed_dir = Path("model-training/data/processed/crowdsourced")
    processed_dir.mkdir(exist_ok=True)
    
    image_count = 0
    
    for image_path in source_dir.glob("**/*"):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Load and resize image
            img = cv2.imread(str(image_path))
            if img is not None:
                # Resize to standard size
                img_resized = cv2.resize(img, (512, 512))
                
                # Save with sequential naming
                new_name = f"crowdsourced_{image_count:04d}.jpg"
                cv2.imwrite(str(processed_dir / new_name), img_resized)
                image_count += 1
    
    print(f"Processed {image_count} crowdsourced images!")

if __name__ == "__main__":
    organize_crowdsourced_data()