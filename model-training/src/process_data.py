import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.raw_dir = Path("data/raw/crowdsourced")
        self.processed_dir = Path("data/processed")
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            "data/processed/images",
            "data/processed/thumbnails", 
            "data/annotations",
            "data/processed/train",
            "data/processed/val"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def process_all_images(self):
        """Process all raw images"""
        image_files = []
        
        # Find all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(self.raw_dir.rglob(ext))
        
        print(f"Found {len(image_files)} images to process")
        
        processed_count = 0
        metadata = []
        
        for img_path in image_files:
            try:
                processed_data = self.process_single_image(img_path, processed_count)
                if processed_data:
                    metadata.append(processed_data)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count} images...")
                        
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save metadata
        self.save_metadata(metadata)
        print(f"✅ Successfully processed {processed_count} images!")
        
        return processed_count
    
    def process_single_image(self, img_path, index):
        """Process a single image"""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Get original dimensions
        original_height, original_width = img.shape[:2]
        
        # Check if image is too small
        if original_height < 100 or original_width < 100:
            print(f"Skipping {img_path.name}: too small ({original_width}x{original_height})")
            return None
        
        # Resize while maintaining aspect ratio
        processed_img = self.resize_image(img, target_size=512)
        
        # Create filename
        new_filename = f"sticker_{index:04d}.jpg"
        
        # Save processed image
        processed_path = self.processed_dir / "images" / new_filename
        cv2.imwrite(str(processed_path), processed_img)
        
        # Create thumbnail
        thumbnail = self.resize_image(img, target_size=128)
        thumbnail_path = self.processed_dir / "thumbnails" / new_filename
        cv2.imwrite(str(thumbnail_path), thumbnail)
        
        # Return metadata
        return {
            "filename": new_filename,
            "original_path": str(img_path),
            "original_size": [original_width, original_height],
            "processed_size": list(processed_img.shape[:2][::-1]),  # [width, height]
            "file_size": img_path.stat().st_size,
            "processed_date": datetime.now().isoformat()
        }
    
    def resize_image(self, img, target_size=512):
        """Resize image while maintaining aspect ratio"""
        height, width = img.shape[:2]
        
        # Calculate scaling factor
        scale = target_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create square image with padding
        square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate padding
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        
        # Place resized image in center
        square_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return square_img
    
    def save_metadata(self, metadata):
        """Save processing metadata"""
        metadata_file = self.processed_dir / "metadata.json"
        
        summary = {
            "total_images": len(metadata),
            "processing_date": datetime.now().isoformat(),
            "images": metadata
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Metadata saved to {metadata_file}")
    
    def split_train_val(self, val_split=0.2):
        """Split processed images into train/validation sets"""
        images_dir = self.processed_dir / "images"
        train_dir = self.processed_dir / "train"
        val_dir = self.processed_dir / "val"
        
        # Get all processed images
        image_files = list(images_dir.glob("*.jpg"))
        
        if len(image_files) == 0:
            print("❌ No processed images found!")
            return
        
        # Shuffle and split
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(image_files)
        
        split_idx = int(len(image_files) * (1 - val_split))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy files to train/val directories
        for img_file in train_files:
            shutil.copy2(img_file, train_dir / img_file.name)
        
        for img_file in val_files:
            shutil.copy2(img_file, val_dir / img_file.name)
        
        print(f"📂 Split complete:")
        print(f"   Training: {len(train_files)} images")
        print(f"   Validation: {len(val_files)} images")

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Process all images
    count = processor.process_all_images()
    
    if count > 0:
        # Split into train/validation
        processor.split_train_val()
        print("✅ Data processing complete!")
    else:
        print("❌ No images were processed")