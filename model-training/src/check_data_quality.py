_import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class DataQualityChecker:
    def __init__(self):
        self.processed_dir = Path("data/processed/images")
    
    def check_image_quality(self):
        """Analyze image quality metrics"""
        image_files = list(self.processed_dir.glob("*.jpg"))
        
        if len(image_files) == 0:
            print("❌ No processed images found!")
            return
        
        print(f"🔍 Analyzing {len(image_files)} images...")
        
        quality_metrics = {
            'blur_scores': [],
            'brightness_scores': [],
            'contrast_scores': [],
            'file_sizes': []
        }
        
        low_quality_images = []
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Calculate quality metrics
            blur_score = self.calculate_blur_score(img)
            brightness = self.calculate_brightness(img)
            contrast = self.calculate_contrast(img)
            file_size = img_path.stat().st_size / 1024  # KB
            
            quality_metrics['blur_scores'].append(blur_score)
            quality_metrics['brightness_scores'].append(brightness)
            quality_metrics['contrast_scores'].append(contrast)
            quality_metrics['file_sizes'].append(file_size)
            
            # Flag low quality images
            if blur_score < 100 or brightness < 50 or brightness > 200:
                low_quality_images.append({
                    'filename': img_path.name,
                    'blur_score': blur_score,
                    'brightness': brightness,
                    'contrast': contrast
                })
        
        self.print_quality_report(quality_metrics, low_quality_images)
        return quality_metrics
    
    def calculate_blur_score(self, img):
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def calculate_brightness(self, img):
        """Calculate average brightness"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 2])
    
    def calculate_contrast(self, img):
        """Calculate contrast using standard deviation"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def print_quality_report(self, metrics, low_quality):
        """Print quality analysis report"""
        print("\n" + "="*50)
        print("📊 DATA QUALITY REPORT")
        print("="*50)
        
        print(f"📸 Total Images: {len(metrics['blur_scores'])}")
        print(f"⚠️  Low Quality Images: {len(low_quality)}")
        
        print(f"\n🌟 Quality Metrics:")
        print(f"   Blur Score - Avg: {np.mean(metrics['blur_scores']):.1f}, Min: {np.min(metrics['blur_scores']):.1f}")
        print(f"   Brightness - Avg: {np.mean(metrics['brightness_scores']):.1f}, Range: {np.min(metrics['brightness_scores']):.1f}-{np.max(metrics['brightness_scores']):.1f}")
        print(f"   Contrast - Avg: {np.mean(metrics['contrast_scores']):.1f}")
        print(f"   File Size - Avg: {np.mean(metrics['file_sizes']):.1f} KB")
        
        if low_quality:
            print(f"\n⚠️  Low Quality Images to Review:")
            for img in low_quality[:5]:  # Show first 5
                print(f"   {img['filename']}: blur={img['blur_score']:.1f}, brightness={img['brightness']:.1f}")
    
    def visualize_sample_images(self, num_samples=9):
        """Display sample images for visual inspection"""
        image_files = list(self.processed_dir.glob("*.jpg"))
        
        if len(image_files) < num_samples:
            num_samples = len(image_files)
        
        # Select random sample
        sample_files = np.random.choice(image_files, num_samples, replace=False)
        
        # Create subplot
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, img_path in enumerate(sample_files):
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(img_path.name, fontsize=8)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("data/processed/sample_images.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Sample visualization saved as 'data/processed/sample_images.png'")

if __name__ == "__main__":
    checker = DataQualityChecker()
    checker.check_image_quality()
    checker.visualize_sample_images()