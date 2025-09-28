# model-training/src/download_from_drive.py
import os
import gdown
from pathlib import Path

def download_drive_folder():
    """Download all images from shared Google Drive folder"""
    
    # Your specific folder ID
    folder_id = "1_lXVacC-jMqso_vWrEwUyvCDcVY0eaJM"
    
    # Create output directory
    output_dir = Path("model-training/data/raw/crowdsourced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download folder contents
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{folder_id}",
            output=str(output_dir),
            quiet=False,
            use_cookies=False
        )
        print("✅ Downloaded crowdsourced images!")
        
        # Count downloaded images
        image_count = len(list(output_dir.glob("**/*.jpg"))) + len(list(output_dir.glob("**/*.png")))
        print(f"📸 Found {image_count} images")
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        print("💡 Make sure the folder is set to 'Anyone with the link can edit'")

if __name__ == "__main__":
    download_drive_folder()