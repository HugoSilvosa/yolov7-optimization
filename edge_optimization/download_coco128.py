import os
import urllib.request
import zipfile
from pathlib import Path

def main():
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    zip_path = Path("coco128.zip")
    extract_dir = Path("data")  # Will extract to data/coco128/

    print(f"Downloading COCO-128 from {url}...")
    try:
        # Download file
        urllib.request.urlretrieve(url, zip_path)
        print("Download completed successfully.")
        
        # Unzip
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed successfully.")
        
        # Clean up zip
        if zip_path.exists():
            zip_path.unlink()
            print("Removed temporary zip file.")
            
        # Verify images count
        img_dir = Path("data/coco128/images/train2017")
        if img_dir.exists():
            images = list(img_dir.glob("*.*"))
            print(f"Verification successful: Found {len(images)} images in {img_dir}")
            if len(images) != 128:
                print("WARNING: Expected exactly 128 images, but found a different amount!")
        else:
            print(f"ERROR: Expected directory {img_dir} does not exist!")

    except Exception as e:
        print(f"An error occurred: {e}")
        if zip_path.exists():
            zip_path.unlink()

if __name__ == "__main__":
    main()
