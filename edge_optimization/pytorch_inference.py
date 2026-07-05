import os
import sys
import json
import torch
import cv2
from pathlib import Path

# Add parent directory to Python path to import YOLOv7 modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

def main():
    weights = "../yolov7-tiny.pt"
    img_dir = Path("data/coco128/images/train2017")
    output_json = "data/pytorch_detections.json"
    num_images = 20
    
    device = torch.device("cpu")
    print(f"Loading PyTorch model: {weights} on CPU...")
    model = attempt_load(weights, map_location=device)
    model.eval()
    
    # Get image paths
    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo']
    images = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in img_formats])[:num_images]
    print(f"Found {len(images)} images for evaluation.")
    
    results = {}
    
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{num_images}] Processing {img_path.name}...")
        
        # Load image
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        # Preprocess exactly like detect.py
        img = letterbox(img0, 640, stride=32, auto=True)[0]  # note auto=True is default in PyTorch NMS
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img) if 'np' in globals() else img
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        # Inference
        with torch.no_grad():
            pred = model(img_tensor)[0]
            
        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
        
        img_detections = []
        if pred is not None and len(pred):
            # Scale coordinates back to original image size
            scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape)
            
            for det in pred:
                # det format: [x1, y1, x2, y2, confidence, class_id]
                x1, y1, x2, y2, conf, cls = det.tolist()
                img_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": int(cls)
                })
                
        results[img_path.name] = img_detections
        print(f"  Detected {len(img_detections)} objects.")
        
    # Write to JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Detections saved to {output_json}")

if __name__ == "__main__":
    # Import numpy here just in case
    import numpy as np
    main()
