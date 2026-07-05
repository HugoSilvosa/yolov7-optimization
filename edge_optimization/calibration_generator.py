import os
import cv2
import numpy as np
from pathlib import Path

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def get_calibration_images():
    img_dir = Path("data/coco128/images/train2017")
    if not img_dir.exists():
        raise FileNotFoundError(f"Calibration image directory {img_dir} not found. Run download_coco128.py first.")
    
    # Supported image extensions
    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo']
    images = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in img_formats])
    return images

def representative_dataset_gen_nhwc():
    images = get_calibration_images()
    for img_path in images:
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            continue
        # Preprocessing:
        # 1. Letterbox resize to 640x640 (static dimensions, auto=False)
        img = letterbox(img0, (640, 640), auto=False)[0]
        # 2. BGR to RGB
        img = img[:, :, ::-1]
        # 3. Uint8 to Float32 and normalization to [0.0, 1.0]
        img = img.astype(np.float32) / 255.0
        # 4. Add batch dimension: [1, 640, 640, 3]
        img = np.expand_dims(img, axis=0)
        yield [img]

def representative_dataset_gen_nchw():
    images = get_calibration_images()
    for img_path in images:
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            continue
        # Preprocessing:
        # 1. Letterbox resize to 640x640 (static dimensions, auto=False)
        img = letterbox(img0, (640, 640), auto=False)[0]
        # 2. BGR to RGB and transpose to NCHW [3, 640, 640]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # 3. Uint8 to Float32 and normalization to [0.0, 1.0]
        img = img.astype(np.float32) / 255.0
        # 4. Add batch dimension: [1, 3, 640, 640]
        img = np.expand_dims(img, axis=0)
        yield [img]

def main():
    print("Validating Calibration Data Generator...")
    try:
        images = get_calibration_images()
        print(f"Total calibration images available: {len(images)}")
        
        # Test NHWC Generator
        nhwc_gen = representative_dataset_gen_nhwc()
        first_nhwc = next(nhwc_gen)[0]
        print(f"NHWC Tensor Shape: {first_nhwc.shape}")
        print(f"NHWC Tensor Dtype: {first_nhwc.dtype}")
        print(f"NHWC Tensor Range: [{first_nhwc.min():.4f}, {first_nhwc.max():.4f}]")
        assert first_nhwc.shape == (1, 640, 640, 3), f"Unexpected shape: {first_nhwc.shape}"
        assert first_nhwc.dtype == np.float32, f"Unexpected dtype: {first_nhwc.dtype}"
        assert 0.0 <= first_nhwc.min() <= first_nhwc.max() <= 1.0, f"Normalization out of bounds: [{first_nhwc.min()}, {first_nhwc.max()}]"
        
        # Test NCHW Generator
        nchw_gen = representative_dataset_gen_nchw()
        first_nchw = next(nchw_gen)[0]
        print(f"NCHW Tensor Shape: {first_nchw.shape}")
        print(f"NCHW Tensor Dtype: {first_nchw.dtype}")
        print(f"NCHW Tensor Range: [{first_nchw.min():.4f}, {first_nchw.max():.4f}]")
        assert first_nchw.shape == (1, 3, 640, 640), f"Unexpected shape: {first_nchw.shape}"
        assert first_nchw.dtype == np.float32, f"Unexpected dtype: {first_nchw.dtype}"
        assert 0.0 <= first_nchw.min() <= first_nchw.max() <= 1.0, f"Normalization out of bounds: [{first_nchw.min()}, {first_nchw.max()}]"
        
        print("\nAll generator assertions passed successfully!")
    except Exception as e:
        print(f"Verification failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
