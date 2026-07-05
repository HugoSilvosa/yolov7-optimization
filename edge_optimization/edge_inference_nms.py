import os
import cv2
import numpy as np
from pathlib import Path
from calibration_generator import letterbox

# Safe interpreter retrieval (numpy ABI helper)
def get_interpreter(model_path):
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
        return interpreter
    except Exception:
        import tensorflow.lite as tf_lite
        return tf_lite.Interpreter(model_path=model_path)

# COCO Class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def decode_detections(raw_output, stride, anchors):
    # raw_output shape: (1, 3, ny, nx, 85)
    raw_output = raw_output[0]  # Remove batch dimension -> (3, ny, nx, 85)
    na, ny, nx, no = raw_output.shape
    
    # Sigmoid activation function
    # Clamping raw inputs to avoid numeric overflow in exp
    raw_output = np.clip(raw_output, -15.0, 15.0)
    y = 1.0 / (1.0 + np.exp(-raw_output))
    
    # Create coordinate grid
    grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    grid = np.stack((grid_x, grid_y), axis=-1)  # Shape (ny, nx, 2)
    grid = np.expand_dims(grid, axis=0)         # Shape (1, ny, nx, 2)
    
    # Reshape anchors -> (3, 1, 1, 2)
    anchors = np.array(anchors, dtype=np.float32).reshape(na, 1, 1, 2)
    
    # Decode coordinates:
    # cx = (sigmoid(dx) * 2 - 0.5 + grid_x) * stride
    # cy = (sigmoid(dy) * 2 - 0.5 + grid_y) * stride
    xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
    
    # w = (sigmoid(dw) * 2) ** 2 * anchor_w
    # h = (sigmoid(dh) * 2) ** 2 * anchor_h
    wh = (y[..., 2:4] * 2.0) ** 2 * anchors
    
    # Concatenate decoded values
    decoded = np.concatenate((xy, wh, y[..., 4:]), axis=-1)
    
    # Flatten grid to (3 * ny * nx, 85)
    return decoded.reshape(-1, 85)

def main():
    model_path = "models/yolov7_tiny_int8.tflite"
    image_path = "../../inference/images/horses.jpg"
    output_image_path = "plots/horses_detected.jpg"
    
    conf_threshold = 0.25
    nms_threshold = 0.45
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image {image_path} not found.")
        
    print(f"Loading TFLite Model: {model_path}")
    interpreter = get_interpreter(model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    
    # Sort output details by output grid size (height/width) in descending order:
    # Stride 8 (80x80), Stride 16 (40x40), Stride 32 (20x20)
    output_details = sorted(output_details, key=lambda x: x['shape'][2], reverse=True)
    
    # 1. Load and Preprocess Image
    print(f"Loading and preprocessing image: {image_path}")
    img0 = cv2.imread(image_path)
    orig_h, orig_w = img0.shape[:2]
    
    # Letterbox to 640x640 static size
    img_padded, ratio, (dw, dh) = letterbox(img0, (640, 640), auto=False)
    img_rgb = img_padded[:, :, ::-1]  # BGR to RGB
    img_float = img_rgb.astype(np.float32) / 255.0  # Normalize to [0.0, 1.0]
    img_batch = np.expand_dims(img_float, axis=0)  # Shape [1, 640, 640, 3]
    
    # 2. Input Handling
    if input_details['dtype'] == np.int8:
        print("Model expects INT8 input. Quantizing input tensor...")
        input_scale, input_zero_point = input_details['quantization']
        img_input = np.round((img_batch / input_scale) + input_zero_point).astype(np.int8)
    else:
        print("Model expects FLOAT32 input. Passing raw float32 input...")
        img_input = img_batch.astype(np.float32)
    
    # 3. Invoke Model Inference
    interpreter.set_tensor(input_details['index'], img_input)
    interpreter.invoke()
    
    # 4. Retrieve, Dequantize, and Decode outputs from the 3 branches
    decoded_preds = []
    
    # Anchor configs corresponding to Stride 8, 16, 32
    anchors_list = [
        [[10, 13], [16, 30], [33, 23]],       # Stride 8
        [[30, 61], [62, 45], [59, 119]],      # Stride 16
        [[116, 90], [156, 198], [373, 326]]   # Stride 32
    ]
    
    print("Decoding outputs from all detection branches...")
    for idx, detail in enumerate(output_details):
        raw_output = interpreter.get_tensor(detail['index'])
        ny = detail['shape'][2]
        stride = 640 // ny
        anchors = anchors_list[idx]
        
        # Dequantize branch
        if detail['dtype'] == np.int8:
            output_scale, output_zero_point = detail['quantization']
            predictions_branch = (raw_output.astype(np.float32) - output_zero_point) * output_scale
        else:
            predictions_branch = raw_output.astype(np.float32)
            
        # Decode boxes
        decoded_branch = decode_detections(predictions_branch, stride, anchors)
        decoded_preds.append(decoded_branch)
        
    # Concatenate all grid layers together
    predictions = np.concatenate(decoded_preds, axis=0)
    
    # 5. Bounding Box Decoding & Confidence Filtering
    boxes = []
    confidences = []
    class_ids = []
    
    print(f"Predictions statistics: min={predictions.min():.4f}, max={predictions.max():.4f}")
    print(f"Max objectness score: {predictions[:, 4].max():.4f}")
    print(f"Max class probability: {predictions[:, 5:].max():.4f}")
    
    for pred in predictions:
        obj_conf = pred[4]
        cls_probs = pred[5:]
        class_scores = obj_conf * cls_probs
        
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]
        
        if score > conf_threshold:
            cx, cy, w, h = pred[0:4]
            # Convert center-based box (cx, cy, w, h) to top-left based (x, y, w, h)
            x_min = cx - w / 2.0
            y_min = cy - h / 2.0
            
            boxes.append([float(x_min), float(y_min), float(w), float(h)])
            confidences.append(float(score))
            class_ids.append(int(class_id))
            
    print(f"Total candidates matching score threshold (> {conf_threshold}): {len(boxes)}")
    
    # 6. Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print(f"Total detections remaining after NMS: {len(indices)}")
    
    # 7. Draw Detections mapped to original image dimensions
    if len(indices) > 0:
        flat_indices = np.array(indices).flatten()
        colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0]]
        
        for idx in flat_indices:
            box = boxes[idx]
            x_min_padded, y_min_padded, w_padded, h_padded = box
            score = confidences[idx]
            class_id = class_ids[idx]
            class_name = COCO_CLASSES[class_id]
            
            # Map back to original image size
            x_min_unpad = x_min_padded - dw
            y_min_unpad = y_min_padded - dh
            x_min_orig = x_min_unpad / ratio[0]
            y_min_orig = y_min_unpad / ratio[1]
            w_orig = w_padded / ratio[0]
            h_orig = h_padded / ratio[1]
            
            # Clip coordinates
            x1 = max(0, int(round(x_min_orig)))
            y1 = max(0, int(round(y_min_orig)))
            x2 = max(0, min(int(round(x_min_orig + w_orig)), orig_w))
            y2 = max(0, min(int(round(y_min_orig + h_orig)), orig_h))
            
            color = colors[class_id % len(colors)]
            cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {score:.2f}"
            print(f"Detected: {label} at bbox [{x1}, {y1}, {x2}, {y2}]")
            
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(y1, label_size[1] + 10)
            cv2.rectangle(img0, (x1, y_label - label_size[1] - 4), (x1 + label_size[0], y_label + base_line - 4), color, -1)
            cv2.putText(img0, label, (x1, y_label - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
            
        cv2.imwrite(output_image_path, img0)
        print(f"\nThe output image with detection overlays saved to: {output_image_path}")
    else:
        print("No objects detected.")

if __name__ == "__main__":
    main()
