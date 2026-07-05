import os
import json
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

def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0.0:
        return 0.0
    return inter_area / union_area

def decode_detections(raw_output, stride, anchors):
    # raw_output shape: (1, 3, ny, nx, 85)
    raw_output = raw_output[0]  # Remove batch dimension -> (3, ny, nx, 85)
    na, ny, nx, no = raw_output.shape
    
    raw_output = np.clip(raw_output, -15.0, 15.0)
    y = 1.0 / (1.0 + np.exp(-raw_output))
    
    grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    grid = np.stack((grid_x, grid_y), axis=-1)  # Shape (ny, nx, 2)
    grid = np.expand_dims(grid, axis=0)         # Shape (1, ny, nx, 2)
    
    anchors = np.array(anchors, dtype=np.float32).reshape(na, 1, 1, 2)
    
    xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
    wh = (y[..., 2:4] * 2.0) ** 2 * anchors
    
    decoded = np.concatenate((xy, wh, y[..., 4:]), axis=-1)
    return decoded.reshape(-1, 85)

def run_tflite_detection(interpreter, image_path, conf_threshold=0.25, nms_threshold=0.45):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    
    # Sort output details by output grid size (height/width) in descending order:
    # Stride 8 (80x80), Stride 16 (40x40), Stride 32 (20x20)
    output_details = sorted(output_details, key=lambda x: x['shape'][2], reverse=True)
    
    img0 = cv2.imread(str(image_path))
    if img0 is None:
        return []
    orig_h, orig_w = img0.shape[:2]
    
    # Preprocess
    img_padded, ratio, (dw, dh) = letterbox(img0, (640, 640), auto=False)
    img_rgb = img_padded[:, :, ::-1]  # BGR to RGB
    img_float = img_rgb.astype(np.float32) / 255.0  # Normalize to [0.0, 1.0]
    img_batch = np.expand_dims(img_float, axis=0)  # Shape [1, 640, 640, 3]
    
    # Input Quantization
    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        img_input = np.round((img_batch / input_scale) + input_zero_point).astype(np.int8)
    else:
        img_input = img_batch.astype(np.float32)
        
    interpreter.set_tensor(input_details['index'], img_input)
    interpreter.invoke()
    
    # Retrieve, Dequantize, and Decode outputs from the 3 branches
    decoded_preds = []
    
    # Anchor configs corresponding to Stride 8, 16, 32
    anchors_list = [
        [[10, 13], [16, 30], [33, 23]],       # Stride 8
        [[30, 61], [62, 45], [59, 119]],      # Stride 16
        [[116, 90], [156, 198], [373, 326]]   # Stride 32
    ]
    
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
    
    # Filter
    boxes = []
    confidences = []
    class_ids = []
    
    for pred in predictions:
        obj_conf = pred[4]
        cls_probs = pred[5:]
        class_scores = obj_conf * cls_probs
        
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]
        
        if score > conf_threshold:
            cx, cy, w, h = pred[0:4]
            x_min = cx - w / 2.0
            y_min = cy - h / 2.0
            boxes.append([float(x_min), float(y_min), float(w), float(h)])
            confidences.append(float(score))
            class_ids.append(int(class_id))
            
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    tflite_detections = []
    if len(indices) > 0:
        flat_indices = np.array(indices).flatten()
        for idx in flat_indices:
            box = boxes[idx]
            x_min_padded, y_min_padded, w_padded, h_padded = box
            score = confidences[idx]
            class_id = class_ids[idx]
            
            # Map back to original image size
            x_min_orig = (x_min_padded - dw) / ratio[0]
            y_min_orig = (y_min_padded - dh) / ratio[1]
            w_orig = w_padded / ratio[0]
            h_orig = h_padded / ratio[1]
            
            x1 = max(0, int(round(x_min_orig)))
            y1 = max(0, int(round(y_min_orig)))
            x2 = max(0, min(int(round(x_min_orig + w_orig)), orig_w))
            y2 = max(0, min(int(round(y_min_orig + h_orig)), orig_h))
            
            tflite_detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": score,
                "class_id": class_id
            })
            
    return tflite_detections

def evaluate_model(model_path, pytorch_results, img_dir, num_images=20):
    print(f"\nEvaluating accuracy for: {model_path}")
    interpreter = get_interpreter(model_path)
    interpreter.allocate_tensors()
    
    img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo']
    images = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in img_formats])[:num_images]
    
    total_py_dets = 0
    total_tf_dets = 0
    matched_dets = 0
    
    iou_accumulator = 0.0
    conf_diff_accumulator = 0.0
    class_match_count = 0
    
    for img_path in images:
        img_name = img_path.name
        py_dets = pytorch_results.get(img_name, [])
        tf_dets = run_tflite_detection(interpreter, img_path)
        
        total_py_dets += len(py_dets)
        total_tf_dets += len(tf_dets)
        
        # Match detections (greedy IoU matching)
        matched_indices = set()
        for py_det in py_dets:
            py_box = py_det["bbox"]
            py_cls = py_det["class_id"]
            py_conf = py_det["confidence"]
            
            best_iou = 0.0
            best_tf_det = None
            best_tf_idx = -1
            
            for idx, tf_det in enumerate(tf_dets):
                if idx in matched_indices:
                    continue
                if tf_det["class_id"] != py_cls:
                    continue
                
                iou = calculate_iou(py_box, tf_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tf_det = tf_det
                    best_tf_idx = idx
            
            if best_iou >= 0.5:
                matched_dets += 1
                matched_indices.add(best_tf_idx)
                iou_accumulator += best_iou
                conf_diff_accumulator += abs(py_conf - best_tf_det["confidence"])
                class_match_count += 1
                
    # Calculate stats
    match_rate = (matched_dets / total_py_dets * 100.0) if total_py_dets > 0 else 0.0
    avg_iou = (iou_accumulator / matched_dets) if matched_dets > 0 else 0.0
    avg_conf_mae = (conf_diff_accumulator / matched_dets) if matched_dets > 0 else 0.0
    
    print("----------------------------------------")
    print(f"Total PyTorch Detections: {total_py_dets}")
    print(f"Total TFLite Detections:  {total_tf_dets}")
    print(f"Matched Detections (IoU >= 0.5): {matched_dets}")
    print(f"Match Rate:              {match_rate:.2f}%")
    if matched_dets > 0:
        print(f"Average IoU of Matches:   {avg_iou:.4f}")
        print(f"Confidence Score MAE:    {avg_conf_mae:.4f}")
    print("----------------------------------------")
    
    return {
        "total_py": total_py_dets,
        "total_tf": total_tf_dets,
        "matched": matched_dets,
        "match_rate": match_rate,
        "avg_iou": avg_iou,
        "avg_conf_mae": avg_conf_mae
    }

def main():
    pytorch_json = "data/pytorch_detections.json"
    img_dir = Path("data/coco128/images/train2017")
    
    if not os.path.exists(pytorch_json):
        raise FileNotFoundError(f"PyTorch detections file {pytorch_json} not found. Run pytorch_inference.py first.")
        
    with open(pytorch_json, "r") as f:
        pytorch_results = json.load(f)
        
    # Evaluate Dynamic Range model
    drq_model = "models/yolov7-tiny_saved_model/yolov7-tiny_dynamic_range.tflite"
    evaluate_model(drq_model, pytorch_results, img_dir)
    
    # Evaluate strict INT8 model
    int8_model = "models/yolov7_tiny_int8.tflite"
    evaluate_model(int8_model, pytorch_results, img_dir)

if __name__ == "__main__":
    main()
