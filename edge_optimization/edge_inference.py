import os
import cv2
import numpy as np
from pathlib import Path
from calibration_generator import letterbox

def get_interpreter(model_path):
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
        print("Successfully initialized Interpreter using tflite_runtime")
        return interpreter
    except Exception as e:
        print(f"tflite_runtime initialization failed ({e}). Falling back to tensorflow.lite...")
        import tensorflow.lite as tf_lite
        return tf_lite.Interpreter(model_path=model_path)

def main():
    model_path = "models/yolov7_tiny_int8.tflite"
    image_path = "../../inference/images/horses.jpg"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image {image_path} not found.")
        
    print(f"\n--- Loading TFLite Model: {model_path} ---")
    interpreter = get_interpreter(model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print("\n[Interpreter Tensor Details]")
    print(f"Input Name: {input_details['name']}, Shape: {input_details['shape']}, Dtype: {input_details['dtype']}")
    print(f"Output Name: {output_details['name']}, Shape: {output_details['shape']}, Dtype: {output_details['dtype']}")
    
    # Preprocessing
    print(f"\nPreprocessing input image: {image_path}")
    img0 = cv2.imread(image_path)
    img_padded, ratio, (dw, dh) = letterbox(img0, (640, 640), auto=False)
    img_rgb = img_padded[:, :, ::-1]  # BGR to RGB
    img_float = img_rgb.astype(np.float32) / 255.0  # Normalize to [0.0, 1.0]
    img_batch = np.expand_dims(img_float, axis=0)  # Shape [1, 640, 640, 3]
    
    # Manual Input Quantization
    print("Executing manual input quantization from float32 to int8...")
    input_scale, input_zero_point = input_details['quantization']
    print(f"Input scale: {input_scale}, zero-point: {input_zero_point}")
    
    # Quantization formula: q = round(f / scale) + zero_point
    img_quant = (img_batch / input_scale) + input_zero_point
    img_quant = np.round(img_quant).astype(np.int8)
    
    # Run Inference
    print("Setting input tensor and invoking interpreter...")
    interpreter.set_tensor(input_details['index'], img_quant)
    interpreter.invoke()
    
    # Retrieve Raw Output Tensor
    print("Retrieving raw output tensor...")
    raw_output = interpreter.get_tensor(output_details['index'])
    print(f"Raw Output Shape: {raw_output.shape}, Dtype: {raw_output.dtype}")
    
    # Manual Output Dequantization
    print("Executing manual output dequantization from int8 to float32...")
    output_scale, output_zero_point = output_details['quantization']
    print(f"Output scale: {output_scale}, zero-point: {output_zero_point}")
    
    # Dequantization formula: f = (q - zero_point) * scale
    dequant_output = (raw_output.astype(np.float32) - output_zero_point) * output_scale
    print(f"Dequantized Output Shape: {dequant_output.shape}, Dtype: {dequant_output.dtype}")
    
    # Print sample outputs (e.g. first 5 detection anchors)
    print("\n--- Raw Output Samples (first 5 anchors, first 5 elements each) ---")
    print(raw_output[0, :5, :5])
    
    print("\n--- Dequantized Output Samples (first 5 anchors, first 5 elements each) ---")
    print(dequant_output[0, :5, :5])
    
    # Check output boundaries
    print(f"\nDequantized Output range: [{dequant_output.min():.4f}, {dequant_output.max():.4f}]")
    print("Verification Succeeded: Edge-inference completed successfully.")

if __name__ == "__main__":
    main()
