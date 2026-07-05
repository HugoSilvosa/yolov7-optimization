import os
import cv2
import numpy as np
import tensorflow as tf
from calibration_generator import letterbox

def run_tflite_inference(model_path, input_tensor):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Check if the model expects int8 input
    if input_details['dtype'] == np.int8:
        # Quantize the input float image using input scale and zero point
        scale, zero_point = input_details['quantization']
        input_tensor_quant = (input_tensor / scale) + zero_point
        input_tensor_quant = np.round(input_tensor_quant).astype(np.int8)
        interpreter.set_tensor(input_details['index'], input_tensor_quant)
    else:
        interpreter.set_tensor(input_details['index'], input_tensor)
        
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details['index'])
    
    # Dequantize if the output is int8
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
        
    return output

def print_stats(name, tensor):
    print(f"\n=== Statistics for {name} ===")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Min: {tensor.min():.6f}")
    print(f"Max: {tensor.max():.6f}")
    print(f"Mean: {tensor.mean():.6f}")
    print(f"Std Dev: {tensor.std():.6f}")
    
    # Count non-zero elements
    non_zero_count = np.count_nonzero(tensor)
    total_elements = tensor.size
    print(f"Non-zero elements: {non_zero_count} / {total_elements} ({100.0 * non_zero_count / total_elements:.2f}%)")
    
    # Check for collapse (all zeros or extremely small values)
    if non_zero_count == 0:
        print("WARNING: Quantization collapse detected! (All outputs are zero)")
    elif tensor.max() < 1e-4:
        print("WARNING: Extremely low output values detected, possible quantization collapse!")
    else:
        print("Status: Stable (Non-zero outputs detected)")

def main():
    image_path = "../inference/images/horses.jpg"
    float_model = "yolov7-tiny_saved_model/yolov7-tiny_float32.tflite"
    quant_model = "yolov7-tiny_saved_model/yolov7-tiny_full_integer.tflite"
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image {image_path} not found.")
        
    # Preprocess image
    img0 = cv2.imread(image_path)
    img = letterbox(img0, (640, 640), auto=False)[0]
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Shape [1, 640, 640, 3] NHWC
    
    print(f"Input image preprocessed shape: {img.shape}")
    
    # Run FP32 inference
    print("\nRunning inference on FP32 model...")
    fp32_output = run_tflite_inference(float_model, img)
    print_stats("FP32 Model Output", fp32_output)
    
    # Run INT8 Hybrid inference
    print("\nRunning inference on Full Integer model...")
    int8_output = run_tflite_inference(quant_model, img)
    print_stats("Full Integer Model Output", int8_output)

if __name__ == "__main__":
    main()
