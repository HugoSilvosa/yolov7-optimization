import os
import tensorflow as tf
from calibration_generator import representative_dataset_gen_nhwc

def convert_drq(saved_model_dir, output_path):
    print("Converting model to TFLite with Dynamic Range Quantization...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Dynamic Range Quantized model saved to: {output_path}")

def convert_int8(saved_model_dir, output_path):
    print("Converting model to TFLite with Full Integer Quantization...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen_nhwc
    # Standard full integer quantization fallback to float is allowed (defaults to hybrid)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Full Integer Quantized model saved to: {output_path}")

def print_model_info(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n--- Model: {os.path.basename(model_path)} ---")
    print(f"File Size: {size_mb:.2f} MB")
    print("Input Details:")
    for detail in input_details:
        print(f"  Name: {detail['name']}, Shape: {detail['shape']}, Dtype: {detail['dtype']}")
    print("Output Details:")
    for detail in output_details:
        print(f"  Name: {detail['name']}, Shape: {detail['shape']}, Dtype: {detail['dtype']}")

def main():
    saved_model_dir = "models/yolov7-tiny_saved_model"
    drq_model_path = os.path.join(saved_model_dir, "yolov7-tiny_dynamic_range.tflite")
    int8_model_path = os.path.join(saved_model_dir, "yolov7-tiny_full_integer.tflite")

    # Run Dynamic Range Quantization
    convert_drq(saved_model_dir, drq_model_path)
    print_model_info(drq_model_path)

    # Run Full Integer Quantization
    convert_int8(saved_model_dir, int8_model_path)
    print_model_info(int8_model_path)

if __name__ == "__main__":
    main()
