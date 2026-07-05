import os
import tensorflow as tf
from calibration_generator import representative_dataset_gen_nhwc

def main():
    saved_model_dir = "models/yolov7-tiny_saved_model"
    output_path = "models/yolov7_tiny_int8.tflite"
    
    print(f"Finalizing full INT8 TFLite model from SavedModel at: {saved_model_dir}")
    
    # Initialize converter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # 1. Apply default optimization (quantizes weights)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 2. Pass representative dataset (for dynamic range calibration of activations)
    converter.representative_dataset = representative_dataset_gen_nhwc
    
    # 3. Restrict supported ops to INT8 builtins only (enforces integer operations)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 4. Strictly enforce int8 input and output types (eliminates float32 wrapper casting)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert and write to file
    print("Executing conversion (this calibrates activation layers)...")
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Success: Strictly integer quantized TFLite model written to {output_path}")
    
    # 5. Verify zero-point and scale metadata using get_input_details / get_output_details
    print("\n--- Verifying TFLite Model Metadata ---")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model File Size: {size_mb:.2f} MB")
    
    print("\n[Input Details]")
    for idx, detail in enumerate(input_details):
        print(f"Input {idx}: Name: {detail['name']}, Shape: {detail['shape']}, Dtype: {detail['dtype']}")
        print(f"Quantization Parameters (Scale, Zero Point): {detail['quantization']}")
        assert detail['dtype'] == tf.int8.as_numpy_dtype, f"Input {idx} is not int8!"
    
    print("\n[Output Details]")
    for idx, detail in enumerate(output_details):
        print(f"Output {idx}: Name: {detail['name']}, Shape: {detail['shape']}, Dtype: {detail['dtype']}")
        print(f"Quantization Parameters (Scale, Zero Point): {detail['quantization']}")
        assert detail['dtype'] == tf.int8.as_numpy_dtype, f"Output {idx} is not int8!"
    
    print("\nValidation Succeeded: All input and output tensors are strictly INT8.")

if __name__ == "__main__":
    main()
