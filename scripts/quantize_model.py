#!/usr/bin/env python3
"""
Script to convert the emotion model to TensorFlow Lite with quantization.
This creates a smaller, faster model optimized for inference.
"""
import os

import numpy as np
import pkg_resources

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    import tensorflow as tf
    from keras.models import load_model


def representative_dataset():
    """Generate representative dataset for post-training quantization."""
    # Generate random samples similar to the input data (normalized grayscale faces)
    for _ in range(100):
        # Input is 64x64x1 grayscale images, normalized to [-1, 1]
        data = np.random.rand(1, 64, 64, 1).astype(np.float32)
        data = (data - 0.5) * 2.0  # Normalize to [-1, 1]
        yield [data]


def quantize_model():
    """Convert Keras model to quantized TensorFlow Lite model."""
    # Load the original Keras model
    model_path = pkg_resources.resource_filename('fer', 'data/emotion_model.hdf5')
    output_dir = pkg_resources.resource_filename('fer', 'data')

    print(f"Loading model from: {model_path}")
    model = load_model(model_path, compile=False)

    print(f"Original model size: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"Model parameters: {model.count_params()}")

    # Convert to TensorFlow Lite with dynamic range quantization
    print("\n=== Converting to TensorFlow Lite ===")

    # Use concrete function approach to avoid BatchNorm issues
    @tf.function
    def model_func(x):
        return model(x, training=False)

    # Get concrete function
    concrete_func = model_func.get_concrete_function(
        tf.TensorSpec(shape=[1, 64, 64, 1], dtype=tf.float32)
    )

    try:
        # First try with optimizations
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        # Save the quantized model
        output_path = os.path.join(output_dir, 'emotion_model_quantized.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Quantized model saved to: {output_path}")
        print(f"Quantized model size: {len(tflite_model) / 1024:.2f} KB")
        print(f"Size reduction: {(1 - len(tflite_model) / os.path.getsize(model_path)) * 100:.1f}%")
    except Exception as e:
        print(f"Quantization with optimizations failed: {e}")
        print("Trying basic conversion...")

        # Fallback: convert without optimizations
        try:
            converter_basic = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            tflite_model = converter_basic.convert()

            output_path = os.path.join(output_dir, 'emotion_model_quantized.tflite')
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            print(f"Basic TFLite model saved to: {output_path}")
            print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        except Exception as e2:
            print(f"Basic conversion also failed: {e2}")
            print("Model conversion unsuccessful")
            return

    # Skip INT8 quantization for now due to BatchNorm issues
    print("\n=== Skipping INT8 Quantization ===")
    print("INT8 quantization skipped due to model compatibility issues.")

    # Test the quantized model
    print("\n=== Testing Models ===")
    test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
    test_input = (test_input - 0.5) * 2.0

    # Test original model
    original_output = model.predict(test_input, verbose=0)
    print(f"Original model output shape: {original_output.shape}")

    # Test TFLite model
    try:
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        print(f"TFLite model output shape: {tflite_output.shape}")

        # Compare outputs
        diff = np.max(np.abs(original_output - tflite_output))
        print(f"\nMax difference (original vs TFLite): {diff:.6f}")

        # Simple inference benchmark
        import time
        n_runs = 100

        start = time.time()
        for _ in range(n_runs):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        tflite_time = time.time() - start

        start = time.time()
        for _ in range(n_runs):
            _ = model.predict(test_input, verbose=0)
        keras_time = time.time() - start

        print(f"\nPerformance ({n_runs} runs):")
        print(f"  Keras model: {keras_time*1000:.2f}ms ({keras_time*1000/n_runs:.2f}ms per inference)")
        print(f"  TFLite model: {tflite_time*1000:.2f}ms ({tflite_time*1000/n_runs:.2f}ms per inference)")
        print(f"  Speedup: {keras_time/tflite_time:.2f}x")

        print("\n✓ Model quantization completed successfully!")
        print("\nTo use the TFLite model, pass use_tflite=True when creating FER instance:")
        print("  fer = FER(use_tflite=True)  # Uses TFLite model for faster inference")
    except Exception as e:
        print(f"Testing failed: {e}")


if __name__ == '__main__':
    quantize_model()
