import tensorflow as tf

print("--- TF Hardware Status ---")
print("TensorFlow Version:", tf.__version__)
print("Built with CUDA Support:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')
print("Physical GPUs Found:", len(gpus))

if gpus:
    for gpu in gpus:
        print("  Device:", gpu.name)
else:
    print("\n[!] No GPU found by TensorFlow.")
    print("    Note: Modern TF (>2.10) requires WSL2 on Windows to access the GPU.")