from tensorflow.keras.models import load_model

# Load the model from the .h5 file
model = load_model('posture_model_11_26.h5')

print(model.summary())

# Access individual layers
for layer in model.layers:
    print(f"Layer name: {layer.name}")
    print(f"Layer type: {layer.__class__.__name__}")
    print(f"Layer output shape: {layer.output_shape}")
    print(f"Number of parameters: {layer.count_params()}")

# Extract weights for each layer
for layer in model.layers:
    weights = layer.get_weights()
    print(f"Layer {layer.name} weights: {weights}")