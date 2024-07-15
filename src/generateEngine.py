import torch  # PyTorch library for building and training neural networks
import torch.nn as nn  # PyTorch module containing various neural network layers and functions
import tensorrt as trt  # TensorRT library for optimizing and running deep learning models on NVIDIA GPUs
import numpy as np  # NumPy library for numerical operations

class SimpleTokenizer(nn.Module):
    """
    A simple tokenizer model using PyTorch, which consists of an embedding layer and a fully connected layer.
    """
    def __init__(self, vocab_size=256, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer to convert input indices to dense vectors
        self.fc = nn.Linear(embedding_dim, vocab_size)  # Fully connected layer to map embeddings back to the vocabulary size

    def forward(self, x):
        x = self.embedding(x)  # Apply the embedding layer
        x = torch.mean(x, dim=1)  # Compute the mean of the embeddings across the sequence dimension
        return self.fc(x)  # Apply the fully connected layer and return the output

# Create and save the PyTorch model
model = SimpleTokenizer()  # Instantiate the SimpleTokenizer model
dummy_input = torch.randint(0, 256, (1, 100)).long()  # Generate a dummy input tensor with random integers
# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "simple_tokenizer.onnx", input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size', 1: 'sequence'}})

# Set up TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # Logger to capture TensorRT warnings and errors

def build_engine(onnx_file_path):
    """
    Build a TensorRT engine from an ONNX file.
    """
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # Define the explicit batch flag
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_file_path, 'rb') as model:  # Open the ONNX model file
            if not parser.parse(model.read()):  # Parse the ONNX model
                print('ERROR: Failed to parse the ONNX file.')  # Print an error message if parsing fails
                for error in range(parser.num_errors):  # Iterate over and print all parsing errors
                    print(parser.get_error(error))
                return None
        
        config = builder.create_builder_config()  # Create a builder configuration
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # Set the memory pool limit to 1GB
        
        profile = builder.create_optimization_profile()  # Create an optimization profile
        profile.set_shape("input", (1, 1), (1, 100), (1, 200))  # Set the shape range for the input tensor
        config.add_optimization_profile(profile)  # Add the optimization profile to the configuration
        
        return builder.build_serialized_network(network, config)  # Build and serialize the network using the configuration

# Build the TensorRT engine from the ONNX model
engine = build_engine("simple_tokenizer.onnx")

# Save the TensorRT engine to a file
with open("simple_tokenizer.trt", "wb") as f:
    f.write(engine)  # Write the serialized engine to the file

print("TensorRT engine created and saved as 'simple_tokenizer.trt'")  # Print a message indicating the engine has been saved
