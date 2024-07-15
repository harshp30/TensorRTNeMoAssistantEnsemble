import tensorrt as trt  # Import TensorRT for building and using optimized neural network inference engines
import pycuda.driver as cuda  # Import PyCUDA for GPU memory allocation and management
import pycuda.autoinit  # Automatically initialize PyCUDA
import numpy as np  # Import NumPy for numerical operations

# Initialize TensorRT logger to capture warning messages
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# Initialize TensorRT plugins
trt.init_libnvinfer_plugins(None, "")

class TensorRTModel:
    def __init__(self, engine_path):
        """
        Initialize the TensorRTModel class.
        
        Args:
            engine_path (str): Path to the TensorRT engine file.
        """
        self.engine_path = engine_path  # Path to the TensorRT engine file
        self.engine = None  # Placeholder for the TensorRT engine
        self.context = None  # Placeholder for the TensorRT execution context
        self.buffers = None  # Placeholder for input/output buffers
        self.stream = cuda.Stream()  # Create a CUDA stream for asynchronous operations

    def load_engine(self):
        """
        Load the TensorRT engine from the specified file.
        """
        with open(self.engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            # Deserialize the engine from the file
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # Create an execution context for the engine
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self):
        """
        Allocate memory buffers for input and output tensors.
        """
        self.buffers = []  # Initialize the buffers list
        # Iterate over all I/O tensors
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)  # Get the name of the tensor
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))  # Get the data type of the tensor
            shape = self.engine.get_tensor_shape(name)  # Get the shape of the tensor
            size = trt.volume(shape)  # Calculate the total number of elements in the tensor
            try:
                # Allocate host memory for the tensor
                host_mem = cuda.pagelocked_empty(size, dtype)
                # Allocate device memory for the tensor
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            except pycuda._driver.MemoryError:
                raise  # Raise an error if memory allocation fails
            # Append the host and device memory buffers to the list
            self.buffers.append((host_mem, device_mem))

    def infer(self, input_data):
        """
        Perform inference using the TensorRT engine.
        
        Args:
            input_data (np.ndarray): Input data for the inference.
        
        Returns:
            list: List of output data from the inference.
        """
        input_name = self.engine.get_tensor_name(0)  # Get the name of the first input tensor
        input_shape = self.engine.get_tensor_shape(input_name)  # Get the shape of the first input tensor
        input_data = np.resize(input_data, input_shape)  # Resize the input data to match the input shape

        # Set the input shape for the execution context
        self.context.set_input_shape(input_name, input_shape)

        # Copy the input data to the host memory buffer
        np.copyto(self.buffers[0][0], input_data.ravel())
        # Copy the input data from host to device asynchronously
        cuda.memcpy_htod_async(self.buffers[0][1], self.buffers[0][0], self.stream)

        # Execute the inference asynchronously
        self.context.execute_async_v2(bindings=[int(inp[1]) for inp in self.buffers], stream_handle=self.stream.handle)

        # Copy the output data from device to host asynchronously
        for out in self.buffers[1:]:
            cuda.memcpy_dtoh_async(out[0], out[1], self.stream)
        # Synchronize the stream to ensure all operations are complete
        self.stream.synchronize()

        # Return the output data as a list
        return [out[0] for out in self.buffers[1:]]
