import sys  # Provides access to system-specific parameters and functions
import os  # Provides a way of using operating system dependent functionality
import nltk  # Natural Language Toolkit, used for text processing tasks
from tensorrt_model import TensorRTModel  # Importing TensorRTModel class from tensorrt_model module
from llm_utils import generate_response, generate_code_with_mistral  # Importing functions from llm_utils module
from image_utils import generate_image_with_sdxl_turbo, process_image_with_neva  # Importing functions from image_utils module
from video_utils import process_video_generation  # Importing function from video_utils module
from file_utils import process_file  # Importing function from file_utils module
from text_utils import truncate_text, preprocess_input, postprocess_output  # Importing functions from text_utils module

# Download necessary NLTK data
nltk.download('punkt', quiet=True)  # Downloading punkt dataset for tokenization, suppressing verbose output
nltk.download('stopwords', quiet=True)  # Downloading stopwords dataset, suppressing verbose output
nltk.download('wordnet', quiet=True)  # Downloading wordnet dataset for lemmatization, suppressing verbose output

def process_input(user_input, conversation_history, trt_model=None):
    """
    Processes the user input and generates a response based on the input and the conversation history.
    """
    try:
        # Process 'code' command
        if user_input.lower().startswith('code '):  # Check if the input starts with 'code '
            code_prompt = user_input[5:]  # Extract the code prompt by removing 'code ' prefix
            response = generate_code_with_mistral(code_prompt)  # Generate code using Mistral
            if response:  # If response is successfully generated
                conversation_history.extend([
                    {"role": "user", "content": f"Generate code: {code_prompt}"},  # Add user input to conversation history
                    {"role": "assistant", "content": response}  # Add assistant response to conversation history
                ])
            else:
                response = "Failed to generate code."

        # Process 'image' command
        elif user_input.lower().startswith('image '):  # Check if the input starts with 'image '
            image_prompt = user_input[6:]  # Extract the image prompt by removing 'image ' prefix
            response = generate_image_with_sdxl_turbo(image_prompt)  # Generate image using SDXL-Turbo
            conversation_history.extend([
                {"role": "user", "content": f"Generate image: {image_prompt}"},  # Add user input to conversation history
                {"role": "assistant", "content": response}  # Add assistant response to conversation history
            ])

        # Process 'video' command
        elif user_input.lower().startswith('video '):  # Check if the input starts with 'video '
            video_prompt = user_input[6:]  # Extract the video prompt by removing 'video ' prefix
            response = process_video_generation(video_prompt)  # Generate video using Stable Video Diffusion
            conversation_history.extend([
                {"role": "user", "content": f"Generate video: {video_prompt}"},  # Add user input to conversation history
                {"role": "assistant", "content": response}  # Add assistant response to conversation history
            ])

        # Process with TensorRT model if available
        elif trt_model:  # If TensorRT model is available
            preprocessed_input = preprocess_input(user_input)  # Preprocess the user input
            input_name = trt_model.engine.get_tensor_name(0)  # Get the name of the first input tensor
            input_shape = trt_model.engine.get_tensor_shape(input_name)  # Get the shape of the first input tensor
            
            # Adjust the input shape if necessary
            if len(preprocessed_input) < input_shape[1]:
                preprocessed_input = np.pad(preprocessed_input, (0, input_shape[1] - len(preprocessed_input)))  # Pad the input
            else:
                preprocessed_input = preprocessed_input[:input_shape[1]]  # Trim the input

            preprocessed_input = preprocessed_input.reshape(input_shape)  # Reshape the input to match the expected input shape
            trt_output = trt_model.infer(preprocessed_input)  # Perform inference using TensorRT model
            processed_output = postprocess_output(trt_output[0])  # Postprocess the output from the model
            conversation_history.append({
                "role": "user", 
                "content": f"Based on this processed input: {processed_output}, respond to the original query: {user_input}"
            })  # Add processed input to conversation history
            response = generate_response(conversation_history)  # Generate response based on conversation history

        # Process general input
        else:
            conversation_history.append({"role": "user", "content": user_input})  # Add user input to conversation history
            response = generate_response(conversation_history)  # Generate response based on conversation history

        # Append assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": response or "I apologize, but I couldn't generate a response. Please try again."})
        
        return response

    except Exception as e:  # Catch any exception that occurs
        error_message = f"\nAn error occurred in process_input: {str(e)}"
        print(error_message)  # Print the error message
        return error_message  # Return the error message

def main():
    """
    Main function to run the conversation interface.
    """
    try:
        trt_model = TensorRTModel('./simple_tokenizer.trt')  # Initialize TensorRT model with the given path
        trt_model.load_engine()  # Load the TensorRT engine
        trt_model.allocate_buffers()  # Allocate buffers for input and output
    except Exception as e:  # Catch any exception that occurs
        print(f"\nFalling back to CPU processing. Reason: {e}")  # Print fallback message with the exception reason
        trt_model = None  # Set TensorRT model to None for CPU processing

    # Print welcome messages and instructions
    print("\nWelcome to the context-aware LLaMA conversation interface with Neva-22 image processing, Mistral code generation, and SDXL-Turbo image generation.\n")
    print("- Type 'exit' to end the conversation.")
    print("- To upload a file, type 'upload <file_path>'.")
    print("- To generate code using Mistral, type 'code <your_prompt>'.")
    print("- To generate an image using SDXL-Turbo, type 'image <your_prompt>'.")
    print("- To generate a video using Stable Video Diffusion, type 'video <your_prompt>'.\n")
    
    # Initialize conversation history
    conversation_history = [{"role": "system", "content": "You are a helpful AI assistant. Maintain context throughout the conversation and provide coherent responses based on the ongoing dialogue."}]
    
    while True:
        user_input = input(">> ")  # Read user input from the console
        if user_input.lower() == 'exit':  # If the user types 'exit'
            print("\nCome back any time!\n")
            break  # Exit the loop and end the program
        elif user_input.lower().startswith('upload '):  # If the user types 'upload '
            file_path = user_input[7:].strip()  # Extract the file path by removing 'upload ' prefix
            try:
                file_content = process_file(file_path)  # Process the uploaded file
                print(f"File '{file_path}' processed successfully.")
                truncated_content = truncate_text(file_content)  # Truncate the file content if it's too long
                conversation_history.append({"role": "user", "content": f"I've uploaded a file with the following content:\n\n{truncated_content}\n\nPlease analyze this content and provide insights or answer any questions I might have about it."})  # Add file content to conversation history
                process_input("Please acknowledge that you've received and processed the file content, and let me know you're ready for questions about it.", conversation_history, trt_model)  # Acknowledge file processing
            except Exception as e:  # Catch any exception that occurs
                print(f"Error processing file: {str(e)}")
        else:
            process_input(user_input, conversation_history, trt_model)  # Process user input and generate response

if __name__ == "__main__":
    main()  # Run the main function
