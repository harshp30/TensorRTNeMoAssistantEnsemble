import requests  # For making HTTP requests to APIs
import json  # For parsing JSON responses
import base64  # For encoding and decoding base64 data
import io  # For handling I/O operations
from PIL import Image  # For image processing
import os  # For interacting with the operating system
from datetime import datetime  # For getting the current date and time
from config import nvidia_api_key, stable_video_diffusion_url  # Importing API keys and URLs from config
from image_utils import generate_image_with_sdxl_turbo  # Importing the function to generate images

def generate_video_with_stable_diffusion(image_base64):
    """
    Generate a video using Stable Video Diffusion from a base64 encoded image.
    
    Args:
        image_base64 (str): The base64 encoded image data.
    
    Returns:
        bytes: The video data as bytes, or None if an error occurred.
    """
    try:
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",  # Authentication header with the API key
            "Accept": "application/json",  # Accept JSON responses
        }
        payload = {
            "image": f"data:image/jpeg;base64,{image_base64}",  # Include the image in the payload
            "cfg_scale": 2.5,  # Configuration scale parameter
            "seed": 0  # Seed for random number generation
        }

        print('Sending request to Stable Video Diffusion API...')
        response = requests.post(stable_video_diffusion_url, headers=headers, json=payload)  # Make the API request
        
        response.raise_for_status()  # Raise an exception for non-200 status codes
        
        response_body = response.json()  # Parse the response JSON
        if 'video' in response_body:
            video_data = base64.b64decode(response_body['video'])  # Decode the base64 encoded video
            return video_data  # Return the video data
        else:
            print('No video data found in response')
            return None  # Return None if no video data is found
    except requests.RequestException as e:
        print(f"API request error: {str(e)}")  # Print the API request error
        if hasattr(e.response, 'text'):
            print(f"Error response content: {e.response.text}")  # Print the error response content if available
        return None  # Return None if an API request error occurred
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")  # Print any other unexpected error
        return None  # Return None if an unexpected error occurred

def process_video_generation(prompt):
    """
    Generate a video based on a prompt by first generating an image and then converting it to a video.
    
    Args:
        prompt (str): The text prompt to generate the video.
    
    Returns:
        str: The file path of the generated video, or an error message if the process failed.
    """
    # First, generate the image using the existing function
    image_path, image_base64 = generate_image_with_sdxl_turbo(prompt, save_image=False)
    if isinstance(image_path, str) and (image_path.startswith("Error") or image_path.startswith("An error occurred")):
        return f"Failed to generate initial image: {image_path}"  # Return error message if image generation failed
    
    if image_base64 is None:
        return "Failed to generate base64 encoded image data."  # Return error message if no image data is generated
    
    # Convert PNG to JPEG
    try:
        image_data = base64.b64decode(image_base64)  # Decode the base64 encoded image
        img = Image.open(io.BytesIO(image_data))  # Open the image
        rgb_img = img.convert('RGB')  # Convert the image to RGB format
        jpeg_buffer = io.BytesIO()  # Create an in-memory buffer for the JPEG
        rgb_img.save(jpeg_buffer, format='JPEG')  # Save the image as JPEG in the buffer
        jpeg_base64 = base64.b64encode(jpeg_buffer.getvalue()).decode('utf-8')  # Encode the buffer content to base64
    except Exception as e:
        return f"Failed to convert PNG to JPEG: {str(e)}"  # Return error message if conversion fails
    
    # Generate video using the JPEG image
    video_data = generate_video_with_stable_diffusion(jpeg_base64)  # Generate the video from the JPEG image
    if video_data is None:
        return "Failed to generate video from the initial image."  # Return error message if video generation fails
    
    # Save the video
    directory_path = '/home/paperspace/Projects/NvidiaTesnorRTLLM/video_generation'  # Define the directory path
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate a timestamp
    filename = f"{prompt.replace(' ', '_')}_{timestamp}.mp4"  # Create a filename with the prompt and timestamp
    filepath = os.path.join(directory_path, filename)  # Create the full file path
    
    with open(filepath, 'wb') as f:
        f.write(video_data)  # Write the video data to the file
        print(f"Video generated and saved as {filepath}")  # Print confirmation message
    
    return f"Video generated and saved as {filepath}"  # Return the file path of the generated video
