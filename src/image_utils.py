import requests  # For making HTTP requests
import json  # For handling JSON data
import base64  # For encoding and decoding base64
import io  # For handling byte streams
from PIL import Image  # For image processing
from datetime import datetime  # For date and time handling
import os  # For interacting with the operating system
from config import nvidia_api_key, sdxl_turbo_url, neva_api_url  # Importing configurations from config file

def generate_image_with_sdxl_turbo(prompt, save_image=True):
    """
    Generates an image using the SDXL Turbo API based on the given prompt.
    Optionally saves the generated image.

    Args:
        prompt (str): The text prompt for generating the image.
        save_image (bool): Whether to save the generated image. Default is True.

    Returns:
        tuple: File path of the saved image (if applicable) and base64 representation of the image.
    """
    try:
        # Set up headers for the API request
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",  # API key for authentication
            "Accept": "application/json",
        }

        # Set up the payload with the text prompt and additional parameters
        payload = {
            "text_prompts": [{"text": prompt}],
            "seed": 0,
            "sampler": "K_EULER_ANCESTRAL",
            "steps": 2
        }

        # Optionally print a message if saving the image
        if save_image:
            print('Sending request to SDXL Turbo API...')

        # Make the API request
        response = requests.post(sdxl_turbo_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        response_body = response.json()
        
        # Check if the response contains image data
        if 'artifacts' in response_body and len(response_body['artifacts']) > 0:
            image_base64 = response_body['artifacts'][0]['base64']  # Get the base64 image data
            image_data = base64.b64decode(image_base64)  # Decode the base64 image data
            
            # Save the image as PNG if required
            if save_image:
                image = Image.open(io.BytesIO(image_data))  # Open the image from byte data
                
                # Define the directory path
                directory_path = '/home/paperspace/Projects/NvidiaTesnorRTLLM/image_generation'
                os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist

                # Generate a timestamp for the filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prompt.replace(' ', '_')}_{timestamp}.png"
                filepath = os.path.join(directory_path, filename)

                # Save the image to the specified file path
                try:
                    image.save(filepath, format='PNG')
                    print(f"Image generated and saved as {filepath}")
                except IOError as e:
                    print(f"IOError while saving image: {str(e)}")
                    return f"Error saving image: {str(e)}", None
                
                return filepath, image_base64  # Return the file path and base64 image data
            else:
                return None, image_base64  # Return only the base64 image data
        else:
            print("Unexpected response format from SDXL-Turbo API")
            print(f"Response body: {response_body}")
            return "Error: Unexpected response format from SDXL-Turbo API", None
    except requests.RequestException as e:
        print(f"API request error: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Error response content: {e.response.text}")
        return f"API request error: {str(e)}", None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return f"An error occurred while generating image with SDXL-Turbo: {str(e)}", None

def process_image_with_neva(image_path):
    """
    Processes an image using the Neva-22 API for detailed analysis.

    Args:
        image_path (str): The file path of the image to be processed.

    Returns:
        str: Analysis result from Neva-22 API.
    """
    try:
        # Get basic image information
        with Image.open(image_path) as img:
            width, height = img.size
            format = img.format
            mode = img.mode
        image_info = f"Image Information:\nFormat: {format}\nSize: {width}x{height}\nMode: {mode}"

        # Encode the image to base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        # Set up headers for the API request
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",  # API key for authentication
            "Accept": "application/json"
        }
        
        # Set up the payload with the image and additional parameters
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'Analyze and describe this image in detail. Include any text you can read, objects, people, actions, colors, and overall scene description. <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.70,
            "seed": 0,
            "stream": False
        }
        
        # Make the API request
        response = requests.post(neva_api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            neva_description = result['choices'][0]['message']['content']
            return f"{image_info}\n\nNeva-22 Image Analysis:\n{neva_description}"
        else:
            return f"{image_info}\n\nError: Unable to process the image with Neva-22. API response did not contain expected data."
    except requests.RequestException as e:
        return f"Error processing image with Neva-22: API request failed. {str(e)}"
    except IOError as e:
        return f"Error processing image with Neva-22: Unable to read the image file. {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error processing image with Neva-22: Invalid JSON response from API. {str(e)}"
    except Exception as e:
        return f"Error processing image with Neva-22: An unexpected error occurred. {str(e)}"

def get_image_info(file_path):
    """
    Retrieves basic information about an image.

    Args:
        file_path (str): The file path of the image.

    Returns:
        str: Basic information about the image.
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            format = img.format
            mode = img.mode
            return f"Image Information:\nFormat: {format}\nSize: {width}x{height}\nMode: {mode}"
    except Exception as e:
        return f"Error getting image information: {str(e)}"
