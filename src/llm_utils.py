import requests  # For making HTTP requests
import json  # For handling JSON data
from config import nvidia_api_key, llama_api_url, mistral_api_url  # Importing configurations from config file
from text_utils import truncate_text  # Importing utility function for truncating text

def generate_response(messages, max_tokens=1024, temperature=0.5):
    """
    Generates a response using the LLaMA API based on the given messages.

    Args:
        messages (list): A list of message dictionaries containing the conversation history.
        max_tokens (int): The maximum number of tokens in the response. Default is 1024.
        temperature (float): Sampling temperature for the response generation. Default is 0.5.

    Returns:
        str: The generated response.
    """
    try:
        # Set up headers for the API request
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",  # API key for authentication
            "Content-Type": "application/json"
        }
        
        # Truncate messages if they are too long
        truncated_messages = []
        for message in messages:
            truncated_content = truncate_text(message['content'])  # Truncate the content of the message
            truncated_messages.append({"role": message['role'], "content": truncated_content})  # Append the truncated message
        
        # Set up the payload with the model parameters and truncated messages
        data = {
            "model": "meta/llama3-70b-instruct",
            "messages": truncated_messages,
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stream": True
        }

        # Make the API request to generate the response
        response = requests.post(llama_api_url, headers=headers, json=data, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        full_response = ""
        # Process the streamed response line by line
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')  # Decode the line to a string
                if line_text.startswith('data: '):  # Check if the line starts with 'data: '
                    if line_text.strip() == 'data: [DONE]':  # Check if the line indicates the end of the response
                        break
                    try:
                        json_response = json.loads(line_text[6:])  # Parse the JSON response
                        if 'choices' in json_response and len(json_response['choices']) > 0:
                            delta = json_response['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                print(content, end="", flush=True)  # Print the content to the console
                                full_response += content  # Append the content to the full response
                    except json.JSONDecodeError:
                        continue

        print()  # Add a newline after the response
        return full_response  # Return the full response
    except requests.RequestException as e:
        print(f"\nAPI request error: {str(e)}")
        if e.response is not None:
            print(f"Response content: {e.response.content}")
        return None
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return None

def generate_code_with_mistral(prompt, max_tokens=1024, temperature=0.5):
    """
    Generates code using the Mistral API based on the given prompt.

    Args:
        prompt (str): The text prompt for generating the code.
        max_tokens (int): The maximum number of tokens in the response. Default is 1024.
        temperature (float): Sampling temperature for the response generation. Default is 0.5.

    Returns:
        str: The generated code.
    """
    try:
        # Set up headers for the API request
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",  # API key for authentication
            "Content-Type": "application/json"
        }
        
        # Set up the payload with the model parameters and prompt
        data = {
            "model": "mistralai/mistral-large",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stream": True
        }

        # Make the API request to generate the code
        response = requests.post(mistral_api_url, headers=headers, json=data, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        full_response = ""
        # Process the streamed response line by line
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')  # Decode the line to a string
                if line_text.startswith('data: '):  # Check if the line starts with 'data: '
                    if line_text.strip() == 'data: [DONE]':  # Check if the line indicates the end of the response
                        break
                    try:
                        json_response = json.loads(line_text[6:])  # Parse the JSON response
                        if 'choices' in json_response and len(json_response['choices']) > 0:
                            delta = json_response['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                print(content, end="", flush=True)  # Print the content to the console
                                full_response += content  # Append the content to the full response
                    except json.JSONDecodeError:
                        continue

        print()  # Add a newline after the response
        return full_response  # Return the full response
    except requests.RequestException as e:
        print(f"\nAPI request error: {str(e)}")
        if e.response is not None:
            print(f"Response content: {e.response.content}")
        return None
    except Exception as e:
        print(f"\nAn error occurred while generating code with Mistral: {str(e)}")
        return None
