import re  # Import regular expressions module for text manipulation
from nltk.tokenize import word_tokenize  # Import word_tokenize from nltk for tokenizing text
from nltk.corpus import stopwords  # Import stopwords from nltk to filter out common words
from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer from nltk for lemmatizing words
import numpy as np  # Import numpy for numerical operations

# Set of English stop words
stop_words = set(stopwords.words('english'))
# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def truncate_text(text, max_tokens=6000):
    """
    Truncate the text to a maximum number of tokens.
    
    Args:
        text (str): The input text to be truncated.
        max_tokens (int): The maximum number of tokens to retain.

    Returns:
        str: The truncated text.
    """
    # Ensure text is a string
    if not isinstance(text, str):
        try:
            text = str(text)  # Convert to string if not already
        except Exception as e:
            return f"Error converting text to string: {str(e)}"
    
    tokens = text.split()  # Split text into tokens
    if len(tokens) > max_tokens:
        # If token count exceeds max_tokens, truncate and add '[truncated]'
        return ' '.join(tokens[:max_tokens]) + ' [truncated]'
    return text

def preprocess_input(text):
    """
    Preprocess the input text by converting to lowercase, removing non-alphabetic characters,
    tokenizing, lemmatizing, removing stop words, and converting to a numpy array of ASCII values.
    
    Args:
        text (str): The input text to be processed.

    Returns:
        np.ndarray: The processed text as an array of ASCII values.
    """
    # Ensure text is a string
    if not isinstance(text, str):
        try:
            text = str(text)  # Convert to string if not already
        except Exception as e:
            return np.array([], dtype=np.float32)  # Return empty array on error
    
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text)  # Tokenize text into words
    # Lemmatize tokens and remove stop words
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    processed_text = ' '.join(processed_tokens)  # Join tokens back into a string
    # Convert characters to their ASCII values and return as a numpy array
    return np.array([ord(c) for c in processed_text], dtype=np.float32)

def postprocess_output(output):
    """
    Convert a numpy array of ASCII values back to a string.
    
    Args:
        output (np.ndarray): The input array of ASCII values.

    Returns:
        str: The output string.
    """
    # Convert ASCII values back to characters and join into a string
    return ''.join([chr(int(i)) for i in output])
