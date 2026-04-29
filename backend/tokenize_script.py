
from transformers import AutoTokenizer
import json

FILE_PATH = "./.claude.json"

try:
    # 1. Read the entire file content (using simple open, bypassing Read tool limits)
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # 2. Initialize the tokenizer (assuming a standard Gemma 4 model structure)
    # Note: In a real environment, the specific model path/name must be used.
    # Using a placeholder model name that should work with the library.
    model_file="C:\\Users\\tmijieux\\.ollama\\models\\blobs\\sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a"
    tokenizer = AutoTokenizer.from_pretrained(model_file)

    # 3. Encode the content and count tokens
    # Using encode_plus to handle the string input
    encoding = tokenizer(file_content)
    token_count = len(encoding['input_ids'])

    print(f"Successfully tokenized the file content.")
    print(f"Total number of tokens in {FILE_PATH}: {token_count}")

except FileNotFoundError:
    print(f"Error: The file {FILE_PATH} was not found.")
except Exception as e:
    print(f"An error occurred during tokenization: {e}")
