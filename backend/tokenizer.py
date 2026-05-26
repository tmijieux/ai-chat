"""
Simple Python script to tokenize text using Qwen3.5 tokenizer via AutoTokenizer.
"""

from transformers import AutoTokenizer


def load_qwen35_tokenizer(model_name_or_path: str = "Qwen/Qwen3.5") -> AutoTokenizer:
    """
    Load the Qwen3.5 tokenizer using AutoTokenizer.

    Args:
        model_name_or_path: The model name or path to load the tokenizer from.
                           Default: "Qwen/Qwen3.5"

    Returns:
        AutoTokenizer: The loaded tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def tokenize_text(text: str, tokenizer:AutoTokenizer|None=None):
    """
    Tokenize input text using the Qwen3.5 tokenizer.

    Args:
        text: The input text to tokenize.
        tokenizer: Optional tokenizer instance. If None, loads the default Qwen3.5 tokenizer.

    Returns:
        dict: Tokenized output containing:
            - input_ids: List of token IDs
            - attention_mask: List of attention mask values
            - special_tokens_mask: List of special token mask values
    """
    if tokenizer is None:
        tokenizer = load_qwen35_tokenizer()

    # Tokenize the text
    tokenized = tokenizer(
        text,
        return_tensors="pt",  # Returns PyTorch tensors
        truncation=True,      # Truncate if text is too long
        padding=True          # Pad to max length in batch
    )

    return tokenized


def tokenize_text_batch(texts: list, tokenizer=None, max_length: int = 512):
    """
    Tokenize a batch of texts using the Qwen3.5 tokenizer.

    Args:
        texts: List of input texts to tokenize.
        tokenizer: Optional tokenizer instance. If None, loads the default Qwen3.5 tokenizer.
        max_length: Maximum sequence length for each token.

    Returns:
        dict: Tokenized output containing:
            - input_ids: Tensor of token IDs
            - attention_mask: Tensor of attention mask values
            - special_tokens_mask: Tensor of special token mask values
    """
    if tokenizer is None:
        tokenizer = load_qwen35_tokenizer()

    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )

    return tokenized


def get_tokenizer_info(tokenizer: AutoTokenizer|None=None):
    """
    Get information about the loaded tokenizer.

    Args:
        tokenizer: Optional tokenizer instance. If None, loads the default Qwen3.5 tokenizer.

    Returns:
        dict: Tokenizer information including:
            - vocab_size: Vocabulary size
            - model_max_length: Maximum sequence length
            - pad_token: Padding token
            - eos_token: End-of-sequence token
            - bos_token: Beginning-of-sequence token
    """
    if tokenizer is None:
        tokenizer = load_qwen35_tokenizer()

    return {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token
    }


def main():
    # Example usage
    print("Loading Qwen3.5 tokenizer...")


    model_file = "C:\\Users\\tmijieux\\.ollama\\models\\blobs\\sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a"
    tokenizer = load_qwen35_tokenizer(model_file)

    print("\nTokenizer Info:")
    info = get_tokenizer_info(tokenizer)
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Example 1: Tokenize a single text
    print("\n" + "="*50)
    print("Example 1: Tokenizing a single text")
    text = "Hello, how are you today?"
    tokenized = tokenize_text(text, tokenizer)
    print(f"Input text: {text}")
    print(f"Input IDs shape: {tokenized['input_ids'].shape}")
    print(f"Attention mask shape: {tokenized['attention_mask'].shape}")

    # Example 2: Tokenize a batch of texts
    print("\n" + "="*50)
    print("Example 2: Tokenizing a batch of texts")
    texts = [
        "Hello, how are you?",
        "I am doing well, thank you!",
        "This is a test of the tokenizer.",
        " text with <think> in between ",
        " text with <|im_start|> in between ",
        " text with <|im_end|> in between ",
    ]
    batch_tokenized = tokenize_text_batch(texts, tokenizer)
    print(f"Input texts count: {len(texts)}")
    print(f"Input IDs shape: {batch_tokenized['input_ids'].shape}")
    print(f"Attention mask shape: {batch_tokenized['attention_mask'].shape}")

    # Example 3: Show tokenized output
    print("\n" + "="*50)
    print("Example 3: Tokenized output for first text")
    print(f"Input IDs: {batch_tokenized['input_ids'][0].tolist()}")
    print(f"Attention Mask: {batch_tokenized['attention_mask'][0].tolist()}")

    print("\nDone!")

if __name__ == "__main__":
    main()
