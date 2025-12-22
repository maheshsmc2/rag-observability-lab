from transformers import AutoTokenizer

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def tokenize(text):
    return tokenizer.encode(text, add_special_tokens=False)

def detokenize(token_ids):
    return tokenizer.decode(token_ids)

def sliding_window_chunk(text, max_tokens=250, overlap=80):
    token_ids = tokenize(text)
    chunks = []

    start = 0
    end = max_tokens

    while start < len(token_ids):
        window = token_ids[start:end]
        chunk_text = detokenize(window).strip()
        chunks.append(chunk_text)

        start = start + (max_tokens - overlap)
        end = start + max_tokens

    return chunks
