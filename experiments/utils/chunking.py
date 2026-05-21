from transformers import AutoTokenizer


def truncate_context(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> tuple[str, int]:
    if not text:
        return "", 0
    tokens = tokenizer.encode(text)
    total_retrieved = len(tokens)
    if total_retrieved > max_tokens:
        return tokenizer.decode(tokens[:max_tokens]), max_tokens
    return text, total_retrieved