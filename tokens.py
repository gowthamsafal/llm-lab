import tiktoken

enc = tiktoken.get_encoding("o200k_base")   # load a tokenizer (its learned BPE vocabulary)

texts = [
    "Hello world",
    "hello world",
    "Tokenization is unbelievably interesting!",
    "antidisestablishmentarianism",
    "2 + 2 = 4, and 12345678 is a number",
    "నమస్కారం, ఎలా ఉన్నారు?",
    "def add(a, b):\n    return a + b",
    "Gowtham",
    " Gowtham",
    "aaaaaaaaaaaaaaaaaaaa",
    "I love LLMs",
]

for text in texts:
    ids = enc.encode(text)                      # text → list of token IDs (what the model sees)
    pieces = [enc.decode([i]) for i in ids]     # each ID back to its text piece, so you can see the splits
    print(f"{len(ids):>3} tokens | {pieces}")   # print the count and the pieces
