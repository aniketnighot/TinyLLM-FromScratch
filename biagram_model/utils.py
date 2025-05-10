def get_vocab(text):
    vocab = sorted(set(text))
    vocab_dict = {ch: i for i, ch in enumerate(vocab)}
    return vocab, vocab_dict

def get_numerical_tokens(vocab_dict, text):
    return [vocab_dict[c] for c in text]

def decode_tokens(vocab, tokens):
    return ''.join([vocab[i] for i in tokens])
