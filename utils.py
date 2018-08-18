import re
import pickle


def build_vocab(text):
    """
    Builds vocabulary from the given text
    :param text: string from which vocabulary has to be built
    :return: dict word2idx which contains unique words mapped to unique index values
    and idx2word which has the reverse mapping
    """
    text = text.lower().strip()
    spcl_token_dict = token_lookup()
    for key, token in spcl_token_dict.items():
        text = text.replace(key, ' {} '.format(token)).strip()
    while '  ' in text:
        text = text.replace('  ', ' ')
    pat_alphabetic = re.compile('(([\d\w\<\>\_\:])+)')
    tokens = [match.group() for match in pat_alphabetic.finditer(text)]
    vocab = {token: True for token in tokens}
    vocab = list(vocab)
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {value: key for key, value in word2idx.items()}
    return tokens, word2idx, idx2word


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    return {';': "<semicolon>",
            "'": "<inverted_comma>",
            '"': "<quotation_mark>",
            ',': "<comma>",
            '\n': "<new_line>",
            '!': "<exclamation_mark>",
            '-': "<hyphen>",
            '--': "<hyphens>",
            '.': "<period>",
            '?': "<question_mark>",
            '(': "<left_paren>",
            ')': "<right_paren>",
            '♪': "<music_note>",
            '[': "<left_square>",
            ']': "<right_square>",
            }


def preprocess(data_path):
    """
    Loads data and preprocesses it to produce vocabulary, converts
    text to integer sequences and saves everything as pickled file
    :param data_path: path to text file containing data
    :return: text sequences converted to integer sequence
    """
    with open(data_path, "r") as in_file:
        text = in_file.readlines()
    for t in text:
        print(t)
    # tokens, word2idx, idx2word = build_vocab(text)
    # text_seq = [word2idx[word] for word in tokens]
    # data = {'word2idx': word2idx, 'idx2word': idx2word}
    # pickle.dump(data, open('preprocessed_data.pkl', 'wb'))
    # print("\nWordMap successfully stored as preprocessed_data.pkl")
    # return text_seq


def load_data(data_path):
    """
    Loads pickled WordMap data
    :param data_path: path to pickled wordmap file
    :return: word2idx dict and idx2word dict
    """
    with open(data_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data['word2idx'], data['idx2word']
