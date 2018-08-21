import re
import pickle
import numpy as np
from gensim.models import Word2Vec


def build_vocab(text):
    """
    Builds vocabulary from the given text
    :param text: string from which vocabulary has to be built
    :return: dict word2idx which contains unique words mapped to unique index values
    and idx2word which has the reverse mapping
    """
    text = clean_text(text)
    pat_alphabetic = re.compile('(([\d\w<>_:])+)')
    tokens = [match.group() for match in pat_alphabetic.finditer(text)]
    vocab = {token: True for token in tokens}
    vocab = list(vocab)
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {value: key for key, value in word2idx.items()}
    return tokens, word2idx, idx2word


def clean_text(text):
    """
    Cleans text data
    :param text: str object containing text data
    :return: cleaned text
    """
    text = text.lower().strip()
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token)).strip()
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text


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


def preprocess(text):
    """
    Preprocesses text data to produce vocabulary, converts
    text to integer sequences and saves everything as pickled file
    :param text: text data to build vocabulary
    :return: text sequences converted to integer sequence
    """
    tokens, word2idx, idx2word = build_vocab(text)
    text_seq = [word2idx[word] for word in tokens]
    return text_seq


def save_data(word2idx, idx2word, embeddings=None, save_as="./training_data.pkl"):
    """
    Saves wordmap and embeddings as a pickle file
    :param word2idx: dict containing word to integer mapping
    :param idx2word: dict containing integer to word mapping
    :param embeddings: numpy array containing word embeddings
    :param save_as: path to where the pickle file should be stored
    """
    data = {'word2idx': word2idx,
            'idx2word': idx2word,
            'embeddings': embeddings
            }
    pickle.dump(data, open(save_as, 'wb'))
    print("\nData successfully stored as {}\n".format(save_as))


def load_data(data_path):
    """
    Loads pickled WordMap data and embeddings
    :param data_path: path to pickled wordmap file
    :return: word2idx dict, idx2word dict and word2vec embeddings
    """
    with open(data_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data['word2idx'], data['idx2word'], data['embeddings']


def load_text(data_path):
    """
    Loads data from text file
    :param data_path: path to text file
    :return: text data from file
    """
    with open(data_path, "r") as in_file:
        text = in_file.read()
    return text


def word2vec(text, embed_size=300):
    """
    Trains Word2Vec model on given text data and returns word embeddings
    :param text: str object containing text data
    :param embed_size: number of dimensions in word vectors
    :return: trained word embeddings in numpy array
    """
    _, word2idx, _ = build_vocab(text)
    text = text.split('\n')
    pat_alphabetic = re.compile('(([\d\w<>_:])+)')
    sentences = [[match.group() for match in pat_alphabetic.finditer(clean_text(sentence))] +['<new_line>']
                 for sentence in text]
    model = Word2Vec(sentences, size=embed_size, window=5, min_count=1, workers=4)
    print(model)
    vocab = list(word2idx)
    embeddings = np.zeros([len(vocab), embed_size], dtype=np.float32)
    for i, word in enumerate(vocab):
        embeddings[i] = model.wv[word]
    return embeddings
