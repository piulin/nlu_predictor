import unicodedata
import re

import time
import math
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def tokenize(text: str):
    """
    Tokenize a text.
    :param text: Text string.
    :return: tokens of the text
    """

    # space_splits = re.findall(r"[\w']+", text)

    return word_tokenize(text)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"[0-9]+", r"NUM", s)
    #s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    return s

def detokenize(s):
    return TreebankWordDetokenizer().detokenize(s)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
