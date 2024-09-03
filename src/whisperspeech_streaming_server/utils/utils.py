import re
import json
import numpy as np


def split_sentence(text, min_len=10):
    sentences = split_sentences_latin(text, min_len=min_len)
    return sentences


def split_sentences_latin(text, min_len=10):
    """Split Long sentences into list of short ones

    Args:
        text: Input sentences.

    Returns:
        List[str]: list of output sentences.
        :param text:
        :param min_len:
    """
    # deal with dirty sentences
    text = re.sub("[。！？；]", ".", text)
    text = re.sub("[，]", ",", text)
    text = re.sub("[“”]", '"', text)
    text = re.sub("[‘’]", "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    text = re.sub("[\n\t ]+", " ", text)
    text = re.sub("([,.!?;])", r"\1 $#!", text)
    # split
    sentences = [s.strip() for s in text.split("$#!")]
    if len(sentences[-1]) == 0:
        del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        # print(sent)
        new_sent.append(sent)
        count_len += len(sent.split(" "))
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(" ".join(new_sent))
            new_sent = []
    return merge_short_sentences_latin(new_sentences)


def merge_short_sentences_latin(sens):
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentence is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out
