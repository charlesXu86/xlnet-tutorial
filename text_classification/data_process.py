# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   data_process.py
 
@Time    :   2019-08-28 10:16
 
@Desc    :
 
'''

import tensorflow  as tf
import sys
import six
import unicodedata
import sentencepiece as spm
import collections

flags = tf.flags
FLAGS = flags.FLAGS

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

VOCAB_SIZE = 32000
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs

def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    # return_unicode is used only for py2
    # note(zhiliny): in some systems, sentencepiece only accepts str for py2
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    return new_pieces


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_class_ids(text, max_seq_length, tokenize_fn):
    texts = tokenize_fn(text)
    if len(texts) > max_seq_length - 2:
        texts = texts[:max_seq_length - 2]
    tokens = []
    segment_ids = []
    for token in texts:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)

    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)

    input_ids = tokens
    input_mask = [0] * len(input_ids)
    if len(input_ids) < max_seq_length:
        delta_len = max_seq_length - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def get_pair_ids(text_a, text_b, max_seq_length, tokenize_fn):
    tokens_a = tokenize_fn(text_a)
    tokens_b = tokenize_fn(text_b)
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = []
    segment_ids = []
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(SEG_ID_B)
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_B)

    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)

    input_ids = tokens
    input_mask = [0] * len(input_ids)

    if len(input_ids) < max_seq_length:
        delta_len = max_seq_length - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


SPIECE_UNDERLINE = '▁'



def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


def tokenize_fn(text):
    text = preprocess_text(text, lower=True)
    return encode_ids(sp, text)


def get_vocab(path):
    maps = collections.defaultdict()
    i = 0
    with tf.gfile.GFile(path, "r") as  f:
        for line in f.readlines():
            maps[line.strip()] = i
            i = i + 1
    f.close()
    return maps


def writedataclass(inputpath, vocab, outputpath, max_seq_length, tokenize_fn):
    eachonum = 5000
    num = 0
    recordfilenum = 0
    ftrecordfilename = ("xlnetreading.tfrecords-%.3d" % recordfilenum)
    writer = tf.python_io.TFRecordWriter(outputpath + ftrecordfilename)
    with  open(inputpath)  as f:
        for text in f.readlines():
            texts = text.split("\t")
            content = texts[0].lower().strip()
            label = vocab.get(texts[1].strip())
            num = num + 1
            input_ids, input_mask, segment_ids = get_class_ids(content, max_seq_length, tokenize_fn)
            if num > eachonum:
                num = 1
                recordfilenum = recordfilenum + 1
                ftrecordfilename = ("xlnetreading.tfrecords-%.3d" % recordfilenum)
                writer = tf.python_io.TFRecordWriter(outputpath + ftrecordfilename)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
                             'input_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=input_mask)),
                             'segment_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
                             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                             }))
            serialized = example.SerializeToString()
            writer.write(serialized)
    writer.close()
    f.close()


