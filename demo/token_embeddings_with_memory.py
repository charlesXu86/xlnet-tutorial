# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   token_embeddings_with_memory.py
 
@Time    :   2019-08-28 18:00
 
@Desc    :
 
'''

import os
import sys

import numpy as np

from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import load_trained_model_from_checkpoint
from keras_xlnet import ATTENTION_TYPE_UNI

from xlnet.tokenization_xlnet import XLNetTokenizer



os.environ["CUDA_VISIBLE_DEVICES"] = "3"
checkpoint_path = "/home/xsq/model/chinese_xlnet_mid_L-24_H-768_A-12"
vocab_path = os.path.join(checkpoint_path, 'spiece.model')
config_path = os.path.join(checkpoint_path, 'xlnet_config.json')
model_path = os.path.join(checkpoint_path, 'xlnet_model.ckpt')


# Tokenize inputs
tokenizer = XLNetTokenizer(vocab_path)
text = "第二次买了，清洗得干净且脸没有紧绷感"
text2 = "感觉很棒"

token = tokenizer.tokenize(text2)
tokens = tokenizer.encode(text)

target_len = 3
total_len = len(tokens)

# Load pre-trained model
model = load_trained_model_from_checkpoint(
    config_path=config_path,
    checkpoint_path=model_path,
    batch_size=1,
    memory_len=total_len,
    target_len=target_len,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_UNI,
)

# Predict
for mem_len in range(0, total_len, target_len):
    index = mem_len // target_len
    sub_tokens = tokens[index * target_len:(index + 1) * target_len]
    token_input = np.expand_dims(np.array(sub_tokens + [Tokenizer.SYM_PAD] * (target_len - len(sub_tokens))), axis=0)
    segment_input = np.zeros_like(token_input)
    memory_length_input = np.array([[mem_len]])
    results = model.predict_on_batch([token_input, segment_input, memory_length_input])
    for i in range(3):
        if index * 3 + i < 7:
            print(results[0, i, :5])
    print('')
