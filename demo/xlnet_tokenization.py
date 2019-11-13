# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   xlnet_tokenization.py
 
@Time    :   2019-09-19 14:12
 
@Desc    :
 
'''

import sentencepiece as spm
from xlnet.prepro_utils import preprocess_text, encode_ids

from ner.NER_Config import Config

cf = Config()

text = '这是输入'

sp_model = spm.SentencePieceProcessor()
sp_model.load(cf.spiece_model_file)

text = preprocess_text(text, lower=True)
ids = encode_ids(sp_model, text)
print(ids)