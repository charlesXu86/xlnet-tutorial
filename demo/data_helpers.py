# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   data_helpers.py
 
@Time    :   2019-11-17 11:21
 
@Desc    :
 
'''

import numpy as np

from xlnet_embedding import sentence2idx, idx2sentence, XlnetEmbedding
from domain_cls import Domain_cls_with_Xlnet

dcx = Domain_cls_with_Xlnet()

def encode_data(X, y=None):
    x = []

    for sample in X:
        if type(sample) == tuple and len(sample) == 2:
            encoded = sentence2idx(dcx.xlnet_hyper["len_max"], sample[0], sample[1])
        else:
            encoded = sentence2idx(dcx.xlnet_hyper["len_max"], sample)
        x.append(encoded)

    x_1 = np.array([i[0][0] for i in x])
    x_2 = np.array([i[1][0] for i in x])
    x_3 = np.array([i[2][0] for i in x])
    if dcx.xlnet_hyper["trainable"] == True:
        x_all = [x_1, x_2, x_3, np.zeros(np.shape(x_1))]
    else:
        x_all = [x_1, x_2, x_3]

    if y != None:
        _, y = dcx.get_label_list()
        onehot_label = []
        for (sample, label) in enumerate(y):
            onehot = [0] * dcx.model_hyper["label"]
            onehot[sample] = 1
            onehot_label.append(onehot)

        onehot_label = np.array(onehot_label)
        return x_all, onehot_label
    return x_all