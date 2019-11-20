# -*- coding: utf-8 -*-

'''
@Author  :   Xu

@Software:   PyCharm

@File    :   domain_cls.py.py

@Time    :   2019-09-19 14:12

@Desc    :

'''

import os
import pathlib
import keras
from xlnet_embedding import sentence2idx, idx2sentence, XlnetEmbedding
from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.models import *
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import codecs
import pickle as pkl
import tensorflow as tf

basedir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class f1_callback(keras.callbacks.Callback):

    def __init__(self, val_data):
        super(f1_callback, self).__init__()
        self.x_val = val_data[0]
        self.y_val = val_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_val, batch_size=model_hyper["batch_size"])
        y_pred = np.argmax(y_pred, axis=1)
        y_val = np.argmax(self.y_val, axis=1)
        result = classification_report(y_val, y_pred)

        print(result)
        return

class Domain_cls_with_Xlnet(object):

    def __init__(self):
        super(Domain_cls_with_Xlnet, self).__init__()
        self.max_seq_length = 30
        self.labels = []
        self.num_class = 6
        self.batch_size = 16
        self.epoches = 5
        self.patience = 3  # 早停
        self.lr = 5e-5
        self.saved_models_dir = basedir + '/saved_models'

        # xlnet参数配置
        self.xlnet_model = '/Data/public/XLNet/chinese_xlnet_base_L-12_H-768_A-12'
        self.path_fineture = basedir + '/saved_models/embedding_trainable.h5'
        self.layer_indexes = [-2]
        self.trainable = True
        self.attention_type = 'bi'  # ['bi', 'uni']
        self.memory_len = 0
        self.merge_type = 'add'   # 选择多层输出时处理多层输出的方式： ["add", "avg", "max", "concat"]
        self.sess = tf.Session(graph=tf.Graph())
        # with self.sess.graph.as_default():
        #     self.model = load_model(os.path.join(self.saved_models_dir, 'xlnet_domain_model.h5'))
        #     self.model = load_model(os.path.join(basedir + '/demo/model/embedding_trainable.h5'))

    # 自行修改需要的配置
    def get_config(self):
        global model_hyper, xlnet_hyper

        model_hyper = {
            'len_max': 30,  # 句子最大长度, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 小心OOM
            'label': 10,  # 类别数
            'batch_size': 16,
            'epochs': 5,  # 训练最大轮次
            'patience': 3,  # 早停,2-3就好
            'lr': 5e-5,  # 学习率
            'model_path': '/home/xsq/nlp_code/xlnet-tutorial/demo/model/model.h5',  # 模型保存地址
        }

        xlnet_hyper = {
            # 下载的参数路径
            'model_path': "/Data/public/XLNet/chinese_xlnet_base_L-12_H-768_A-12",
            # 微调后保存地址
            'path_fineture': "./model/embedding_trainable.h5",
            # 选择输出的层数 范围 [0, 12(24)], 12或24取决于用的是base还是mid, -1即最后一层 12/24
            'layer_indexes': [-2],
            'len_max': model_hyper["len_max"],
            'batch_size': model_hyper["batch_size"],
            # 是否微调embedding
            'trainable': True,
            # ['bi', 'uni']
            'attention_type': 'bi',
            'memory_len': 0,
            # 选择多层输出时处理多层输出的方式： ["add", "avg", "max", "concat"]
            'merge_type': "add"
        }

    # def load_dict(self):

    def get_labels(self, labels):
        return sorted(set(labels), key=labels.index)  # 使用有序列表而不是集合。保证了标签正确

    def get_label_list(self):
        '''
        读取模型训练是动态产生的label_list.pkl文件
        :return:
        '''
        label_list = pkl.load(open(basedir + '/demo/label_list.pkl', 'rb'))
        label_id = pkl.load(open(basedir + '/demo/label2id.pkl', 'rb'))
        print(label_list, label_id)
        return label_list, label_id


    def process_datas(self, filename, mode='train'):
        '''
        数据预处理
        1、训练数据格式为：  体育\t文本
        :param filename:
        :param mode:
        :return:
        '''
        def process_train():
            with open(filename, 'r', encoding='utf-8') as fin:
                row_data = fin.readlines()
                np.random.shuffle(row_data)
                X, y = [], []
                label_map = {}
                for i in range(len(row_data)):
                    line = row_data[i].strip().split('\t')
                    if len(line) < 2:
                        continue
                    X.append(line[1])
                    y.append(line[0])

                # {v: k for k, v in m.items()}
                label_list = self.get_labels(y)
                label2id = dict((label, index) for index, label in enumerate(label_list))
                y = self.char2id(label2id, y)
                for (i, label) in enumerate(label_list):
                    label_map[label] = i
                with open('label_list.pkl', 'wb') as f:
                    pkl.dump(y, f)
                with open('label2id.pkl', 'wb') as f:
                    pkl.dump(label2id, f)
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
                length = model_hyper["batch_size"]
                print(np.shape(X_train))
                print(np.shape(X_val))
                train_length = (len(y_train) // length) * length
                val_length = (len(y_val) // length) * length
                print(train_length, val_length)
                return X_train[:train_length], X_val[:val_length], y_train[:train_length], y_val[:val_length]

        if mode == "train":
            return process_train()
        # else:
        #     return process_test()

    def char2id(self, label, sting):
        return [label.get(c, 1) for c in sting]

    def process_data(self, filename, mode="train"):
        '''
        数据预处理
        1、训练数据格式为：  体育\t文本
        :param filename:
        :param mode:
        :return:
        '''
        def process_train():
            nonlocal filename
            f = codecs.open(filename, "r", "UTF-8")
            X, y = [], []
            for line in f:
                line = line.strip().split("\t")
                if len(line) < 2:
                    continue
                # 双输入要以tuple形式保存   即  X.append( ("text1", "text2") )
                # X.append((line[0].replace(" ", "")[:len(line[0].replace(" ", ""))//2], line[0].replace(" ", "")[len(line[0].replace(" ", ""))//2:]))
                X.append(line[0].replace(" ", ""))
                y.append(int(line[1]))
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
            length = model_hyper["batch_size"]
            print(np.shape(X_train))
            print(np.shape(X_val))
            train_length = (len(y_train) // length) * length
            val_length = (len(y_val) // length) * length
            print(train_length, val_length)
            return X_train[:train_length], X_val[:val_length], y_train[:train_length], y_val[:val_length]

        def process_test():
            nonlocal filename
            f = codecs.open(filename, "r", "UTF-8")
            X, y = [], []
            for line in f:
                line = line.strip().split("\t")
                if len(line) < 2:
                    continue
                X.append(line[0].replace(" ", ""))
                # X.append((line[0].replace(" ", "")[:len(line[0].replace(" ", ""))//2], line[0].replace(" ", "")[len(line[0].replace(" ", ""))//2:]))
                y.append(int(line[1]))
            print(np.shape(X))
            return X, y

        if mode == "train":
            return process_train()
        else:
            return process_test()

    def process_predict(self, msg):
        X = []
        X.append(msg.replace(" ", ""))
        # X.append((line.replace(" ", "")[:len(line.replace(" ", ""))//2], line.replace(" ", "")[len(line.replace(" ", ""))//2:]))
        return X


    def encode_data(self, X, y=None):
        x = []

        for sample in X:
            if type(sample) == tuple and len(sample) == 2:
                encoded = sentence2idx(xlnet_hyper["len_max"], sample[0], sample[1])
            else:
                encoded = sentence2idx(xlnet_hyper["len_max"], sample)
            x.append(encoded)

        x_1 = np.array([i[0][0] for i in x])
        x_2 = np.array([i[1][0] for i in x])
        x_3 = np.array([i[2][0] for i in x])
        if xlnet_hyper["trainable"] == True:
            x_all = [x_1, x_2, x_3, np.zeros(np.shape(x_1))]
        else:
            x_all = [x_1, x_2, x_3]

        if y != None:
            # _, y = self.get_label_list()
            onehot_label = []
            for sample in y:
                onehot = [0] * model_hyper["label"]
                onehot[sample] = 1
                onehot_label.append(onehot)

            onehot_label = np.array(onehot_label)
            return x_all, onehot_label
        return x_all

    def encode_data_2(self, X, y=None):
        x = []

        for sample in X:
            if type(sample) == tuple and len(sample) == 2:
                encoded = sentence2idx(xlnet_hyper["len_max"], sample[0], sample[1])
            else:
                encoded = sentence2idx(xlnet_hyper["len_max"], sample)
            x.append(encoded)

        x_1 = np.array([i[0][0] for i in x])
        x_2 = np.array([i[1][0] for i in x])
        x_3 = np.array([i[2][0] for i in x])
        if xlnet_hyper["trainable"] == True:
            x_all = [x_1, x_2, x_3, np.zeros(np.shape(x_1))]
        else:
            x_all = [x_1, x_2, x_3]

        if y != None:
            # _, y = self.get_label_list()
            onehot_label = []
            for sample in y:
                onehot = [0] * model_hyper["label"]
                onehot[sample] = 1
                onehot_label.append(onehot)

            onehot_label = np.array(onehot_label)
            return x_all, onehot_label
        return x_all


    def create_model(self):
        if embedding.built == False:
            embedding.build()
        emb = embedding.output
        # 自行修改embedding后的模型结构
        # fast text
        x = GlobalAveragePooling1D()(emb)
        output = Dense(model_hyper["label"], activation='softmax')(x)
        model = Model(inputs=embedding.input, outputs=output)
        # model.summary()
        return model


    def init(self):
        global embedding
        self.get_config()
        embedding = XlnetEmbedding(hyper_parameters=xlnet_hyper)


    def train(self, filename):
        X_train, X_val, y_train, y_val = self.process_datas(filename, mode="train")
        model = self.create_model()
        encoded_x_train, encoded_y_train = self.encode_data_2(X_train, y_train)
        encoded_x_val, encoded_y_val = self.encode_data(X_val, y_val)

        model.compile(
            optimizer=Adam(lr=model_hyper["lr"], beta_1=0.9, beta_2=0.999, decay=0.0),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            encoded_x_train, encoded_y_train,
            batch_size=model_hyper["batch_size"],
            epochs=model_hyper["epochs"],
            validation_data=(encoded_x_val, encoded_y_val),
            callbacks=[
                EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-8, patience=model_hyper["patience"]),
                ModelCheckpoint(monitor='val_loss', mode='min', filepath=model_hyper["model_path"], verbose=1,
                                save_best_only=True, save_weights_only=True),
                f1_callback(val_data=[encoded_x_val, encoded_y_val])
            ]
        )
        # 保存embedding
        if xlnet_hyper["trainable"]:
            embedding.model.save(xlnet_hyper["path_fineture"])

    def test(self, filename):
        global model_hyper
        X_test, y_test = self.process_data(filename, mode="test")
        model = self.create_model()
        if os.path.exists(model_hyper["model_path"]):
            model.load_weights(model_hyper["model_path"])
        else:
            raise RuntimeError("model path {} doesn't exist!".format(model_hyper["model_path"]))
        encoded_x_test, encoded_y_test = self.encode_data(X_test, y_test)

        # batsh_size 可以改大一点，但是必须可以整除测试样本数量
        y_pred = model.predict(encoded_x_test, batch_size=20)
        y_pred = np.argmax(y_pred, axis=1)
        y_val = np.argmax(encoded_y_test, axis=1)
        result = classification_report(y_val, y_pred)
        print(result)
        acc = accuracy_score(y_val, y_pred)
        print("acc = {}".format(acc))

    def predict(self, msg):
        global model_hyper
        # X_pre = self.process_data(filename, mode="predict")
        X_pre = [msg]
        model = self.create_model()
        if os.path.exists(model_hyper["model_path"]):
            model.load_weights(model_hyper["model_path"])
        else:
            raise RuntimeError("model path {} doesn't exist!".format(model_hyper["model_path"]))
        encoded_x_pre = self.encode_data(X_pre)

        # # batsh_size 可以改大一点，但是必须可以整除样本数量
        y_pred = model.predict(encoded_x_pre, batch_size=1)
        label, label_id = self.get_label_list()
        y_preds = np.argmax(y_pred, axis=1).tolist()[0]
        print(y_preds, type(y_preds))
        return y_pred


if __name__ == '__main__':
    domain = Domain_cls_with_Xlnet()
    domain.init()

    data1 = '/Data/xiaobensuan/cnews/'

    data2 = '/home/xsq/nlp_code/xlnet-tutorial/data/cls/'
    # domain.train(data1 + "train.txt")
    # domain.test(data1 + "test.txt")
    sentence = '科比退役了'
    domain.predict(sentence)