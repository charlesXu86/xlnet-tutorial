from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..') # walkaround due to submodule absolute import...

import collections
import os
import sys
import time
import json
sys.path.append("./xlnet")


import tensorflow as tf
import numpy as np
import sentencepiece as sp
import codecs

from xlnet import xlnet
from xlnet import prepro_utils
from xlnet import model_utils

from ner.NER_Config import Config

cf = Config()    # 初始化超参数配置

MIN_FLOAT = -1e30

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self,
                 guid,
                 text,
                 label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 input_masks,
                 segment_ids,
                 label_ids):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class NerProcessor(object):
    """Processor for the NER data set."""
    def __init__(self, data_dir, task_name):
        self.labels = ('U', 'A_beg', 'O_mid', 'O_beg', 'A', 'A_mid', 'O_end', 'A_end', 'O', 'X', '<pad>', '<cls>', '<sep>')
        self.data_dir = data_dir
        self.task_name = task_name
    
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        example_list = self._get_example(self._read_data(os.path.join(self.data_dir, "train.txt")), 'train')
        return example_list
    
    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        example_list = self._get_example(self._read_data(os.path.join(self.data_dir, "dev.txt")), 'dev')
        return example_list
    
    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        example_list = self._get_example(self._read_data(os.path.join(self.data_dir, "test.txt")), 'test')
        return example_list
    
    def _read_text(self, input_file):
        if os.path.exists(input_file):
            with open(input_file, "rb") as file:
                data_list = []
                for line in file:
                    if len(line) > 2:
                        data_list.append(line.strip())

                return data_list
        else:
            raise FileNotFoundError("data path not found: {0}".format(input_file))
    


    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split('\t')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines
    
    def _get_example(self, data_list, set_type):
        example_list = []
        for (i, data) in enumerate(data_list):
            guid = "%s-%s" % (set_type, i)
            text = data[1]
            label = data[0]
            example = InputExample(guid=guid, text=text, label=label)
            example_list.append(example)
        
        return example_list

class XLNetTokenizer(object):
    """Default text tokenizer for XLNet"""
    def __init__(self,
                 sp_model_file,
                 lower_case=False):
        """Construct XLNet tokenizer"""
        self.sp_processor = sp.SentencePieceProcessor()
        self.sp_processor.Load(sp_model_file)
        self.lower_case = lower_case
    
    def tokenize(self, text):
        """Tokenize text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        tokenized_pieces = prepro_utils.encode_pieces(self.sp_processor, processed_text)
        return tokenized_pieces
    
    def encode(self, text):
        """Encode text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        encoded_ids = prepro_utils.encode_ids(self.sp_processor, processed_text)
        return encoded_ids
    
    def token_to_id(self,
                    token):
        """Convert token to id for XLNet"""
        return self.sp_processor.PieceToId(token)
    
    def id_to_token(self,
                    id):
        """Convert id to token for XLNet"""
        return self.sp_processor.IdToPiece(id)
    
    def tokens_to_ids(self,
                      tokens):
        """Convert tokens to ids for XLNet"""
        return [self.sp_processor.PieceToId(token) for token in tokens]
    
    def ids_to_tokens(self,
                      ids):
        """Convert ids to tokens for XLNet"""
        return [self.sp_processor.IdToPiece(id) for id in ids]

class XLNetExampleConverter(object):
    """Default example converter for XLNet"""
    def __init__(self,
                 label_list,
                 max_seq_length,
                 tokenizer):
        """Construct XLNet example converter"""
        self.special_vocab_list = ["<unk>", "<s>", "</s>", "<cls>", "<sep>", "<pad>", "<mask>", "<eod>", "<eop>"]
        self.special_vocab_map = {}
        for (i, special_vocab) in enumerate(self.special_vocab_list):
            self.special_vocab_map[special_vocab] = i
        
        self.segment_vocab_list = ["<a>", "<b>", "<cls>", "<sep>", "<pad>"]
        self.segment_vocab_map = {}
        for (i, segment_vocab) in enumerate(self.segment_vocab_list):
            self.segment_vocab_map[segment_vocab] = i
        
        self.label_list = label_list
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i
        
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
    
    def convert_single_example(self, example, logging=False):
        '''
        对单个样本进行分析, 然后将字转化为id，标签转化为id，然后结构化到InputFeature中
        :param example:
        :param logging:
        :return:
        '''
        default_feature = InputFeatures(
            input_ids=[0] * self.max_seq_length,
            input_masks=[1] * self.max_seq_length,
            segment_ids=[0] * self.max_seq_length,
            label_ids=[0] * self.max_seq_length)
        
        if isinstance(example, PaddingInputExample):
            return default_feature
        
        token_items = self.tokenizer.tokenize(example.text)
        label_items = example.label.split(" ")
        
        if len(label_items) != len([token for token in token_items if token.startswith(prepro_utils.SPIECE_UNDERLINE)]):
            return default_feature
        
        tokens = []
        labels = []
        idx = 0
        for token in token_items:
            if token.startswith(prepro_utils.SPIECE_UNDERLINE):
                label = label_items[idx]
                idx += 1
            else:
                label = "X"
            
            tokens.append(token)
            labels.append(label)
        
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[0:(self.max_seq_length - 2)]

        if len(labels) > self.max_seq_length - 2:
            labels = labels[0:(self.max_seq_length - 2)]
        
        printable_tokens = [prepro_utils.printable_text(token) for token in tokens]
        
        # The convention in XLNet is:
        # (a) For sequence pairs:
        #  tokens:      is it a dog ? [SEP] no , it is not . [SEP] [CLS] 
        #  segment_ids: 0  0  0 0   0 0     1  1 1  1  1   1 1     2
        # (b) For single sequences:
        #  tokens:      this dog is big . [SEP] [CLS] 
        #  segment_ids: 0    0   0  0   0 0     2
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the last vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense when
        # the entire model is fine-tuned.
        input_tokens = []
        segment_ids = []
        label_ids = []
        
        for i, token in enumerate(tokens):
            input_tokens.append(token)
            segment_ids.append(self.segment_vocab_map["<a>"])
            label_ids.append(self.label_map[labels[i]])

        input_tokens.append("<sep>")
        segment_ids.append(self.segment_vocab_map["<a>"])
        label_ids.append(self.label_map["<sep>"])
        
        input_tokens.append("<cls>")
        segment_ids.append(self.segment_vocab_map["<cls>"])
        label_ids.append(self.label_map["<cls>"])
        
        input_ids = self.tokenizer.tokens_to_ids(input_tokens)
        
        # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are attended to.
        input_masks = [0] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        if len(input_ids) < self.max_seq_length:
            pad_seq_length = self.max_seq_length - len(input_ids)
            input_ids = [self.special_vocab_map["<pad>"]] * pad_seq_length + input_ids
            input_masks = [1] * pad_seq_length + input_masks
            segment_ids = [self.segment_vocab_map["<pad>"]] * pad_seq_length + segment_ids
            label_ids = [self.label_map["<pad>"]] * pad_seq_length + label_ids
        
        assert len(input_ids) == self.max_seq_length
        assert len(input_masks) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length
        
        if logging:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(printable_tokens))
            tf.logging.info("labels: %s" % " ".join(labels))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_masks: %s" % " ".join([str(x) for x in input_masks]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        feature = InputFeatures(
            input_ids=input_ids,
            input_masks=input_masks,
            segment_ids=segment_ids,
            label_ids=label_ids)
        
        return feature
    
    def convert_examples_to_features(self, examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        for (idx, example) in enumerate(examples):
            if idx % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

            feature = self.convert_single_example(example, logging=(idx < 5))
            features.append(feature)

        return features
    
    def file_based_convert_examples_to_features(self, examples, output_file):
        '''
        将数据转化为TF_Record 结构，作为模型数据输入
        :param examples:
        :param output_file: tf_record数据
        :return:
        '''
        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        
        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for (idx, example) in enumerate(examples):
                if idx % 10000 == 0:
                    tf.logging.info("Writing example %d of %d" % (idx, len(examples)))

                feature = self.convert_single_example(example, logging=(idx < 5))

                features = collections.OrderedDict()
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_masks"] = create_float_feature(feature.input_masks)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["label_ids"] = create_int_feature(feature.label_ids)

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())

class XLNetInputBuilder(object):
    """Default input builder for XLNet"""
    @staticmethod
    def get_input_builder(features,
                          seq_length,
                          is_training,
                          drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        all_input_ids = []
        all_input_masks = []
        all_segment_ids = []
        all_label_ids = []
        
        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_masks.append(feature.input_masks)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_ids)
        
        def input_fn(params,
                     input_context=None):
            batch_size = params["batch_size"]
            num_examples = len(features)
            
            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                "input_masks": tf.constant(all_input_masks, shape=[num_examples, seq_length], dtype=tf.float32),
                "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                "label_ids": tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            })
            
            if input_context is not None:
                tf.logging.info("Input pipeline id %d out of %d", input_context.input_pipeline_id, input_context.num_replicas_in_sync)
                d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100, seed=np.random.randint(10000))
            
            d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            return d
        
        return input_fn
    
    @staticmethod
    def get_file_based_input_fn(input_file,
                                seq_length,
                                is_training,
                                drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_masks": tf.FixedLenFeature([seq_length], tf.float32),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        }
        
        def _decode_record(record,
                           name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)
            
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example
        
        def input_fn(params, input_context=None):
            """The actual input function."""
            batch_size = params["batch_size"]
            
            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            
            if input_context is not None:
                tf.logging.info("Input pipeline id %d out of %d", input_context.input_pipeline_id, input_context.num_replicas_in_sync)
                d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100, seed=np.random.randint(10000))
            
            d = d.apply(tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
            
            return d
        
        return input_fn
    
    @staticmethod
    def get_serving_input_fn(seq_length):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        def serving_input_fn():
            with tf.variable_scope("serving"):
                features = {
                    'input_ids': tf.placeholder(tf.int32, [None, seq_length], name='input_ids'),
                    'input_masks': tf.placeholder(tf.float32, [None, seq_length], name='input_masks'),
                    'segment_ids': tf.placeholder(tf.int32, [None, seq_length], name='segment_ids')
                }

                return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()
        
        return serving_input_fn

class XLNetModelBuilder(object):
    """Default model builder for XLNet"""
    def __init__(self,
                 model_config,
                 use_tpu=False):
        """Construct XLNet model builder"""
        self.model_config = model_config
        self.use_tpu = use_tpu
    
    def _get_masked_data(self,
                         data_ids,
                         label_list):
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        
        pad_id = tf.constant(label_map["<pad>"], shape=[], dtype=tf.int32)
        out_id = tf.constant(label_map["O"], shape=[], dtype=tf.int32)
        x_id = tf.constant(label_map["X"], shape=[], dtype=tf.int32)
        cls_id = tf.constant(label_map["<cls>"], shape=[], dtype=tf.int32)
        sep_id = tf.constant(label_map["<sep>"], shape=[], dtype=tf.int32)

        masked_data_ids = (tf.cast(tf.not_equal(data_ids, pad_id), dtype=tf.int32) *
            tf.cast(tf.not_equal(data_ids, out_id), dtype=tf.int32) *
            tf.cast(tf.not_equal(data_ids, x_id), dtype=tf.int32) *
            tf.cast(tf.not_equal(data_ids, cls_id), dtype=tf.int32) *
            tf.cast(tf.not_equal(data_ids, sep_id), dtype=tf.int32))

        return masked_data_ids
    
    def _create_model(self,
                      input_ids,
                      input_masks,
                      segment_ids,
                      label_ids,
                      label_list,
                      mode):
        """Creates XLNet-NER model"""
        model = xlnet.XLNetModel(
            xlnet_config=self.model_config,
            run_config=xlnet.create_run_config(mode == tf.estimator.ModeKeys.TRAIN, True, cf),
            input_ids=tf.transpose(input_ids, perm=[1,0]),
            input_mask=tf.transpose(input_masks, perm=[1,0]),
            seg_ids=tf.transpose(segment_ids, perm=[1,0]))
        
        initializer = model.get_initializer()
        
        with tf.variable_scope("ner", reuse=tf.AUTO_REUSE):
            result = tf.transpose(model.get_sequence_output(), perm=[1,0,2])
            result_mask = tf.cast(tf.expand_dims(1 - input_masks, axis=-1), dtype=tf.float32)
            
            dense_layer = tf.keras.layers.Dense(units=len(label_list), activation=None, use_bias=True,
                kernel_initializer=initializer, bias_initializer=tf.zeros_initializer,
                kernel_regularizer=None, bias_regularizer=None, trainable=True)
            
            dropout_layer = tf.keras.layers.Dropout(rate=0.1, seed=np.random.randint(10000))
            
            result = dense_layer(result)
            if mode == tf.estimator.ModeKeys.TRAIN:
                result = dropout_layer(result)
            
            masked_predict = result * result_mask + MIN_FLOAT * (1 - result_mask)
            predict_ids = tf.cast(tf.argmax(tf.nn.softmax(masked_predict, axis=-1), axis=-1), dtype=tf.int32)
        
        loss = tf.constant(0.0, dtype=tf.float32)
        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL] and label_ids is not None:
            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
                label = tf.cast(label_ids, dtype=tf.float32)
                label_mask = tf.cast(1 - input_masks, dtype=tf.float32)
                masked_label = tf.cast(label * label_mask, dtype=tf.int32)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_label, logits=masked_predict)
                loss = tf.reduce_sum(cross_entropy * label_mask) / tf.reduce_sum(tf.reduce_max(label_mask, axis=-1))
        
        return loss, predict_ids
    
    def get_model_fn(self, label_list):
        """Returns `model_fn` closure for TPUEstimator."""
        def model_fn(features,
                     labels,
                     mode,
                     params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""
            def metric_fn(label_ids,
                          predict_ids):
                precision = tf.metrics.precision(labels=label_ids, predictions=predict_ids)
                recall = tf.metrics.recall(labels=label_ids, predictions=predict_ids)

                metric = {
                    "precision": precision,
                    "recall": recall,
                }

                return metric
            
            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_masks = features["input_masks"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"] if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL] else None

            loss, predict_ids = self._create_model(input_ids, input_masks, segment_ids, label_ids, label_list, mode)
            
            scaffold_fn = model_utils.init_from_checkpoint(cf)
            
            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op, _, _ = model_utils.get_train_op(cf, loss)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            elif mode == tf.estimator.ModeKeys.EVAL:
                masked_label_ids = self._get_masked_data(label_ids, label_list)
                masked_predict_ids = self._get_masked_data(predict_ids, label_list)
                eval_metrics = (metric_fn, [masked_label_ids, masked_predict_ids])
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={ "predict": predict_ids },
                    scaffold_fn=scaffold_fn)
            
            return output_spec
        
        return model_fn

class XLNetPredictRecorder(object):
    """Default predict recorder for XLNet"""
    def __init__(self,
                 output_dir,
                 label_list,
                 max_seq_length,
                 tokenizer,
                 predict_tag=None):
        """Construct XLNet predict recorder"""
        self.output_path = os.path.join(output_dir, "predict.{0}.json".format(predict_tag if predict_tag else str(time.time())))
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
    
    def _write_to_json(self,
                       data_list,
                       data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:  
            json.dump(data_list, file, indent=4)
    
    def _write_to_text(self,
                       data_list,
                       data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:
            for data in data_list:
                file.write("{0}\n".format(data))
    
    def record(self,
               predicts):
        decoded_results = []
        for predict in predicts:
            input_tokens = self.tokenizer.ids_to_tokens(predict["input_ids"])
            input_masks = predict["input_masks"]
            input_labels = [self.label_list[idx] for idx in predict["label_ids"]]
            output_predicts = [self.label_list[idx] for idx in predict["predict_ids"]]
            
            decoded_tokens = []
            decoded_labels = []
            decoded_predicts = []
            results = zip(input_tokens, input_masks, input_labels, output_predicts)
            for input_token, input_mask, input_label, output_predict in results:
                if input_token in ["<cls>", "<sep>"] or input_mask == 1:
                    continue
                
                if output_predict in ["<pad>", "<cls>", "<sep>", "X"]:
                    output_predict = "O"
                
                if input_token.startswith(prepro_utils.SPIECE_UNDERLINE):
                    decoded_tokens.append(input_token)
                    decoded_labels.append(input_label)
                    decoded_predicts.append(output_predict)
                else:
                    decoded_tokens[-1] = decoded_tokens[-1] + input_token
            
            decoded_text = "".join(decoded_tokens).replace(prepro_utils.SPIECE_UNDERLINE, " ")
            decoded_label = " ".join(decoded_labels)
            decoded_predict = " ".join(decoded_predicts)
            
            decoded_result = {
                "text": prepro_utils.printable_text(decoded_text),
                "label": decoded_label,
                "predict": decoded_predict,
            }

            decoded_results.append(decoded_result)
        
        self._write_to_json(decoded_results, self.output_path)

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    
    np.random.seed(cf.random_seed)
    
    processor = NerProcessor(
        data_dir=cf.train_data,
        task_name=cf.task_name.lower())
    
    # label_list = processor.get_labels()
    label_list = processor.labels
    
    model_config = xlnet.XLNetConfig(json_path=cf.model_config_path)
    
    model_builder = XLNetModelBuilder(
        model_config=model_config,
        use_tpu=cf.use_tpu)
    
    model_fn = model_builder.get_model_fn(label_list)
    
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    tpu_config = model_utils.configure_tpu(cf)
    
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=cf.use_tpu,
        model_fn=model_fn,
        config=tpu_config,
        export_to_tpu=cf.use_tpu,
        train_batch_size=cf.train_batch_size,
        eval_batch_size=cf.eval_batch_size,
        predict_batch_size=cf.predict_batch_size)
    
    tokenizer = XLNetTokenizer(
        sp_model_file=cf.spiece_model_file,
        lower_case=cf.lower_case)
    
    example_converter = XLNetExampleConverter(
        label_list=label_list,
        max_seq_length=cf.max_seq_length,
        tokenizer=tokenizer)
    
    if cf.do_train and cf.do_eval:   # 开始训练

        train_file = os.path.join(cf.output_dir, "train.tf_record")
        tf.logging.info("Use tfrecord samples: {}".format(len(train_file)))

        train_examples = processor.get_train_examples()    # train data
        np.random.shuffle(train_examples)

        example_converter.file_based_convert_examples_to_features(train_examples, train_file)
        train_steps = int(len(train_examples) * cf.num_train_epochs / cf.train_batch_size)
        cf.warmup_steps = int(0.1 * train_steps)

        
        tf.logging.info("***** Run training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", cf.train_batch_size)
        tf.logging.info("  Num steps = %d", cf.train_steps)

        
        # train_features = example_converter.convert_examples_to_features(train_examples)

        # if not os.path.exists(train_file):
        #     train_features = example_converter.file_based_convert_examples_to_features(train_examples, train_file)


        # 读取TF_record数据
        # train_input_fn = XLNetInputBuilder.get_input_builder(train_features, cf.max_seq_length, True, True)
        train_input_fn = XLNetInputBuilder.get_file_based_input_fn(
                                    input_file= train_file,
                                    seq_length=cf.max_seq_length,
                                    is_training=True,
                                    drop_remainder=True
        )
        
        estimator.train(input_fn=train_input_fn, max_steps=train_steps)

        eval_examples = processor.get_dev_examples()
        
        tf.logging.info("***** Run evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", cf.eval_batch_size)

        # early stop hook
        # early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        #                             estimator=estimator,
        #                             metric_name='loss',
        #                             max_steps_without_decrease=cf.num_train_steps,
        #                             eval_dir=None,
        #                             min_steps=0,
        #                             run_every_secs=None,
        #                             run_every_steps=cf.save_checkpoints_steps
        # )

        eval_features = example_converter.convert_examples_to_features(eval_examples)
        eval_input_fn = XLNetInputBuilder.get_input_builder(eval_features, cf.max_seq_length, False, False)

        # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=cf.num_train_steps,
        #                                     hooks=[early_stopping_hook])
        # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        result = estimator.evaluate(input_fn=eval_input_fn)
        
        precision = result["precision"]
        recall = result["recall"]
        f1_score = 2.0 * precision * recall / (precision + recall)
        
        tf.logging.info("***** Evaluation result *****")
        tf.logging.info("  Precision (token-level) = %s", str(precision))
        tf.logging.info("  Recall (token-level) = %s", str(recall))
        tf.logging.info("  F1 score (token-level) = %s", str(f1_score))
    
    if cf.do_predict:
        predict_examples = processor.get_test_examples()
        
        tf.logging.info("***** Run prediction *****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", cf.predict_batch_size)
        
        predict_features = example_converter.convert_examples_to_features(predict_examples)
        predict_input_fn = XLNetInputBuilder.get_input_builder(predict_features, cf.max_seq_length, False, False)
        
        result = estimator.predict(input_fn=predict_input_fn)
        
        predict_recorder = XLNetPredictRecorder(
            output_dir=cf.output_dir,
            label_list=label_list,
            max_seq_length=cf.max_seq_length,
            tokenizer=tokenizer,
            predict_tag=cf.predict_tag)
        
        predicts = [{
            "input_ids": feature.input_ids,
            "input_masks": feature.input_masks,
            "label_ids": feature.label_ids,
            "predict_ids": predict["predict"].tolist()
        } for feature, predict in zip(predict_features, result)]
        
        predict_recorder.record(predicts)
    
    if cf.do_export:
        tf.logging.info("***** Running exporting *****")
        tf.io.gfile.makedirs(cf.export_dir)
        serving_input_fn = XLNetInputBuilder.get_serving_input_fn(cf.max_seq_length)
        estimator.export_saved_model(cf.export_dir, serving_input_fn, as_text=False)

if __name__ == "__main__":
    main()
