# -*- coding: utf-8 -*-

'''
@Author  :   Xu

@Software:   PyCharm

@File    :   sentiment_classifier.py

@Time    :   2019-09-15 08:22

@Desc    :

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import os
import sys
import csv
import collections
import numpy as np
import json

import tensorflow as tf

import sentencepiece as spm

sys.path.append('..')

from xlnet.data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
from xlnet import model_utils
from xlnet import function_builder
from xlnet.classifier_utils import PaddingInputExample
from xlnet.classifier_utils import convert_single_example
from xlnet.prepro_utils import preprocess_text, encode_ids

from sentiment_classification.Config import Config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cf = Config()

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) == 0: continue
                lines.append(line)
            return lines


class CSCProcessor(DataProcessor):
    def get_labels(self):
        return ["0", "1"]

    def get_train_examples(self, data_dir):
        set_type = "train"
        input_file = os.path.join(data_dir, set_type + ".tsv")
        tf.logging.info("using file %s" % input_file)
        lines = self._read_tsv(input_file)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)

            text_a = line[1]
            label = line[0]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_devtest_examples(self, data_dir, set_type="dev"):
        input_file = os.path.join(data_dir, set_type + ".tsv")
        tf.logging.info("using file %s" % input_file)
        lines = self._read_tsv(input_file)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)

            text_a = line[1]
            label = line[0]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenize_fn, output_file,
        num_passes=1):
    """Convert a set of `InputExample`s to a TFRecord file."""

    # do not create duplicated records
    if tf.gfile.Exists(output_file) and not cf.overwrite_data:
        tf.logging.info("Do not overwrite tfrecord {} exists.".format(output_file))
        return

    tf.logging.info("Create new tfrecord {}.".format(output_file))

    writer = tf.python_io.TFRecordWriter(output_file)

    if num_passes > 1:
        examples *= num_passes

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example {} of {}".format(ex_index,
                                                              len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenize_fn)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_float_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        if label_list is not None:
            features["label_ids"] = create_int_feature([feature.label_id])
        else:
            features["label_ids"] = create_float_feature([float(feature.label_id)])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    if cf.is_regression:
        name_to_features["label_ids"] = tf.FixedLenFeature([], tf.float32)

    tf.logging.info("Input tfrecord file {}".format(input_file))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params, input_context=None):
        """The actual input function."""
        if cf.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = cf.batch_size
        elif cf.do_eval:
            batch_size = cf.batch_size
        else:
            batch_size = cf.batch_size

        d = tf.data.TFRecordDataset(input_file)
        # Shard the dataset to difference devices
        if input_context is not None:
            tf.logging.info("Input pipeline id %d out of %d",
                            input_context.input_pipeline_id, input_context.num_replicas_in_sync)
            d = d.shard(input_context.num_input_pipelines,
                        input_context.input_pipeline_id)

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = d.shuffle(buffer_size=cf.shuffle_buffer)
            d = d.repeat()

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def get_model_fn(n_class):
    def model_fn(features, labels, mode, params):
        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        #### Get loss from inputs
        if cf.is_regression:
            (total_loss, per_example_loss, logits
             ) = function_builder.get_regression_loss(cf, features, is_training)
        else:
            (total_loss, per_example_loss, logits
             ) = function_builder.get_classification_loss(
                cf, features, n_class, is_training)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        #### load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(cf)

        #### Evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            assert cf.num_hosts == 1

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                eval_input_dict = {
                    'labels': label_ids,
                    'predictions': predictions,
                    'weights': is_real_example
                }
                accuracy = tf.metrics.accuracy(**eval_input_dict)

                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    'eval_accuracy': accuracy,
                    'eval_loss': loss}

            def regression_metric_fn(
                    per_example_loss, label_ids, logits, is_real_example):
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                pearsonr = tf.contrib.metrics.streaming_pearson_correlation(
                    logits, label_ids, weights=is_real_example)
                return {'eval_loss': loss, 'eval_pearsonr': pearsonr}

            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

            #### Constucting evaluation TPUEstimatorSpec with new cache.
            label_ids = tf.reshape(features['label_ids'], [-1])

            if cf.is_regression:
                metric_fn = regression_metric_fn
            else:
                metric_fn = metric_fn
            metric_args = [per_example_loss, label_ids, logits, is_real_example]

            if cf.use_tpu:
                eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=(metric_fn, metric_args),
                    scaffold_fn=scaffold_fn)
            else:
                eval_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=metric_fn(*metric_args))

            return eval_spec

        elif mode == tf.estimator.ModeKeys.PREDICT:
            label_ids = tf.reshape(features["label_ids"], [-1])

            predictions = {
                "logits": logits,
                "labels": label_ids,
                "is_real": features["is_real_example"]
            }

            if cf.use_tpu:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predictions)
            return output_spec

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(cf, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        #### Constucting training TPUEstimatorSpec with new cache.
        if cf.use_tpu:
            #### Creating host calls
            if not cf.is_regression:
                label_ids = tf.reshape(features['label_ids'], [-1])
                predictions = tf.argmax(logits, axis=-1, output_type=label_ids.dtype)
                is_correct = tf.equal(predictions, label_ids)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                monitor_dict["accuracy"] = accuracy

                host_call = function_builder.construct_scalar_host_call(
                    monitor_dict=monitor_dict,
                    model_dir=cf.model_dir,
                    prefix="train/",
                    reduce_fn=tf.reduce_mean)
            else:
                host_call = None

            train_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
                scaffold_fn=scaffold_fn)
        else:
            train_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    #### Validate cf
    if cf.save_steps is not None:
        cf.iterations = min(cf.iterations, cf.save_steps)

    if cf.do_predict:
        predict_dir = cf.predict_dir
        if not tf.gfile.Exists(predict_dir):
            tf.gfile.MakeDirs(predict_dir)

    processors = {
        "csc": CSCProcessor
    }

    if not cf.do_train and not cf.do_eval and not cf.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval, `do_predict` or "
            "`do_submit` must be True.")

    if not tf.gfile.Exists(cf.output_dir):
        tf.gfile.MakeDirs(cf.output_dir)

    task_name = cf.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels() if not cf.is_regression else None

    sp = spm.SentencePieceProcessor()
    sp.Load(cf.spiece_model_file)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=cf.uncased)
        return encode_ids(sp, text)

    run_config = model_utils.configure_tpu(cf)

    model_fn = get_model_fn(len(label_list) if label_list is not None else None)

    spm_basename = os.path.basename(cf.spiece_model_file)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if cf.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=cf.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=cf.batch_size,
            predict_batch_size=cf.batch_size,
            eval_batch_size=cf.batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if cf.do_train:
        train_file_base = "{}.len-{}.train.tf_record".format(
            spm_basename, cf.max_seq_length)
        train_file = os.path.join(cf.output_dir, train_file_base)
        tf.logging.info("Use tfrecord file {}".format(train_file))

        train_examples = processor.get_train_examples(cf.data_dir)
        np.random.shuffle(train_examples)
        tf.logging.info("Num of train samples: {}".format(len(train_examples)))

        file_based_convert_examples_to_features(
            train_examples, label_list, cf.max_seq_length, tokenize_fn,
            train_file, cf.num_passes)

        # here we use epoch number to calculate total train_steps
        cf.train_steps = int(len(train_examples) * cf.num_train_epochs / cf.batch_size)
        cf.warmup_steps = int(0.1 * cf.train_steps)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=cf.max_seq_length,
            is_training=True,
            drop_remainder=True)

        estimator.train(input_fn=train_input_fn, max_steps=cf.train_steps)

    if cf.do_eval or cf.do_predict:
        eval_examples = processor.get_devtest_examples(cf.data_dir, cf.eval_split)
        tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

    if cf.do_eval:
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on. These do NOT count towards the metric (all tf.metrics
        # support a per-instance weight, and these get a weight of 0.0).
        #
        # Modified in XL: We also adopt the same mechanism for GPUs.
        while len(eval_examples) % cf.batch_size != 0:
            eval_examples.append(PaddingInputExample())

        eval_file_base = "{}.len-{}.{}.eval.tf_record".format(
            spm_basename, cf.max_seq_length, cf.eval_split)
        eval_file = os.path.join(cf.output_dir, eval_file_base)

        file_based_convert_examples_to_features(
            eval_examples, label_list, cf.max_seq_length, tokenize_fn,
            eval_file)

        assert len(eval_examples) % cf.batch_size == 0
        eval_steps = int(len(eval_examples) // cf.batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=cf.max_seq_length,
            is_training=False,
            drop_remainder=True)

        # Filter out all checkpoints in the directory
        steps_and_files = []
        filenames = tf.gfile.ListDirectory(cf.model_dir)

        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = join(cf.model_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files.append([global_step, cur_filename])
        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

        # Decide whether to evaluate all ckpts
        if not cf.eval_all_ckpt:
            steps_and_files = steps_and_files[-1:]

        eval_results = []
        for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
            ret = estimator.evaluate(
                input_fn=eval_input_fn,
                steps=eval_steps,
                checkpoint_path=filename)

            ret["step"] = global_step
            ret["path"] = filename

            eval_results.append(ret)

            tf.logging.info("=" * 80)
            log_str = "Eval result | "
            for key, val in sorted(ret.items(), key=lambda x: x[0]):
                log_str += "{} {} | ".format(key, val)
            tf.logging.info(log_str)

        key_name = "eval_pearsonr" if cf.is_regression else "eval_accuracy"
        eval_results.sort(key=lambda x: x[key_name], reverse=True)

        tf.logging.info("=" * 80)
        log_str = "Best result | "
        for key, val in sorted(eval_results[0].items(), key=lambda x: x[0]):
            log_str += "{} {} | ".format(key, val)
        tf.logging.info(log_str)

    if cf.do_predict:
        eval_file_base = "{}.len-{}.{}.predict.tf_record".format(
            spm_basename, cf.max_seq_length, cf.eval_split)
        eval_file = os.path.join(cf.output_dir, eval_file_base)

        file_based_convert_examples_to_features(
            eval_examples, label_list, cf.max_seq_length, tokenize_fn,
            eval_file)

        pred_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=cf.max_seq_length,
            is_training=False,
            drop_remainder=False)

        predict_results = []
        with tf.gfile.Open(os.path.join(predict_dir, "{}.tsv".format(
                task_name)), "w") as fout:
            fout.write("index\tprediction\n")

            for pred_cnt, result in enumerate(estimator.predict(
                    input_fn=pred_input_fn,
                    yield_single_examples=True,
                    checkpoint_path=cf.predict_ckpt)):
                if pred_cnt % 1000 == 0:
                    tf.logging.info("Predicting submission for example: {}".format(
                        pred_cnt))

                logits = [float(x) for x in result["logits"].flat]
                predict_results.append(logits)

                if len(logits) == 1:
                    label_out = logits[0]
                elif len(logits) == 2:
                    if logits[1] - logits[0] > cf.predict_threshold:
                        label_out = label_list[1]
                    else:
                        label_out = label_list[0]
                elif len(logits) > 2:
                    max_index = np.argmax(np.array(logits, dtype=np.float32))
                    label_out = label_list[max_index]
                else:
                    raise NotImplementedError

                fout.write("{}\t{}\n".format(pred_cnt, label_out))

        predict_json_path = os.path.join(predict_dir, "{}.logits.json".format(
            task_name))

        with tf.gfile.Open(predict_json_path, "w") as fp:
            json.dump(predict_results, fp, indent=4)


if __name__ == "__main__":
    main()
