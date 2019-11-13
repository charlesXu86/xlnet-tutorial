# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   xlnet_classification.py
 
@Time    :   2019-08-28 14:52
 
@Desc    :
 
'''

import tensorflow as tf
from xlnet import xlnet
from xlnet import modeling, model_utils
import time
import timedelta
import datetime
import inputs

flags = tf.flags
FLAGS = flags.FLAGS

class XlnetReadingClass(object):
    def __init__(self, model_config_path, is_training, FLAGS, input_ids, segment_ids,
                 input_mask, label, n_class):
        '''

        :param model_config_path:
        :param is_training:
        :param FLAGS:
        :param input_ids:
        :param segment_ids:
        :param input_mask:
        :param label:
        :param n_class:
        '''
        self.xlnet_config = xlnet.XLNetConfig(json_path=model_config_path)
        self.run_config = xlnet.create_run_config(is_training, True, FLAGS)
        self.input_ids = tf.transpose(input_ids, [1, 0])
        self.segment_ids = tf.transpose(segment_ids, [1, 0])
        self.input_mask = tf.transpose(input_mask, [1, 0])

        self.model = xlnet.XLNetModel(
            xlnet_config=self.xlnet_config,
            run_config=self.run_config,
            input_ids=self.input_ids,
            seg_ids=self.segment_ids,
            input_mask=self.input_mask)

        cls_scope = FLAGS.cls_scope
        summary = self.model.get_pooled_out(FLAGS.summary_type, FLAGS.use_summ_proj)
        self.per_example_loss, self.logits = modeling.classification_loss(
            hidden=summary,
            labels=label,
            n_class=n_class,
            initializer=self.model.get_initializer(),
            scope=cls_scope,
            return_logits=True)

        self.total_loss = tf.reduce_mean(self.per_example_loss)

        with tf.name_scope("train_op"):
            self.train_op, _, _ = model_utils.get_train_op(FLAGS, self.total_loss)

        with tf.name_scope("acc"):
            one_hot_target = tf.one_hot(label, n_class)
            self.acc = self.accuracy(self.logits, one_hot_target)

    def accuracy(self, logits, labels):
        arglabels_ = tf.argmax(tf.nn.softmax(logits), 1)
        arglabels = tf.argmax(tf.squeeze(labels), 1)
        acc = tf.to_float(tf.equal(arglabels_, arglabels))
        return tf.reduce_mean(acc)


def main(_):
    print('Loading config...')

    n_class = 38

    input_path = FLAGS.data_dir + "xlnetreading.tfrecords*"

    print("input_path:", input_path)
    files = tf.train.match_filenames_once(input_path)

    """
      inputs是你数据的输入路径
    """
    input_ids, input_mask, segment_ids, label_ids = inputs(files, batch_size=FLAGS.batch_size, num_epochs=5, max_seq_length=FLAGS.max_seq_length)
    model_config_path = FLAGS.model_config_path
    is_training = False
    init_checkpoint = FLAGS.init_checkpoint

    model = XlnetReadingClass(model_config_path, is_training, FLAGS, input_ids
                              , segment_ids, input_mask, label_ids, n_class)

    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map, initialized_variable_names) = model_utils.get_assignment_map_from_checkpoint(tvars,

                                                                                                      init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        print("restore sucess  on cpu or gpu")

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    print("**** Trainable Variables ****")
    for var in tvars:
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
            print("name ={0}, shape = {1}{2}".format(var.name, var.shape,
                                                     init_string))

    print("xlnet reading class  model will start train .........")

    print(session.run(files))
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)
    start_time = time.time()
    for i in range(8000):
        _, loss_train, acc = session.run([model.train_op, model.total_loss, model.acc])
        if i % 100 == 0:
            end_time = time.time()
            time_dif = end_time - start_time
            time_dif = timedelta(seconds=int(round(time_dif)))
            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2},' \
                  + '  Cost: {2}  Time:{3}  acc:{4}'
            print(msg.format(i, loss_train, time_dif, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), acc))
            start_time = time.time()
        if i % 500 == 0 and i > 0:
            saver.save(session, "../exp/reading/model.ckpt", global_step=i)
    coord.request_stop()
    coord.join(threads)
    session.close()

