# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   Config.py
 
@Time    :   2019-09-15 08:22
 
@Desc    :
 
'''

class Config(object):

	"""docstring for Config"""

	def __init__(self):
		super(Config, self).__init__()
		# xlnet
		self.pretrain_xlnet_model_dir = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12/"
		self.spiece_model_file = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12/spiece.model"
		self.model_config_path = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_config.json"
		self.init_checkpoint = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_model.ckpt"

		self.do_train = True
		self.do_eval = True
		self.do_predict = True
		self.data_dir = ''
		self.train_examples_len = 30000
		self.dev_examples_len = 3000
		self.max_seq_length = 100
		self.num_labels = 10
		self.batch_size = 64
		self.num_train_epochs = 2
		self.eval_per_step = 500
		self.learning_rate = 1e-5
		self.use_tpu = False
		self.use_bfloat16 = False
		self.dropout = 0.1
		self.dropatt = 0.1         # Attention dropout rate

		self.init = "normal"
		self.init_std = 0.02
		self.init_range = 0.1
		self.clamp_len = -1
		self.summary_type = "last"
		self.use_summ_proj = True
		self.cls_scope = None
		self.task_name =  "multi_class"
		self.warmup_steps = 0
		self.decay_method = "poly"
		self.train_steps = 10000
		self.min_lr_ratio = 0.0
		self.adam_epsilon = 1e-8
		self.weight_decay = 0.0
		self.clip = 1.0
		self.lr_layer_decay_rate = 1.0
		self.shuffle_buffer = 2048

		self.max_seq_len = 150  # 输入文本片段的最大 char级别 长度
		self.output_dir = 'result'
		self.model_dir = ''   # 训练好的模型路径

		self.overwrite_data = False
		self.is_regression = False
		self.num_hosts = 1
		self.save_steps = None
		self.predict_threshold = 0
		self.uncased = False
		self.predict_dir = None
		self.eval_split = 'dev'
		self.eval_all_ckpt = False
		self.num_passes = 1
		self.predict_ckpt = None

if __name__ == "__main__":
	Config()