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
		self.pretrain_xlnet_model_dir = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12"
		self.spiece_model_file = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12/spiece.model"
		self.model_config_path = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_config.json"
		self.init_checkpoint = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_model.ckpt"

		self.do_train = True
		self.task_name = "ner"
		self.train_data = "/home/xsq/nlp_code/xlnet-tutorial/data/ner"
		self.dev_data = "/home/xsq/nlp_code/xlnet-tutorial/data/test.txt"
		self.train_examples_len = 30000
		self.dev_examples_len = 3000
		self.predict_batch_size = 16
		self.num_labels = 10
		self.random_seed = 40
		self.train_batch_size = 8
		self.eval_batch_size = 8
		self.num_train_epochs = 2
		self.lower_case = False
		self.eval_per_step = 500
		self.learning_rate = 1e-5
		self.use_tpu = False
		self.use_bfloat16 = False
		self.dropout = 0.1
		self.dropatt = 0.1
		self.init = "normal"
		self.init_std = 0.02
		self.init_range = 0.1
		self.clamp_len = -1
		self.summary_type = "last"
		self.use_summ_proj = True
		self.cls_scope = None
		self.warmup_steps = 0
		self.decay_method = "poly"
		self.train_steps = 100000
		self.iterations = 1000
		self.min_lr_ratio = 0.0
		self.adam_epsilon = 1e-8
		self.weight_decay = 0.0
		self.clip = 1.0
		self.lr_layer_decay_rate = 1.0
		self.output_dir = ""
		self.num_train_steps = 100
		self.save_checkpoints_steps = 500
		self.do_predict = False
		self.do_export = True
		self.do_eval = True
		self.export_dir = "export"
		self.master = None
		self.num_core_per_host = 1
		self.num_hosts = 1
		self.max_save = 100000
		self.save_steps = None
		self.predict_tag = "这个很好"

		self.model_dir = None

		self.max_seq_length = 100  # 输入文本片段的最大 char级别 长度
		self.export_dir = "./saved_models/"  # 保存模型路径

if __name__ == "__main__":
	Config()