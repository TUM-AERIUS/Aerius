import tensorflow as tf

import numpy as np


class BB_CNN:
	"""
	Convolutional neural network to predict a bounding box around an object.
	
	Network architecture
	(Conv - ReLu - Max Pool) * M - (Fc - ReLu - Dropout) * N - Fc
	
	Output
	out[, 0]: score for probability, i.e. probability = sigmoid(score), 
	out[, 1]: rel. x coord of bounding box
	out[, 2]: rel. y coord of bounding box
	out[, 3]: logarithm of rel. width of bounding box
	out[, 4]: logarithm of rel. height of bounding box
	"""

	def __init__(self, kernel_size = [3], kernel_stride = [1], num_filters = [4],
			  pool_size = [2], pool_stride = [2], hidden_dim = [100], dropout = 0.5, 
			  weight_scale = 0.001, loss_bb_weight = 0.5, file_name = None):
		"""
		Initialize the bounding boxes CNN by storing its characteristics
		:param kernel_size: list of kernel sizes; all kernels are quadratic
		:param kernel_stride: list of strides of convolutional layers
		:param num_filters: list of number of filters of convolutional layers
		:param pool_size: list of pool sizes; all pool layers use quadratic kernels; if you do not want a pool after a convolutional layer, set pool_size = 1
		:param pool_stride: list of pool strides; if you do not want a pool after a convolutional layer, set pool_stride = 1
		:param hidden_dim: list of number of hidden units of fully connected layers
		:param dropout: dropout probability; set to 0 to not use dropout
		:param weight_scale: max value of positiv truncated normal distribution to draw weights from
		:param loss_bb_weight: weight for bounding boxes L2 norm loss in total loss (bb loss +  cross entropy loss of classification)
		:param file_name: file name of numpy file where weights are stored in
		"""
		self.kernel_size = kernel_size
		self.kernel_stride = kernel_stride
		self.num_filters = num_filters
		self.pool_size = pool_size
		self.pool_stride = pool_stride
		self.hidden_dim = hidden_dim
		self.dropout = dropout
		self.weight_scale = 0.001
		self.loss_bb_weight = loss_bb_weight
		self.var_dict = {}
		
		# Load coefficients if file name is provided
		if file_name is not None:
			self.data_dict = np.load(file_name, encoding='latin1').item()
		else:
			self.data_dict = None
	
	
	def build(self, x, train_mode=None):
		"""
		Method to build the computational graph
		:param x: batch of images of size [batch_size, height, width, in_channels]
		:param train_mode: True for training, False or None otherwise. Dropout is only done during training
		"""
		_, height, width, in_channels = x.get_shape().as_list()
		self.out = x
		
		# Convolutional layers
		num_filters = self.num_filters
		num_filters.insert(0, in_channels)
		pool_count = 1
		conv_count = 1
		for i in range(len(self.kernel_size)):
			self.out = self.conv_layer(self.out, self.kernel_size[i], self.kernel_stride[i], 
						num_filters[i], num_filters[i + 1], 'conv' + str(pool_count) + '_' + str(conv_count))
			conv_count += 1
			height = np.ceil(1.0 * height / self.kernel_stride[i]).astype('int')
			width = np.ceil(1.0 * width / self.kernel_stride[i]).astype('int')
			if (self.pool_size[i] > 1) & (self.pool_stride[i] > 1):
				self.out = self.max_pool(self.out, self.pool_size[i], self.pool_stride[i], 'pool' + str(pool_count))
				height = np.ceil(1.0 * height / self.pool_stride[i]).astype('int')
				width = np.ceil(1.0 * width / self.pool_stride[i]).astype('int')
				pool_count += 1
				conv_count = 1
		
		# Fully connected layers
		hidden_dim = self.hidden_dim
		hidden_dim.insert(0, num_filters[-1] * height * width)
		for i in range(len(hidden_dim) - 1):
			self.out = self.fc_layer(self.out, hidden_dim[i], hidden_dim[i + 1], 'fc' + str(i + 1))
			self.out = tf.nn.relu(self.out)
			if train_mode is not None:
				self.out = tf.cond(train_mode, lambda: tf.nn.dropout(self.out, self.dropout), lambda: self.out)
		
		# Output layer
		self.out = self.fc_layer(self.out, hidden_dim[-1], 5, 'out')
	
	
	def predict(self):
		"""
		Method to get the prediction for the bounding box
		Can only be used after network was build!
		"""
		score_prob, score_pos, score_size = tf.split(self.out, [1, 2, 2], 1)
		self.score_prob = tf.reshape(score_prob, [-1])
		self.pred_prob = tf.sigmoid(self.score_prob)
		pos = tf.map_fn(lambda x: tf.minimum(tf.nn.relu(x), tf.constant(1.)), score_pos)
		size = tf.minimum(tf.exp(score_size), 1. - pos)
		self.pred_bb = tf.concat([pos, size], 1)
	
	
	def loss(self, target_prob, target_bb):
		"""
		Method to get the loss
		Can only be used after network was build and prediction done!
		:param target_prob: indicator whether object is there or not
		:param target_bb: ground truth coordinates of the bounding box
		"""
		weight_sum = tf.reduce_sum(target_prob)
		loss_bb = tf.cond(tf.greater(weight_sum, 0), lambda: tf.reduce_sum(target_prob * tf.reduce_mean((self.pred_bb - target_bb) ** 2, 1)) / tf.reduce_sum(target_prob), lambda: 0.)
		loss_prob = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_prob, logits=self.score_prob))
		self.loss = self.loss_bb_weight * loss_bb + (1 - self.loss_bb_weight) * loss_prob
	
	
	def max_pool(self, x, size, stride, name):
		"""
		Pass x through a max pooling layer
		"""
		return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)
	
	
	def conv_layer(self, x, size, stride, in_channels, out_channels, name):
		"""
		Pass x through a convolutional layer followed by a relu
		"""
		with tf.variable_scope(name):
			filters, biases = self.get_conv_var(size, in_channels, out_channels, name)
				
			out_conv = tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding='SAME')
			out_bias = tf.nn.bias_add(out_conv, biases)
			out_relu = tf.nn.relu(out_bias)
			
			return out_relu
	
	
	def fc_layer(self, x, in_size, out_size, name):
		"""
		Pass x through a fully connected layer
		"""
		with tf.variable_scope(name):
			weights, biases = self.get_fc_var(in_size, out_size, name)
			reshaped_x = tf.reshape(x, [-1, in_size])
			out_fc = tf.nn.bias_add(tf.matmul(reshaped_x, weights), biases)
			return out_fc
	
	
	def get_conv_var(self, filter_size, in_channels, out_channels, name):
		"""
		Create parameters of convolutional layer as tf.Variable
		"""
		initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, self.weight_scale)
		filters = self.get_var(initial_value, name, 0, name + "_filters")
		
		initial_value = tf.truncated_normal([out_channels], 0.0, self.weight_scale)
		biases = self.get_var(initial_value, name, 1, name + "_biases")
		
		return filters, biases
	
	
	def get_fc_var(self, in_size, out_size, name):
		"""
		Create parameters of fully connected layer as tf.Variable
		"""
		initial_value = tf.truncated_normal([in_size, out_size], 0.0, self.weight_scale)
		weights = self.get_var(initial_value, name, 0, name + "_weights")
		
		initial_value = tf.truncated_normal([out_size], 0.0, self.weight_scale)
		biases = self.get_var(initial_value, name, 1, name + "_biases")
		
		return weights, biases
	
	
	def get_var(self, initial_value, name, idx, var_name):
		if self.data_dict is not None and name in self.data_dict:
			value = self.data_dict[name][idx]
		else:
			value = initial_value
		
		var = tf.Variable(value, name=var_name)
		
		self.var_dict[(name, idx)] = var

		return var
	
	
	def save(self, sess, file_name="./bb_cnn.npy"):
		"""
		Save variables to file
		"""
		data_dict = {}
		
		for (name, idx), var in list(self.var_dict.items()):
			var_out = sess.run(var)
			if name not in data_dict:
				data_dict[name] = {}
			data_dict[name][idx] = var_out
		
		np.save(file_name, data_dict)
		
		return file_name
	