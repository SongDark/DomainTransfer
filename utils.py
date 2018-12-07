import tensorflow as tf 

class BasicBlock(object):
    def __init__(self, hidden_units, name):
        self.name = name
        self.hidden_units = hidden_units
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

def gen_rnn_cells(cell_type, hidden_units):
    if cell_type == 'rnn':
        return [tf.nn.rnn_cell.RNNCell(size) for size in hidden_units]
    elif cell_type == 'lstm':
        return [tf.nn.rnn_cell.LSTMCell(size, use_peepholes=False) for size in hidden_units]
    elif cell_type == 'gru':
        return [tf.nn.rnn_cell.GRUCell(size) for size in hidden_units]

def lrelu(x, leak=0.2, name='leaky_relu'):
	return tf.maximum(x, leak*x, name=name) 

def bn(x, is_training, name):
	return tf.contrib.layers.batch_norm(x, 
										decay=0.999, 
										updates_collections=None, 
										epsilon=0.001, 
										scale=True,
										fused=False,
										is_training=is_training,
										scope=name)

def spectral_norm(w, iteration=1, name="sn"):
	'''
	Ref: https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/65218e8cc6916d24b49504c337981548685e1be1/spectral_norm.py
	'''
	w_shape = w.shape.as_list() # [A, B, C]
	w = tf.reshape(w, [-1, w_shape[-1]]) # [AB, C]

	u = tf.get_variable(name+"_u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

	u_hat = u # [1, C]
	v_hat = None 

	for _ in range(iteration):
		v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w))) # [1, AB]
		u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w)) # [1, C]
		
	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

	with tf.control_dependencies([u.assign(u_hat)]):
		# ops here run after u.assign(u_hat)
		w_norm = w / sigma 
		w_norm = tf.reshape(w_norm, w_shape)
	
	return w_norm

def linear(x, output_size, stddev=0.02, bias_start=0.0, name='linear'):
	shape = x.get_shape().as_list()
	with tf.variable_scope(name):
		W = tf.get_variable(
			'weights', [shape[1], output_size], 
			tf.float32, 
			tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable(
			'biases', [output_size], 
			initializer=tf.constant_initializer(bias_start))

	return tf.matmul(x,W) + bias

def dense(x, output_size, stddev=0.02, bias_start=0.0, activation=None, name='dense'):
	shape = x.get_shape().as_list()
	with tf.variable_scope(name):
		W = tf.get_variable(
			'weights', [shape[1], output_size], 
			tf.float32, 
			tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable(
			'biases', [output_size], 
			initializer=tf.constant_initializer(bias_start))
	
	out = tf.matmul(x, W) + bias 
	if activation is not None:
		out = activation(out)
	
	return out

def conv2d(x, channel, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, sn=False, name='conv2d'):
	with tf.variable_scope(name):
		w = tf.get_variable('weights', [k_h, k_w, x.get_shape()[-1], channel], 
				initializer=tf.truncated_normal_initializer(stddev=stddev))
		biases = tf.get_variable('biases', shape=[channel], initializer=tf.zeros_initializer())
        if sn:
            w = spectral_norm(w, name=name+"_sn")
            
	conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding='VALID')
	conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
	return conv

def deconv2d(x, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, sn=False, name='deconv2d'):
    # output_shape: [height, width, output_channels, in_channels]
	with tf.variable_scope(name):
		w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], x.get_shape()[-1]],
					initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', shape=[output_shape[-1]], initializer=tf.zeros_initializer())
        if sn:
            w = spectral_norm(w)
            
	deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding='VALID')
	deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
	return deconv

