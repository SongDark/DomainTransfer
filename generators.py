# coding:utf-8
'''
    generators
    input fixed-width embedding and expected length, return a sequence
'''

import tensorflow as tf 
from utils import *
from decoders import dynamic_decoder

class SeqGenerator_RNN(BasicBlock):
    def __init__(self, hidden_units, generator, out_dim, cell_type='lstm', name=None):
        name = "Seq_Generator_{}".format(cell_type) if name is None else name 
        super(SeqGenerator_RNN, self).__init__(hidden_units, name)
        self.out_dim = out_dim
        self.generator = generator(hidden_units, out_dim, cell_type)
    
    def __call__(self, z, lens, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):

            decoder_output = self.generator(z, lens, with_state=False)

            return decoder_output

class SeqGenerator_CNN(BasicBlock):
    def __init__(self, name=None):
        name = "Seq_Generator_CNN" if name is None else name 
        super(SeqGenerator_CNN, self).__init__(None, name)
    
    def __call__(self, z, lens, is_training=True, reuse=False):
        batch_size = z.get_shape().as_list()[0]
        with tf.variable_scope(self.name, reuse=reuse):
            g = lrelu(bn(linear(z, 1056, name='linear_0'), is_training=is_training, name='bn_0'), name='lrelu_0')
            g = tf.reshape(g, (batch_size, 22, 2, 24)) 

            g = lrelu(bn(deconv2d(g, [batch_size, 55, 2, 36], k_h=12, k_w=1, d_h=2, d_w=1, name='deconv_1'), 
                         is_training=is_training, name='bn_1'), name='lrelu_1')

            g = lrelu(bn(deconv2d(g, [batch_size, 122, 2, 18], k_h=13, k_w=1, d_h=2, d_w=1, name='deconv_2'), 
                         is_training=is_training, name='bn_2'), name='lrelu_2')

            g = deconv2d(g, [batch_size, 256, 3, 1], k_h=13, k_w=2, d_h=2, d_w=1, name='deconv_3')

            return g

