# coding:utf-8
'''
    discriminators (meanwhile serve as classifier)
    input sequences or embeddings, return R/F scalar or classfication logits
'''

import tensorflow as tf 
from utils import *
from encoders import *

class LatentDiscriminator_mlp(BasicBlock):
    def __init__(self, hidden_units, batch_size, class_num=None, name=None):
        name = "Latent_Discriminator_mlp" if name is None else name 
        super(LatentDiscriminator_mlp, self).__init__(hidden_units, name)
        self.class_num = class_num

    def __call__(self, x, reuse=False):
        # requires 'NC' data format
        with tf.variable_scope(self.name, reuse=reuse):
            y = lrelu(dense(x, self.hidden_units[0], name="dense_0"), name="lrelu_0")
            for i in range(len(self.hidden_units) - 1):
                y = dense(y, self.hidden_units[i+1], name="dense_{}".format(i+1))
                y = bn(y, is_training=True, name="bn_{}".format(i+1))
                y = lrelu(y, name="lrelu_{}".format(i+1))

            # without sigmoid
            y_d = dense(y, 1, name="D_dense")
            y_c = dense(y, self.class_num, name='C_dense') if self.class_num is not None else None

        return y_d, y_c 

class SeqDiscriminator_RNN(BasicBlock):
    def __init__(self, hidden_units, batch_size, discriminator, cell_type, class_num=None, name=None):
        name = "Seq_Discriminator_{}".format(cell_type) if name is None else name 
        super(SeqDiscriminator_RNN, self).__init__(hidden_units, name)
        self.class_num = class_num

        self.D = discriminator(hidden_units, cell_type=cell_type, name=name)

    def __call__(self, seqs, lens, reuse=False):
        # requires 'HNW' data format
        _, states = self.D(seqs, lens, reuse=reuse)

        with tf.variable_scope(self.name, reuse=reuse):
            y_d = dense(states[-1].c, 1, name='D_dense')
            y_c = dense(states[-1].c, self.class_num, name='C_dense') if self.class_num is not None else None
        
        return y_d, y_c

class SeqDiscriminator_CNN(BasicBlock):
    # requires 'NHWC' data format
    def __init__(self, class_num=None, mode=0, name=None):
        name = "Seq_Discriminator_CNN" if name is None else name 
        super(SeqDiscriminator_CNN, self).__init__(None, name)
        self.class_num = class_num
        self.mode = mode
    
    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):

            if self.mode == 0:
                # don't add bn at the first layer 
                # e.g. x with shape (bz, 256, 3, 1)
                conv = conv2d(x, channel=18, k_h=13, k_w=2, d_h=1, d_w=1, name='conv_0') # [bz, 244, 2, 1]
                conv = lrelu(conv, name='lrelu_0') 
                conv = tf.contrib.layers.max_pool2d(conv, kernel_size=[2,1], stride=[2,1], scope='maxpool_0') # [bz, 122, 2, 1]
                
                conv = conv2d(conv, channel=36, k_h=13, k_w=1, d_h=1, d_w=1, name='conv_1') # [bz, 110, 2, 36]
                conv = bn(conv, is_training=is_training, name='bn_1')
                conv = lrelu(conv, name='lrelu_1') 
                conv = tf.contrib.layers.max_pool2d(conv, kernel_size=[2,1], stride=[2,1], scope='maxpool_1') # [bz, 55, 2, 36]

                conv = conv2d(conv, channel=24, k_h=12, k_w=1, d_h=1, d_w=1, name='conv_2') # [bz, 44, 2, 24]
                conv = bn(conv, is_training=is_training, name='bn_2')
                conv = lrelu(conv, name='lrelu_2') 
                conv = tf.contrib.layers.max_pool2d(conv, kernel_size=[2,1], stride=[2,1], scope='maxpool_2') # [bz, 22, 2, 24]
                
                conv = tf.layers.flatten(conv) # [bz, 1056]
                conv = dense(conv, 128, activation=lrelu, name='fc') # [bz, 128]
                
                y_d = dense(conv, 1, name='D_dense')
                y_c = dense(conv, self.class_num, name='C_dense') if self.class_num is not None else None
            
            elif self.mode == 1:
                # Spectral Norm Conv
                conv = conv2d(x, channel=18, k_h=13, k_w=2, d_h=1, d_w=1, sn=True, name='conv1') # [bz, 244, 2, 1]
                conv = lrelu(conv)
                conv = tf.contrib.layers.max_pool2d(conv, kernel_size=[2,1], stride=[2,1], scope='maxpool1') # [bz, 122, 2, 1]

                conv = conv2d(conv, channel=36, k_h=13, k_w=1, d_h=1, d_w=1, sn=True, name='conv2') # [bz, 110, 2, 36]
                conv = lrelu(conv) 
                conv = tf.contrib.layers.max_pool2d(conv, kernel_size=[2,1], stride=[2,1], scope='maxpool2') # [bz, 55, 2, 36]

                conv = conv2d(conv, channel=24, k_h=12, k_w=1, d_h=1, d_w=1, sn=True, name='conv3') # [bz, 44, 2, 24]
                conv = lrelu(conv) 
                conv = tf.contrib.layers.max_pool2d(conv, kernel_size=[2,1], stride=[2,1], scope='maxpool3') # [bz, 22, 2, 24]

                conv = tf.layers.flatten(conv)

                y_d = dense(conv, 1, name='D_dense')
                y_c = dense(conv, self.class_num, name='C_dense') if self.class_num is not None else None
            
            elif self.mode == 2:
                # shallow
                # e.g. x with shape [bz, 400, 3, 1]
                conv = conv2d(x, channel=36, k_h=13, k_w=2, d_h=2, d_w=1, sn=True, name='conv1') # [bz, 194, 2, 36]
                conv = lrelu(conv) 
                
                conv = conv2d(conv, channel=18, k_h=13, k_w=1, d_h=2, d_w=1, sn=True, name='conv2') # [bz, 91, 2, 18]
                conv = lrelu(conv)

                conv = tf.layers.flatten(conv)

                y_d = dense(conv, 1, name='D_dense')
                y_c = dense(conv, self.class_num, name='C_dense') if self.class_num is not None else None

        return y_d, y_c

