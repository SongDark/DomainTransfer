# coding:utf-8
'''
    conditional GAN
'''
import tensorflow as tf 
from utils import BasicTrainFramework

class CGAN(BasicTrainFramework):
    def __init__(self, noise_dim, caption_dim, batch_size, version):
        super(CGAN, self).__init__(batch_size, version)
        self.caption_dim = caption_dim
        self.noise_dim = noise_dim
    
    def build_placeholder(self):
        self.right_source = tf.placeholder(shape=(self.batch_size, None, 6, 1), dtype=tf.float32)
        self.wrong_source = tf.placeholder(shape=(self.batch_size, None, 6, 1), dtype=tf.float32)
        self.right_caption = tf.placeholder(shape=(self.batch_size, self.caption_dim), dtype=tf.float32)
        self.z = tf.placeholder(shape=(self.batch_size, self.noise_dim), dtype=tf.float32)
    
    