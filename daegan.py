# coding:utf-8

import tensorflow as tf 
from utils import *
from discriminators import SeqDiscriminator_CNN
from decoders import dynamic_decoder
from datamanager import datamanager
from autoencoder import AutoEncoder_CNN

class DAE_GAN(BasicTrainFramework):
    def __init__(self, batch_size, version):
        super(DAE_GAN, self).__init__(batch_size, version)

        self.data = datamanager(time_major=False)
        self.autoencoder = AutoEncoder_CNN(6, 6, 100, fixed_length=False, name='autoencoder')
        # self.encoder = SeqDiscriminator_CNN(class_num=None, mode=1, fixed_length=False, name='encoder')
        # self.decoder = dynamic_decoder([64,64], 6, 'lstm', name='decoder')

        self.time_major = False

        self.build_placeholder()
        self.build_network()
        self.build_optimizer()

        self.build_sess()

    def build_placeholder(self):
        self.source = tf.placeholder(shape=(self.batch_size, None, 6, 1), dtype=tf.float32)
        self.length = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
        self.labels = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)
        self.target = tf.placeholder(shape=(self.batch_size, None, 6, 1), dtype=tf.float32)
    
    def build_network(self):
        self.latent, self.logit = self.autoencoder(self.source, self.length)
        print self.latent.get_shape().as_list()
        print self.logit.get_shape().as_list()
    
    def build_optimizer(self):
        self.ae_loss = tf.reduce_mean(tf.square(self.logit - self.target))
        self.ae_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.ae_loss, var_list=self.autoencoder.vars)
    
    def train(self,epoches=1):
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            self.data.shuffle_train()
            for idx in range(batches_per_epoch):
                cnt = batches_per_epoch * epoch + idx

                X = self.data(64, var_list=["lens", "AccGyo", "XYZ", "labels"])

                # print X[self.input_pointer].shape, "-->", X[self.output_pointer].shape

                feed_dict = {
                    self.source:X["AccGyo"][:,:,:,None],
                    self.length:X["lens"],
                    self.labels:X["labels"],
                    self.target:X["AccGyo"][:,:,:,None],
                }
                
                self.sess.run(self.ae_solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    loss = self.sess.run(self.ae_loss, feed_dict=feed_dict)
                    print self.version + "[%4d] [%4d/%4d] batchloss=%.4f" % (epoch, idx, batches_per_epoch, loss)
                    # self.writer.add_summary(summary_str, cnt)
            break

dae = DAE_GAN(64, "test")
dae.train(1)