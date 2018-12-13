# coding:utf-8
import tensorflow as tf
from utils import *
from datamanager import datamanager
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

class AutoEncoder_RNN(BasicBlock):
    def __init__(self, 
        encoder, encoder_params,
        decoder, decoder_params,
        data_dim,
        embedding_dim,
        name=None):

        name = "AutoEncoder" if name is None else name
        super(AutoEncoder_RNN,self).__init__(hidden_units={'encoder':encoder_params['hidden_units'],
                                                       'decoder':decoder_params['hidden_units']}, name=name)

        self.encoder = encoder(encoder_params['hidden_units'], cell_type=encoder_params['cell_type'], name='encoder')
        self.decoder = decoder(decoder_params['hidden_units'], cell_type=decoder_params['cell_type'], output_depth=data_dim, name='decoder')

        self.embedding_dim = embedding_dim

    def __call__(self, x, lens, reuse=False):
        # x, [H,N,W,C]
        with tf.variable_scope(self.name, reuse=reuse):
            _, encoder_states = self.encoder(x, lens)
            
            latent_embedding = dense(encoder_states[-1].c, self.embedding_dim)

            decoder_outputs = self.decoder(latent_embedding, lens, with_state=False)

            print decoder_outputs.get_shape().as_list()

            return latent_embedding, decoder_outputs

class AutoEncoder_CNN(BasicBlock):
    def __init__(self, name=None):
        name = "AutoEncoder" if name is None else name

        super(AutoEncoder_CNN, self).__init__(hidden_units=[],name=name)
    
    def __call__(self, x, lens):
        with tf.variable_scope(self.name):
            # x [bz, 256, 3, 1]
            conv = conv2d(x, channel=16, k_h=13, k_w=2, d_h=2, d_w=1, name='conv_0') # [bz, 122, 2, 16]
            conv = lrelu(conv, name='lrelu_0')

            conv = conv2d(conv, channel=32, k_h=13, k_w=2, d_h=2, d_w=1, name='conv_1') # [bz, 55, 1, 32]
            conv = lrelu(conv, name='lrelu_1') 

            conv = conv2d(conv, channel=64, k_h=12, k_w=1, d_h=2, d_w=1, name='conv_2') # [bz, 22, 1, 64]
            conv = lrelu(conv, name='lrelu_2') 
            self.head = conv

            deconv = deconv2d(conv, channel=32, k_h=12, k_w=1, d_h=2, d_w=1, name='deconv_0')
            deconv = lrelu(deconv, name='lrelu_3')

            deconv = deconv2d(deconv, channel=16, k_h=13, k_w=2, d_h=2, d_w=1, name='deconv_1')
            deconv = lrelu(deconv, name='lrelu_4')

            deconv = deconv2d(deconv, channel=1, k_h=13, k_w=2, d_h=2, d_w=1, name='deconv_2')
            deconv = 2 * tf.nn.tanh(deconv)

            mask = tf.sequence_mask(lens)
            mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
            deconv = tf.multiply(deconv, tf.to_float(mask))

        return deconv

    def get_vars(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class AE(BasicTrainFramework):
    def __init__(self, data, batch_size, version=None):
        super(AE, self).__init__(batch_size, version)

        self.autoencoder = AutoEncoder_CNN()
        self.data = data

        self.build_network()
        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        self.source = tf.placeholder(shape=(self.batch_size, None, 3, 1), dtype=tf.float32)
        self.length = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
        self.target = tf.placeholder(shape=(self.batch_size, None, 3, 1), dtype=tf.float32)
    
    def build_optimizer(self):
        self.loss = tf.reduce_mean(tf.square(self.target - self.autoencoder_output))
        self.solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.loss, var_list=self.autoencoder.vars)
    
    def build_summary(self):
        self.sum = tf.summary.merge([tf.summary.scalar("loss", self.loss)])
    
    def build_network(self):
        self.build_placeholder()

        self.autoencoder_output = self.autoencoder(self.source, self.length)

        self.build_optimizer()

        self.build_summary
    
    def sample(self, epoch):
        print "sample after Epoch {}".format(epoch)
        data = datamanager(time_major=False, seed=19931028)
        X = data(64, var_list=["lens", "XYZ", "labels"])

        feed_dict = {
            self.source:X["XYZ"][:,:,:,None],
            self.length:X["lens"],
            self.target:X["XYZ"][:,:,:,None],
        }

        '''samples'''
        # X["pred"] = self.sess.run(self.autoencoder_output, feed_dict=feed_dict)
        # ori, lens, pred = X['XYZ'], X['lens'], X["pred"][:,:,:,0]

        # np.savez(os.path.join(self.fig_dir, "sample_epoch_{}.npz".format(epoch)), **X)
        # for i in range(3):
        #     for j in range(3):
        #         idx = i*3 + j
        #         plt.subplot(3,6, idx*2+1)
        #         plt.plot(ori[idx, :lens[idx], 0], ori[idx, :lens[idx], 1], color='g')
        #         plt.xticks([])
        #         plt.subplot(3,6, idx*2+2)
        #         plt.plot(pred[idx, :lens[idx], 0], pred[idx, :lens[idx], 1], color='r')
        #         plt.xticks([])
        # plt.savefig(os.path.join(self.fig_dir, "sample_epoch_{}.png".format(epoch)))
        # plt.clf()

        '''weights'''
        # W = self.autoencoder.get_vars(os.path.join(self.autoencoder.name, "conv_1", "weights:0"))[0]
        # weights = self.sess.run(W, feed_dict=feed_dict) # [13, 2, 1, 16], [13, 2, 16, 32]
        # for i in range(4):
        #     for j in range(4):
        #         idx = i*4 + j
        #         plt.subplot(4, 4, idx+1)
        #         plt.plot(weights[:, 0, 0, idx], weights[:, 1, 0, idx])
        #         plt.xticks([])
        # plt.savefig(os.path.join(self.fig_dir, "conv1_epoch_{}_0.png".format(epoch)))
        # plt.clf()
        # for i in range(4):
        #     for j in range(4):
        #         idx = i*4 + j
        #         plt.subplot(4, 4, idx+1)
        #         plt.plot(weights[:, :, 0, idx])
        #         plt.xticks([])
        # plt.savefig(os.path.join(self.fig_dir, "conv1_epoch_{}_1.png".format(epoch)))
        # plt.clf()

        '''feature maps'''
        head = self.sess.run(self.autoencoder.head, feed_dict=feed_dict) # [bz, 122, 2, 16]
        print head.shape 
        for i in range(4):
            plt.subplot(4, 4, i*4 + 1)
            plt.plot(X["XYZ"][i, :X["lens"][i], :])
            for j in range(1, 4):
                idx = i*4 + j
                plt.subplot(4, 4, idx+1)
                plt.plot(head[i, :(((X["lens"][i] - 12)//2 - 12)//2-11)//2, :, j])
                plt.xticks([])
        plt.savefig(os.path.join(self.fig_dir, "fm2_epoch_{}_1.png".format(epoch)))
        plt.clf()


    def train(self, epoches=10):
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            
            for idx in range(batches_per_epoch):
                cnt = batches_per_epoch * epoch + idx

                X = self.data(64, var_list=["lens", "XYZ"])

                feed_dict = {
                    self.source:X["XYZ"][:,:,:,None],
                    self.length:X["lens"],
                    self.target:X["XYZ"][:,:,:,None],
                }
                
                self.sess.run(self.solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    print "[%4d/%4d] loss=%.4f" % (idx, batches_per_epoch, loss)
            self.sample(epoch)


import numpy as np
from datamanager import datamanager

ae = AE(datamanager(time_major=False), 64, version="test")
ae.train(20)


# autoencoder = AutoEncoder_CNN()
# source = tf.placeholder(shape=(64, None, 3, 1), dtype=tf.float32)
# length = tf.placeholder(shape=(64, ), dtype=tf.int32)
# out = autoencoder(source, length)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     weight = tf.global_variables()
#     for v in weight:
#         print v
#     print 
#     weight = autoencoder.get_vars(os.path.join(autoencoder.name, "conv_0", "weights:0"))
#     for v in weight:
#         print v





        

        



    

        