# coding:utf-8
import tensorflow as tf
from utils import *
from datamanager import datamanager
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from encoders import *
from decoders import *
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class AutoEncoder_RNN(BasicBlock):
    def __init__(self, 
        encoder, encoder_params,
        decoder, decoder_params,
        input_depth, output_depth,
        embedding_dim,
        name=None):

        name = "AutoEncoder" if name is None else name
        super(AutoEncoder_RNN,self).__init__(hidden_units={'encoder':encoder_params['hidden_units'],
                                                       'decoder':decoder_params['hidden_units']}, name=name)
        self.encoder_params = encoder_params
        self.encoder = encoder(encoder_params['hidden_units'], cell_type=encoder_params['cell_type'], name='encoder')
        self.decoder = decoder(decoder_params['hidden_units'], cell_type=decoder_params['cell_type'], output_depth=output_depth, name='decoder')

        self.embedding_dim = embedding_dim
        self.input_depth = input_depth
        self.output_depth = output_depth

    def __call__(self, x, lens, reuse=False):
        # x, [H,N,W,C]
        with tf.variable_scope(self.name, reuse=reuse):
            _, encoder_states = self.encoder(x, lens)
            if self.encoder_params['cell_type'] == 'lstm':
                latent_embedding = dense(encoder_states[-1].c, self.embedding_dim)
            else:
                latent_embedding = dense(encoder_states[-1], self.embedding_dim)
            print latent_embedding.get_shape().as_list()
            decoder_outputs = self.decoder(latent_embedding, lens, with_state=False)

            print decoder_outputs.get_shape().as_list()

        return latent_embedding, decoder_outputs

class AutoEncoder_CNN(BasicBlock):
    def __init__(self, input_depth, output_depth, embedding_dim=None, fixed_length=False, name=None):
        name = "AutoEncoder" if name is None else name
        super(AutoEncoder_CNN, self).__init__(hidden_units=[],name=name)
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.embedding_dim = embedding_dim
        self.fixed_length = fixed_length
    
    def __call__(self, x, lens, reuse=False):
        if self.input_depth == 3:
            k1, k2, k3 = 2, 2, 1
        else:
            k1, k2, k3 = 3, 3, 2
        k1, k2, k3 = 2, 2, 1
        with tf.variable_scope(self.name, reuse=reuse):
            # x [bz, 256, 3, 1]
            conv = conv2d(x, channel=16, k_h=13, k_w=k1, d_h=2, d_w=1, name='conv_0') # [bz, 122, 2, 16]
            conv = lrelu(conv, name='lrelu_0')

            conv = conv2d(conv, channel=32, k_h=13, k_w=k2, d_h=2, d_w=1, name='conv_1') # [bz, 55, 1, 32]
            conv = lrelu(conv, name='lrelu_1') 

            conv = conv2d(conv, channel=64, k_h=12, k_w=k3, d_h=2, d_w=1, name='conv_2') # [bz, 22, 1, 64]
            conv = lrelu(conv, name='lrelu_2') 

            div = tf.floor_div
            conv_lens = div(div(div(lens-12,2)-12,2)-11,2)
            mask_conv = tf.sequence_mask(conv_lens)
            mask_conv = tf.expand_dims(tf.expand_dims(mask_conv, -1), -1)
            conv = tf.multiply(conv, tf.to_float(mask_conv))

            if self.fixed_length:
                conv = tf.layers.flatten(conv) # [bz, 22*64]
                latent = dense(conv, self.embedding_dim, name='latent_fc') # [bz, emb_dim]
                deconv = dense(latent, conv.get_shape().as_list()[1], name='fc') # [bz, 22*64]
                deconv = tf.reshape(deconv, tf.shape(conv))
            else:
                latent = tf.reduce_sum(conv, axis=1) # [bz, 1, 64]
                latent = tf.divide(latent, tf.to_float(tf.expand_dims(tf.expand_dims(conv_lens, -1), -1)))
                latent = tf.layers.flatten(latent) # [bz, 64]
                deconv = conv

            deconv = deconv2d(deconv, channel=32, k_h=12, k_w=k3, d_h=2, d_w=1, name='deconv_0')
            deconv = lrelu(deconv, name='lrelu_3')

            deconv = deconv2d(deconv, channel=16, k_h=13, k_w=k2, d_h=2, d_w=1, name='deconv_1')
            deconv = lrelu(deconv, name='lrelu_4')

            deconv = deconv2d(deconv, channel=1, k_h=13, k_w=k1, d_h=2, d_w=1, name='deconv_2')
            deconv = 2 * tf.nn.tanh(deconv)

            mask = tf.sequence_mask(lens)
            mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
            deconv = tf.multiply(deconv, tf.to_float(mask))

        return latent, deconv

    def get_vars(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class ConverterA_CNN(BasicBlock):
    def __init__(self, name=None):
        name = "Converter" if name is None else name
        super(ConverterA_CNN, self).__init__(hidden_units=[],name=name)
        self.input_depth = 6
        self.output_depth = 3
    
    def __call__(self, x, lens):
        with tf.variable_scope(self.name):
            # x [bz, 256, 6, 1]
            conv = conv2d(x, channel=16, k_h=13, k_w=3, d_h=2, d_w=1, name='conv_0') # [bz, 122, 4, 16]
            # conv = lrelu(conv, name='lrelu_0')
            conv = tf.nn.tanh(conv)

            conv = conv2d(conv, channel=32, k_h=13, k_w=3, d_h=2, d_w=1, name='conv_1') # [bz, 55, 2, 32]
            # conv = lrelu(conv, name='lrelu_1') 
            conv = tf.nn.tanh(conv)

            conv = conv2d(conv, channel=64, k_h=12, k_w=2, d_h=2, d_w=1, name='conv_2') # [bz, 22, 1, 64]
            # conv = lrelu(conv, name='lrelu_2') 
            conv = tf.nn.tanh(conv)

            deconv = deconv2d(conv, channel=32, k_h=12, k_w=1, d_h=2, d_w=1, name='deconv_0') # [bz, 55, 1, 32]
            # deconv = lrelu(deconv, name='lrelu_3')
            deconv = tf.nn.tanh(deconv)

            deconv = deconv2d(deconv, channel=16, k_h=13, k_w=2, d_h=2, d_w=1, name='deconv_1') # [bz, 122, 2, 16]
            # deconv = lrelu(deconv, name='lrelu_4')
            deconv = tf.nn.tanh(deconv)

            deconv = deconv2d(deconv, channel=1, k_h=13, k_w=2, d_h=2, d_w=1, name='deconv_2') # [bz, 256, 3, 1]
            deconv = 3 * tf.nn.tanh(deconv)

            mask = tf.sequence_mask(lens)
            mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
            deconv = tf.multiply(deconv, tf.to_float(mask))

        return conv, deconv

class ConverterB_CNN(BasicBlock):
    def __init__(self, name=None):
        name = "Converter" if name is None else name
        super(ConverterB_CNN, self).__init__(hidden_units=[],name=name)
    
    def __call__(self, x, lens):
        with tf.variable_scope(self.name):
            # x [bz, 256, 3, 1]
            conv = conv2d(x, channel=16, k_h=13, k_w=2, d_h=2, d_w=1, name='conv_0') # [bz, 122, 2, 16]
            conv = lrelu(conv, name='lrelu_0')

            conv = conv2d(conv, channel=32, k_h=13, k_w=2, d_h=2, d_w=1, name='conv_1') # [bz, 55, 1, 32]
            conv = lrelu(conv, name='lrelu_1') 

            conv = conv2d(conv, channel=64, k_h=12, k_w=1, d_h=2, d_w=1, name='conv_2') # [bz, 22, 1, 64]
            conv = lrelu(conv, name='lrelu_2') 

            deconv = deconv2d(conv, channel=32, k_h=12, k_w=2, d_h=2, d_w=1, name='deconv_0') # [bz, 55, 2, 32]
            deconv = lrelu(deconv, name='lrelu_3')

            deconv = deconv2d(deconv, channel=16, k_h=13, k_w=3, d_h=2, d_w=1, name='deconv_1') # [bz, 122, 4, 16]
            deconv = lrelu(deconv, name='lrelu_4')

            deconv = deconv2d(deconv, channel=1, k_h=13, k_w=3, d_h=2, d_w=1, name='deconv_2') # [bz, 256, 4, 1]
            deconv = 2 * tf.nn.tanh(deconv)

            mask = tf.sequence_mask(lens)
            mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
            deconv = tf.multiply(deconv, tf.to_float(mask))

        return deconv

class AE(BasicTrainFramework):
    def __init__(self, autoencoder, data, batch_size, version=None):
        super(AE, self).__init__(batch_size, version)

        self.autoencoder = autoencoder
        self.input_depth = autoencoder.input_depth
        self.output_depth = autoencoder.output_depth
        self.time_major = not "CNN" in autoencoder.name
        self.data = data
        self.test_X = self.data(64, phase='test', var_list=["lens", "XYZ", "AccGyo", "labels"])

        self.build_network()
        self.build_sess()
        self.build_dirs()

        print self.time_major

    @property
    def input_pointer(self):
        if self.input_depth == 3:
            return "XYZ"
        elif self.input_depth == 6:
            return "AccGyo"
    @property
    def output_pointer(self):
        if self.output_depth == 3:
            return "XYZ"
        elif self.output_depth == 6:
            return "AccGyo"

    
    def build_placeholder(self):
        if self.time_major:
            print "For RNN"
            self.source = tf.placeholder(shape=(None, self.batch_size, self.input_depth), dtype=tf.float32)
            self.length = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
            self.target = tf.placeholder(shape=(None, self.batch_size, self.output_depth), dtype=tf.float32)
        else:
            print "For CNN"
            self.source = tf.placeholder(shape=(self.batch_size, None, self.input_depth, 1), dtype=tf.float32)
            self.length = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
            self.target = tf.placeholder(shape=(self.batch_size, None, self.output_depth, 1), dtype=tf.float32)

            
    def build_optimizer(self):
        self.loss = tf.reduce_mean(tf.square(self.target - self.autoencoder_output))
        self.solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.loss, var_list=self.autoencoder.vars)
    
    def build_summary(self):
        self.sum = tf.summary.merge([tf.summary.scalar("loss", self.loss)])
    
    def build_network(self):
        self.build_placeholder()

        _, self.autoencoder_output = self.autoencoder(self.source, self.length)
        print self.autoencoder_output.get_shape().as_list()

        self.build_optimizer()

        self.build_summary()
    
    def sample(self, epoch):
        print "sample after Epoch {}".format(epoch)
        X = self.test_X

        feed_dict = {
            self.source:X[self.input_pointer],
            self.length:X["lens"],
            self.target:X[self.output_pointer],
        }

        '''outputs'''
        X["pred"] = self.sess.run(self.autoencoder_output, feed_dict=feed_dict)
        np.savez(os.path.join(self.fig_dir, "sample_epoch_{}.npz".format(epoch)), **X)

        if self.time_major:
            ori, lens, pred = np.transpose(X['XYZ'], (1,0,2)), X['lens'], np.transpose(X["pred"], (1,0,2))
        else:
            ori, lens, pred = X['XYZ'], X['lens'], X["pred"][:,:,:,0]
        print ori.shape, pred.shape

        indexes, tmp = [], {}
        for i, label in enumerate(np.argmax(X['labels'], axis=1)):
            if not tmp.has_key(label):
                indexes.append(i)
                tmp[label] = 0
        for i in range(3):
            for j in range(3):
                idx = i*3 + j
                pic_idx = indexes[idx]
                plt.subplot(3,6, idx*2+1)
                plt.plot(ori[pic_idx, :lens[pic_idx], 0], ori[pic_idx, :lens[pic_idx], 1], color='g')
                plt.xticks([])
                plt.subplot(3,6, idx*2+2)
                plt.plot(pred[pic_idx, :lens[pic_idx], 0], pred[pic_idx, :lens[pic_idx], 1], color='r')
                plt.xticks([])
        plt.savefig(os.path.join(self.fig_dir, "sample_epoch_{}.png".format(epoch)))
        plt.clf()

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
        # head = self.sess.run(self.autoencoder.head, feed_dict=feed_dict) # [bz, 122, 2, 16]
        # print head.shape 
        # for i in range(4):
        #     plt.subplot(4, 4, i*4 + 1)
        #     plt.plot(X["XYZ"][i, :X["lens"][i], :])
        #     for j in range(1, 4):
        #         idx = i*4 + j
        #         plt.subplot(4, 4, idx+1)
        #         plt.plot(head[i, :(((X["lens"][i] - 12)//2 - 12)//2-11)//2, :, j])
        #         plt.xticks([])
        # plt.savefig(os.path.join(self.fig_dir, "fm2_epoch_{}_1.png".format(epoch)))
        # plt.clf()

    def test(self):
        # data = datamanager(time_major=False, expand_dim=True, train_ratio=0.8, fold_k=None, seed=233)
        train_loss, test_loss = 0.0, 0.0
        for i in range(107):
            X = self.data(64, phase='train', var_list=["lens", "AccGyo", "XYZ"])
            feed_dict = {
                self.source:X[self.input_pointer],
                self.length:X["lens"],
                self.target:X[self.output_pointer],
            }
            train_loss += self.batch_size * self.sess.run(self.loss, feed_dict=feed_dict)
        for i in range(26):
            X = self.data(64, phase='test', var_list=["lens", "AccGyo", "XYZ"])
            feed_dict = {
                self.source:X[self.input_pointer],
                self.length:X["lens"],
                self.target:X[self.output_pointer],
            }
            test_loss += self.batch_size * self.sess.run(self.loss, feed_dict=feed_dict)
        train_loss /= float(107 * self.batch_size)
        test_loss /= float(26 * self.batch_size)
        print "train_loss=%.4f test_loss=%.4f" % (train_loss, test_loss)
        return train_loss, test_loss
    
    def get_latent(self):
        data = datamanager(time_major=self.time_major)
        for i in range(107):
            X = data(64, phase='train', var_list=["lens", "AccGyo", "XYZ"])
            if not self.time_major:
                X["XYZ"] = X["XYZ"]
                X["AccGyo"] = X["AccGyo"]
            feed_dict = {
                self.source:X[self.input_pointer],
                self.length:X["lens"],
                self.target:X[self.output_pointer],
            }
            latent = self.sess.run(self.autoencoder.latent, feed_dict=feed_dict)
            print latent.shape

    def draw(self):
        Real = [[] for _ in range(62)]
        Fake = [[] for _ in range(62)]
        for i in range(107):
            X = self.data(self.batch_size, phase='test', var_list=["lens", "AccGyo", "XYZ", "labels"])
            labels = np.argmax(X["labels"], 1)
            feed_dict = {
                self.source:X[self.input_pointer],
                self.length:X["lens"],
                self.target:X[self.output_pointer],
            }
            pred = self.sess.run(self.autoencoder_output, feed_dict=feed_dict)
            print pred.shape
            for j in range(self.batch_size):
                if len(Fake[labels[j]]) < 3:
                    # Fake[labels[j]].append(pred[j, :X["lens"][j], :, 0])
                    # Real[labels[j]].append(X["XYZ"][j, :X["lens"][j], :])
                    Fake[labels[j]].append(pred[:X["lens"][j], j, :])
                    Real[labels[j]].append(X["XYZ"][:X["lens"][j], j, :])
            lens = map(len, Fake)
            if min(lens) == 3:
                break 
        print i
        for j in range(62):
            print j, [len(x) for x in Real[j]], [len(x) for x in Fake[j]]
        
        def f(x):
            if x>=0 and x<=9:
                return str(x)
            elif x>=10 and x<36:
                return chr(65 + x - 10)
            elif x>=36:
                return chr(97 + x - 36)

        for k in range(11):
            for i in range(6):
                for j in range(3):
                    idx = i*3 + j 
                    plt.subplot(6, 6, 2*idx + 1)
                    plt.plot(Real[(i+k*6)%62][j][:, 0], Real[(i+k*6)%62][j][:, 1], color='g')
                    plt.xticks([])
                    plt.yticks([])
                    if i==0:
                        plt.title("Truth")
                    if j==0:
                        plt.ylabel(f((i+k*6)%62))
                    plt.subplot(6, 6, 2*idx + 2)
                    plt.plot(Fake[(i+k*6)%62][j][:, 0], Fake[(i+k*6)%62][j][:, 1], color='r')
                    plt.xticks([])
                    plt.yticks([])
                    if i==0:
                        plt.title("Pred")
            plt.savefig("figs/tmp/Seq2Seq_GRU_{}.png".format(k))
            plt.clf()

    def train(self, epoches=10):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            self.data.shuffle_train()
            for idx in range(batches_per_epoch):
                cnt = batches_per_epoch * epoch + idx

                X = self.data(64, var_list=["lens", "AccGyo", "XYZ"])
                if not self.time_major:
                    X["XYZ"] = X["XYZ"]
                    X["AccGyo"] = X["AccGyo"]
                # print X[self.input_pointer].shape, "-->", X[self.output_pointer].shape

                feed_dict = {
                    self.source:X[self.input_pointer],
                    self.length:X["lens"],
                    self.target:X[self.output_pointer],
                }
                
                self.sess.run(self.solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    loss, summary_str = self.sess.run([self.loss, self.sum], feed_dict=feed_dict)
                    print self.version + " [%4d] [%4d/%4d] batchloss=%.4f" % (epoch, idx, batches_per_epoch, loss)
                    self.writer.add_summary(summary_str, cnt)

            # self.test()
            if epoch % 25 == 0:
                self.sample(epoch)
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)
            # self.test()
        # self.test()
        self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)
        print os.path.join(self.model_dir, 'model.ckpt')



'''Seq2Seq'''
# autoencoder = AutoEncoder_RNN(
#     encoder_bi, {"hidden_units":[64,128], "cell_type":"rnn"},
#     dynamic_decoder, {"hidden_units":[64,128], "cell_type":"rnn"},
#     input_depth=6, output_depth=3, embedding_dim=50,
#     name="encoderdecoder_rnn"
# )
# ae = AE(autoencoder, 64, "Seq2Seq_RNN")
# ae.train(100)
# tmp = []
# for fold_id in range(5):
#     autoencoder = AutoEncoder_RNN(
#         encoder_bi, {"hidden_units":[64, 64], "cell_type":"gru"},
#         dynamic_decoder, {"hidden_units":[64, 64], "cell_type":"gru"},
#         input_depth=6, output_depth=3, embedding_dim=50,
#         name="encoderdecoder_gru"
#     )
#     data = datamanager(time_major=True, expand_dim=False, train_ratio=None, fold_k=5, seed=233)
#     ae = AE(autoencoder, data, 64, "Seq2Seq_GRU/fold_{}".format(fold_id))
#     # ae.train(epoches=100)
#     ae.saver.restore(ae.sess, os.path.join(ae.model_dir, "model.ckpt-10699"))
#     a,b = ae.test()
#     tf.reset_default_graph()
#     tmp.append([a,b])
# print tmp

# for fold_id in range(5):
#     autoencoder = ConverterA_CNN(name="encoderdecoder_CNN")
#     data = datamanager(time_major=False, expand_dim=True, train_ratio=None, fold_k=5, seed=233)
#     ae = AE(autoencoder, data, 64, "test/fold_{}".format(fold_id))
#     ae.train(epoches=100)
#     tf.reset_default_graph()
# for fold_id in range(5):
#     autoencoder = ConverterA_CNN(name="encoderdecoder_CNN")
#     data = datamanager(time_major=False, expand_dim=True, train_ratio=None, fold_k=5, seed=233)
#     ae = AE(autoencoder, data, 64, "test/fold_{}".format(fold_id))
#     ae.saver.restore(ae.sess, os.path.join(ae.model_dir, "model.ckpt-10699"))
#     ae.test()
#     tf.reset_default_graph()

# autoencoder = ConverterA_CNN(name="encoderdecoder_CNN")
# data = datamanager(time_major=False, expand_dim=True, train_ratio=None, fold_k=5, seed=233)
# ae = AE(autoencoder, data, 64, "Seq2Seq_CNN/fold_{}".format(0))
# ae.saver.restore(ae.sess, os.path.join(ae.model_dir, "model.ckpt-10699"))
# ae.draw()

# autoencoder = AutoEncoder_RNN(
#     encoder_bi, {"hidden_units":[64, 64], "cell_type":"gru"},
#     dynamic_decoder, {"hidden_units":[64, 64], "cell_type":"gru"},
#     input_depth=6, output_depth=3, embedding_dim=50,
#     name="encoderdecoder_gru"
# )
# data = datamanager(time_major=True, expand_dim=False, train_ratio=None, fold_k=5, seed=233)
# ae = AE(autoencoder, data, 64, "Seq2Seq_GRU/fold_{}".format(0))
# ae.saver.restore(ae.sess, os.path.join(ae.model_dir, "model.ckpt-10699"))
# ae.draw()
        
'''AutoEncoder'''
# autoencoder = AutoEncoder_RNN(
#     encoder_bi, {"hidden_units":[128], "cell_type":"gru"},
#     dynamic_decoder, {"hidden_units":[128], "cell_type":"gru"},
#     input_depth=3, output_depth=3, embedding_dim=50,
#     name="autoencoder_gru"
# )
# ae = AE(autoencoder, 64, "AE_GRU")
# ae.train(100)
# ae.saver.restore(ae.sess, os.path.join(ae.model_dir, "AE_GRU.ckpt-10699"))
# ae.test()

# autoencoder = AutoEncoder_RNN(
#     encoder_bi, {"hidden_units":[128], "cell_type":"lstm"},
#     dynamic_decoder, {"hidden_units":[128], "cell_type":"lstm"},
#     input_depth=3, output_depth=3, embedding_dim=50,
#     name="autoencoder_lstm"
# )
# ae = AE(autoencoder, 64, "AE_LSTM")
# ae.train(100)
    
# autoencoder = AutoEncoder_CNN(
#     input_depth=6, output_depth=6,
#     name="autoencoder_CNN")
# ae = AE(autoencoder, 64, "AE_CNN")
# ae.train(100)

# ae.saver.restore(ae.sess, os.path.join(ae.model_dir, "AE_CNN.ckpt-10699"))
# ae.test()
# ae.get_latent()