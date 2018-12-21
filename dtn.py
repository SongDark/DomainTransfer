# coding:utf-8

import tensorflow as tf 
from utils import * 
from discriminators import SeqDiscriminator_CNN
from autoencoder import ConverterA_CNN, AutoEncoder_CNN
from datamanager import datamanager
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib 
matplotlib.use("Agg")
from matplotlib import pyplot as plt

class DTN(BasicTrainFramework):
    def __init__(self, batch_size, version):
        super(DTN, self).__init__(batch_size, version)

        self.classifier_X = SeqDiscriminator_CNN(class_num=62, mode=0, fixed_length=False, name="cls_x_cnn")
        self.classifier_Y = SeqDiscriminator_CNN(class_num=62, mode=0, fixed_length=False, name="cls_y_cnn")

        self.autoencoder_X = AutoEncoder_CNN(input_depth=6, output_depth=6, fixed_length=False, name="ae_X")
        self.autoencoder_Y = AutoEncoder_CNN(input_depth=3, output_depth=3, fixed_length=False, name="ae_Y")

        self.encoder = ConverterA_CNN(name="encoder_XtoY")
        self.discriminator = SeqDiscriminator_CNN(class_num=None, mode=1, fixed_length=False, name="discriminator")


        self.data_X = datamanager(time_major=False)
        self.data_Y = datamanager(time_major=False, seed=1)

        self.build_placeholder()
        self.build_classifier()
        # self.build_dtn()
        # self.build_ae()
        self.build_gan()
        self.build_optimizer()
        self.build_summary()

        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        self.source_X = tf.placeholder(shape=(self.batch_size, None, 6, 1), dtype=tf.float32)
        self.length_X = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32)
        self.label_X = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)
        self.target_X = tf.placeholder(shape=(self.batch_size, None, 6, 1), dtype=tf.float32)

        self.source_Y = tf.placeholder(shape=(self.batch_size, None, 3, 1), dtype=tf.float32)
        self.length_Y = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32)
        self.label_Y = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)
        self.target_Y = tf.placeholder(shape=(self.batch_size, None, 3, 1), dtype=tf.float32)

        self.G = self.encoder(self.source_X, self.length_X)

    def build_classifier(self):
        _, self.latent_X_real, self.cls_logit_X_real = self.classifier_X(self.source_X, self.length_X)
        _, self.latent_Y_real, self.cls_logit_Y_real = self.classifier_Y(self.source_Y, self.length_Y)
        _, self.latent_Y_fake, self.cls_logit_Y_fake = self.classifier_Y(self.G, self.length_X, reuse=True)

        self.batch_acc_X = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(self.cls_logit_X_real), axis=1), tf.argmax(self.label_X, axis=1))))
        self.batch_acc_Y_real = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(self.cls_logit_Y_real), axis=1), tf.argmax(self.label_Y, axis=1))))
        self.batch_acc_Y_fake = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(self.cls_logit_Y_fake), axis=1), tf.argmax(self.label_X, axis=1))))

    def build_dtn(self):
        pass

    def build_ae(self):
        self.latent_X_real, self.ae_logit_X_real = self.autoencoder_X(self.source_X, self.length_X)
        self.latent_Y_real, self.ae_logit_Y_real = self.autoencoder_Y(self.source_Y, self.length_Y)
        self.latent_Y_fake, self.ae_logit_Y_fake = self.autoencoder_Y(self.G, self.length_X, reuse=True)
        
    def build_gan(self):
        self.D_logit_real, _ = self.discriminator(self.source_Y, self.length_Y)
        self.D_logit_fake, _ = self.discriminator(self.G, self.length_X, reuse=True)

    def build_optimizer(self):
        '''cls'''
        self.C_loss_X = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_X, logits=self.cls_logit_X_real))
        self.C_loss_Y_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_Y, logits=self.cls_logit_Y_real))
        self.C_loss_Y_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_X, logits=self.cls_logit_Y_fake))
        self.C_solver_X = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.C_loss_X, var_list=self.classifier_X.vars)
        self.C_solver_Y_real = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.C_loss_Y_real, var_list=self.classifier_Y.vars)
        self.C_solver_Y_fake = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.C_loss_Y_fake, var_list=self.encoder.vars)

        '''ae'''
        # self.ae_loss_X = tf.reduce_mean(tf.square(self.target_X - self.ae_logit_X_real))
        # self.ae_loss_Y_real = tf.reduce_mean(tf.square(self.target_Y - self.ae_logit_Y_real))
        # self.ae_loss_Y_fake = tf.reduce_mean(tf.square(self.G - self.ae_logit_Y_fake))
        # self.ae_X_solver = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.ae_loss_X, var_list=self.autoencoder_X.vars)
        # self.ae_Y_solver_real = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.ae_loss_Y_real, var_list=self.autoencoder_Y.vars)
        # self.ae_Y_solver_fake = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.ae_loss_Y_fake, var_list=self.encoder.vars)
        
        '''DTN'''
        self.DTN_loss = tf.reduce_mean(tf.square(self.latent_X_real - self.latent_Y_fake))
        self.DTN_solver = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.DTN_loss, var_list=self.encoder.vars + self.classifier_X.vars + self.classifier_Y.vars)
        # self.DTN_solver = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.DTN_loss, var_list=self.encoder.vars)

        '''gan'''
        D_loss_real = - tf.reduce_mean(self.D_logit_real)
        D_loss_fake = tf.reduce_mean(self.D_logit_fake)
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = - D_loss_fake
        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.D_loss, var_list=self.encoder.vars)
        self.D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.discriminator.vars]
    
    def build_summary(self):
        C_loss_X_sum = tf.summary.scalar("C_loss_X", self.C_loss_X)
        C_loss_Y_real_sum = tf.summary.scalar("C_loss_Y_real", self.C_loss_Y_real)
        C_loss_Y_fake_sum = tf.summary.scalar("C_loss_Y_fake", self.C_loss_Y_fake)
        acc_X_sum = tf.summary.scalar("batch_acc_X", self.batch_acc_X)
        acc_Y_real_sum = tf.summary.scalar("batch_acc_Y_real", self.batch_acc_Y_real)
        acc_Y_fake_sum = tf.summary.scalar("batch_acc_Y_fake", self.batch_acc_Y_fake)
        # ae_loss_X_sum = tf.summary.scalar("ae_loss_X", self.ae_loss_X)
        # ae_loss_Y_real_sum = tf.summary.scalar("ae_loss_Y_real", self.ae_loss_Y_real)
        # ae_loss_Y_fake_sum = tf.summary.scalar("ae_loss_Y_fake", self.ae_loss_Y_fake)
        DTN_loss_sum = tf.summary.scalar("DTN_loss", self.DTN_loss)
        D_loss_sum = tf.summary.scalar("D_loss", self.D_loss)
        G_loss_sum = tf.summary.scalar("G_loss", self.G_loss)
        self.sum = tf.summary.merge([
                                        C_loss_X_sum, C_loss_Y_real_sum, C_loss_Y_fake_sum, 
                                        # ae_loss_X_sum, ae_loss_Y_real_sum, ae_loss_Y_fake_sum,
                                        DTN_loss_sum,
                                        acc_X_sum, acc_Y_real_sum, acc_Y_fake_sum,
                                        D_loss_sum, G_loss_sum])
    
    def test(self):
        train_acc, test_acc = 0, 0
        for i in range(107):
            data = self.data_X(self.batch_size, phase='train', var_list=["AccGyo", "lens", "labels"])
            feed_dict={
                self.source_X: data["AccGyo"][:,:,:,None],
                self.length_X: data["lens"],
                self.label_X: data["labels"]
            }
            train_acc += self.batch_size * self.sess.run(self.batch_acc_X, feed_dict=feed_dict)
        train_acc /= float(107 * self.batch_size)
        for i in range(26):
            data = self.data_X(self.batch_size, phase='test', var_list=["AccGyo", "lens", "labels"])
            feed_dict={
                self.source_X: data["AccGyo"][:,:,:,None],
                self.length_X: data["lens"],
                self.label_X: data["labels"]
            }
            test_acc += self.batch_size * self.sess.run(self.batch_acc_X, feed_dict=feed_dict)
        test_acc /= float(26 * self.batch_size)
        print "X train_acc=%.5f test_acc=%.5f" % (train_acc, test_acc)

        train_acc, test_acc = 0, 0
        for i in range(107):
            data = self.data_Y(self.batch_size, phase='train', var_list=["XYZ", "lens", "labels"])
            feed_dict={
                self.source_Y: data["XYZ"][:,:,:,None],
                self.length_Y: data["lens"],
                self.label_Y: data["labels"]
            }
            train_acc += self.batch_size * self.sess.run(self.batch_acc_Y_real, feed_dict=feed_dict)
        train_acc /= float(107 * self.batch_size)
        for i in range(26):
            data = self.data_Y(self.batch_size, phase='test', var_list=["XYZ", "lens", "labels"])
            feed_dict={
                self.source_Y: data["XYZ"][:,:,:,None],
                self.length_Y: data["lens"],
                self.label_Y: data["labels"]
            }
            test_acc += self.batch_size * self.sess.run(self.batch_acc_Y_real, feed_dict=feed_dict)
        test_acc /= float(26 * self.batch_size)
        print "YR train_acc=%.5f test_acc=%.5f" % (train_acc, test_acc)

        train_acc, test_acc = 0, 0
        for i in range(107):
            data = self.data_X(self.batch_size, phase='train', var_list=["AccGyo", "lens", "labels"])
            feed_dict={
                self.source_X: data["AccGyo"][:,:,:,None],
                self.length_X: data["lens"],
                self.label_X: data["labels"]
            }
            train_acc += self.batch_size * self.sess.run(self.batch_acc_Y_fake, feed_dict=feed_dict)
        train_acc /= float(107 * self.batch_size)
        for i in range(26):
            data = self.data_X(self.batch_size, phase='test', var_list=["AccGyo", "lens", "labels"])
            feed_dict={
                self.source_X: data["AccGyo"][:,:,:,None],
                self.length_X: data["lens"],
                self.label_X: data["labels"]
            }
            test_acc += self.batch_size * self.sess.run(self.batch_acc_Y_fake, feed_dict=feed_dict)
        test_acc /= float(26 * self.batch_size)
        print "YF train_acc=%.5f test_acc=%.5f" % (train_acc, test_acc)

    def sample(self, epoch):
        data = datamanager(time_major=False, seed=0)
        data.shuffle_train(0)
        X = data(self.batch_size, var_list=["AccGyo", "lens", "labels", "XYZ"])

        feed_dict = {
            self.source_X : X["AccGyo"][:,:,:,None],
            self.length_X : X["lens"]
        }

        logit = self.sess.run(self.G, feed_dict=feed_dict)
        logit = np.array([filtering(seq, window=5) for seq in logit])

        ori, target, lens = X["AccGyo"], X["XYZ"], X['lens']

        for i in range(4):
            idx = i*3
            plt.subplot(4, 3, 3*i+1)
            plt.plot(ori[idx, :lens[idx], :])
            plt.subplot(4, 3, 3*i+2)
            # plt.plot(logit[idx, :lens[idx], :, 0])
            plt.plot(logit[idx, :lens[idx], 0, 0], logit[idx, :lens[idx], 1, 0]) 
            plt.xticks([])
            plt.title(str(np.argmax(X["labels"][idx])))
            plt.subplot(4, 3, 3*i+3)
            # plt.plot(target[idx, :lens[idx], :])
            plt.plot(target[idx, :lens[idx], 0], target[idx, :lens[idx], 1])
            plt.xticks([])

        plt.savefig(os.path.join(self.fig_dir, "sample_epoch_{}.png".format(epoch)))
        plt.clf()


    def train(self,epoches=10):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        batches_per_epoch = self.data_X.train_num // self.batch_size

        for epoch in range(epoches):
            self.data_X.shuffle_train()
            self.data_Y.shuffle_train()
            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data_X = self.data_X(self.batch_size, var_list=["AccGyo", "lens", "labels"])
                data_Y = self.data_Y(self.batch_size, var_list=["XYZ", "lens", "labels"])

                feed_dict = {
                    self.source_X: data_X["AccGyo"][:,:,:,None],
                    self.length_X: data_X["lens"],
                    # self.label_X: data_X["labels"],
                    self.target_X: data_X["AccGyo"][:,:,:,None],

                    self.source_Y: data_Y["XYZ"][:,:,:,None],
                    self.length_Y: data_Y["lens"],
                    # self.label_Y: data_Y["labels"],
                    self.target_Y: data_Y["XYZ"][:,:,:,None],
                }

                # self.sess.run([self.C_solver_X, self.C_solver_Y_real, self.C_solver_Y_fake, self.DTN_solver], feed_dict=feed_dict)
                # self.sess.run(self.C_solver_X, feed_dict=feed_dict)
                # self.sess.run(self.C_solver_Y_real, feed_dict=feed_dict)
                # self.sess.run(self.C_solver_Y_fake, feed_dict=feed_dict)
                self.sess.run([self.ae_X_solver, self.ae_Y_solver_real, self.ae_Y_solver_fake], feed_dict=feed_dict)
                for _ in range(5):
                    self.sess.run(self.DTN_solver, feed_dict=feed_dict)
                # self.sess.run([self.D_solver, self.D_clip, self.G_solver], feed_dict=feed_dict)

                if cnt % 10 == 0:
                    # c_loss_x, c_loss_y_r, c_loss_y_f, dtn_loss, d_loss, g_loss, sum_str = \
                    #     self.sess.run([self.C_loss_X, self.C_loss_Y_real, self.C_loss_Y_fake, self.DTN_loss, self.D_loss, self.G_loss, self.sum], feed_dict=feed_dict)
                    # print self.version+" Epoch [%3d] [%3d/%3d] X=%.4f Yr=%.4f Yf=%.4f DTN=%.4f D=%.4f G=%.4f" % \
                    #     (epoch, idx, batches_per_epoch, c_loss_x, c_loss_y_r, c_loss_y_f, dtn_loss, d_loss, g_loss) 
                    ae_loss_x, ae_loss_y_r, ae_loss_y_f, dtn_loss, d_loss, g_loss, sum_str = \
                        self.sess.run([self.ae_loss_X, self.ae_loss_Y_real, self.ae_loss_Y_fake, self.DTN_loss, self.D_loss, self.G_loss, self.sum], feed_dict=feed_dict)
                    print self.version+" Epoch [%3d] [%3d/%3d] X=%.4f Yr=%.4f Yf=%.4f DTN=%.4f D=%.4f G=%.4f" % \
                        (epoch, idx, batches_per_epoch, ae_loss_x, ae_loss_y_r, ae_loss_y_f, dtn_loss, d_loss, g_loss)
                    self.writer.add_summary(sum_str, cnt)
                    # print np.argmax(data_X["labels"], axis=1)
            
            # self.test()
            
            # if epoch % 10 == 0:
            if epoch<=20 or epoch%20==0:
                self.sample(epoch)
            print
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)

dtn = DTN(64, "DTN")
# dtn.train(200)
dtn.saver.restore(dtn.sess, os.path.join(dtn.model_dir, "model.ckpt-10699"))
# dtn.test()
dtn.sample(200)

# vs = tf.trainable_variables()
# for v in vs:
#     print v