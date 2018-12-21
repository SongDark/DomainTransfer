# coding:utf-8

import tensorflow as tf
import numpy as np
import os
from datamanager import datamanager
from discriminators import SeqDiscriminator_CNN
from utils import BasicTrainFramework
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Classifier(BasicTrainFramework):
    def __init__(self, class_num, data_dim, batch_size, version):
        super(Classifier, self).__init__(batch_size, version)
        self.data = datamanager(time_major=False)

        self.class_num = class_num
        self.data_dim = data_dim

        # CNN classifier
        self.classifier = SeqDiscriminator_CNN(class_num=class_num, mode=0, fixed_length=False, name="CNN_classifier")

        self.build_network()

        self.build_sess()
        self.build_dirs()
    
    @property
    def input_pointer(self):
        if self.data_dim == 3:
            return "XYZ"
        elif self.data_dim == 6:
            return "AccGyo"
    
    def build_network(self):
        self.build_placeholder()
        self.build_classifier()
        self.build_optimizer()
        self.build_summary()
    
    def build_placeholder(self):
        self.source = tf.placeholder(shape=(self.batch_size, None, self.data_dim, 1), dtype=tf.float32)
        self.length = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
        self.labels = tf.placeholder(shape=(self.batch_size, self.class_num), dtype=tf.float32)

    def build_classifier(self):
        _, self.embedding, self.logits = self.classifier(self.source, self.length)
        self.preds = tf.nn.softmax(self.logits)
    
    def build_optimizer(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits))
        self.batch_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.preds, axis=1), tf.argmax(self.labels, axis=1))))
        self.solver = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(self.loss, var_list=self.classifier.vars)
    
    def build_summary(self):
        self.train_sum = tf.summary.merge([tf.summary.scalar("train_loss", self.loss), tf.summary.scalar("train_acc", self.batch_acc)])
        self.test_sum = tf.summary.merge([tf.summary.scalar("test_loss", self.loss), tf.summary.scalar("test_acc", self.batch_acc)])
    
    def test(self):
        train_acc, test_acc = 0, 0
        for i in range(107):
            data = self.data(self.batch_size, phase='train', var_list=[self.input_pointer, "lens", "labels"])
            feed_dict={
                self.source: data[self.input_pointer][:,:,:,None],
                self.length: data["lens"],
                self.labels: data["labels"]
            }
            train_acc += self.batch_size * self.sess.run(self.batch_acc, feed_dict=feed_dict)
        train_acc /= float(107 * self.batch_size)
        for i in range(26):
            data = self.data(self.batch_size, phase='test', var_list=[self.input_pointer, "lens", "labels"])
            feed_dict={
                self.source: data[self.input_pointer][:,:,:,None],
                self.length: data["lens"],
                self.labels: data["labels"]
            }
            test_acc += self.batch_size * self.sess.run(self.batch_acc, feed_dict=feed_dict)
        test_acc /= float(26 * self.batch_size)
        print "train_acc=%.5f test_acc=%.5f" % (train_acc, test_acc)
    
    def gen_latent(self):
        data = datamanager(time_major=False)
        train_x, train_y = [],[]
        for _ in range(108):
            X = data(self.batch_size, phase='train', var_list=[self.input_pointer, "lens", "labels"])
            feed_dict={
                self.source: X[self.input_pointer][:,:,:,None],
                self.length: X["lens"]
            }
            train_x.append(self.sess.run(self.embedding, feed_dict=feed_dict))
            train_y.append(np.argmax(X["labels"], axis=1))
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y)[:data.train_num]
        
        test_x, test_y = [], []
        for _ in range(27):
            X = data(self.batch_size, phase='test', var_list=[self.input_pointer, "lens", "labels"])
            feed_dict={
                self.source: X[self.input_pointer][:,:,:,None],
                self.length: X["lens"]
            }
            test_x.append(self.sess.run(self.embedding, feed_dict=feed_dict))
            test_y.append(np.argmax(X["labels"], axis=1))
        test_x = np.concatenate(test_x, axis=0)[:data.test_num, :]
        test_y = np.concatenate(test_y)[:data.test_num]

        print train_x.shape, train_y.shape, test_x.shape, test_y.shape
        to_save = {
            "train_embedding":train_x,
            "train_label":train_y,
            "test_embedding":test_x,
            "test_label":test_y,
            "classifier_name": self.version,
            "data_seed":19940610
        }
        print to_save["classifier_name"]
        np.savez("/home/scw4750/songbinxu/autoencoder/data/"+self.version+"_emb.npz", **to_save)
        
        
    
    def sample(self, epoch):
        print "sample at {}".format(epoch)
        res, labels = [], []
        for i in range(107):
            data = self.data(self.batch_size, phase='train', var_list=[self.input_pointer, "lens", "labels"])
            feed_dict={
                self.source: data[self.input_pointer][:,:,:,None],
                self.length: data["lens"],
                self.labels: data["labels"]
            }
            emb = self.sess.run(self.embedding, feed_dict=feed_dict)
            res.append(emb)
            labels.append(np.argmax(data["labels"], axis=1))
        res = np.concatenate(res, axis=0)
        labels = np.concatenate(labels)
        to_save = {"embedding":res, "labels":labels}
        np.savez(os.path.join(self.fig_dir, "sample_{}.npz".format(epoch)), **to_save)

    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data = self.data(self.batch_size, var_list=[self.input_pointer, "lens", "labels"])

                feed_dict={
                    self.source: data[self.input_pointer][:,:,:,None],
                    self.length: data["lens"],
                    self.labels: data["labels"]
                }

                self.sess.run(self.solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    train_loss, train_acc, train_sum = self.sess.run([self.loss, self.batch_acc, self.train_sum], feed_dict=feed_dict)
                    self.writer.add_summary(train_sum, cnt)
                    # test
                    data = self.data(self.batch_size, phase="test", var_list=[self.input_pointer, "lens", "labels"])
                    feed_dict={
                        self.source: data[self.input_pointer][:,:,:,None],
                        self.length: data["lens"],
                        self.labels: data["labels"]
                    }
                    test_loss, test_acc, test_sum = self.sess.run([self.loss, self.batch_acc, self.test_sum], feed_dict=feed_dict)
                    self.writer.add_summary(test_sum, cnt)
                    
                    print "Epoch [%3d] [%3d/%3d] [trainloss=%.4f, testloss=%.4f] [trainacc=%.5f, testacc=%.5f]" \
                        % (epoch, idx, batches_per_epoch, train_loss, test_loss, train_acc, test_acc)
            self.test()
            
            if epoch % 50 == 0:
                self.sample(epoch)
                self.saver.save(self.sess, os.path.join(self.model_dir, 'Classifier_CNN.ckpt'), global_step=cnt)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'Classifier_CNN.ckpt'), global_step=cnt)
        

c = Classifier(class_num=62, data_dim=6, batch_size=64, version="Classifier_CNN_AccGyo_100")
c.train(100)
# c = Classifier(class_num=62, data_dim=3, batch_size=64, version="Classifier_CNN_XYZ")
# c.train(500)

# c = Classifier(class_num=62, data_dim=3, batch_size=64, version="Classifier_CNN_XYZ")
# c.saver.restore(c.sess, os.path.join(c.model_dir, "Classifier_CNN.ckpt-53499"))
# c.gen_latent()