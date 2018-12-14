# coding:utf-8
import tensorflow as tf
from utils import BasicTrainFramework, dense, logging_time
from datamanager import datamanager
from encoders import encoder_bi, encoder_naive
from autoencoder import ConverterA_CNN, ConverterB_CNN
from discriminators import SeqDiscriminator_CNN
import time, os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Double_Cycle(BasicTrainFramework):
    def __init__(self, 
        encoder, en_hidden_units,
        decoder, de_hidden_units, 
        data_X, data_Y,
        batch_size,
        version):
        super(Double_Cycle, self).__init__(batch_size, version)

        self.enX_hidden_units = en_hidden_units
        self.deX_hidden_units = de_hidden_units

        self.encoder = encoder(en_hidden_units, name='encoder')
        self.decoder = decoder(de_hidden_units, name='decoder')

        self.data_X = data_X
        self.data_Y = data_Y

        self.build_network()

        self.build_sess()
        self.build_dirs()
    
    def build_network(self):
        self.build_placeholder()
        self.build_ae()
        self.build_optimizer()
        self.build_summary()

    def build_placeholder(self):
        with tf.variable_scope("placeholders"):
            self.source_X = tf.placeholder(shape=(None, self.batch_size, 6), dtype=tf.float32)
            self.len_X = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
            self.target_X = tf.placeholder(shape=(None, self.batch_size, 6), dtype=tf.float32)
            self.label_X = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)

            self.source_Y = tf.placeholder(shape=(None, self.batch_size, 3), dtype=tf.float32)
            self.len_Y = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
            self.target_Y = tf.placeholder(shape=(None, self.batch_size, 3), dtype=tf.float32)
            self.label_Y = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)
    
    def build_ae(self):
        # X -> Y
        self.en_outputs_1, _ = self.encoder(self.source_X, self.len_X)
        self.en_outputs_1 = tf.reshape(self.en_outputs_1, (-1, self.en_outputs_1.get_shape().as_list()[-1]))
        self.en_outputs_1 = dense(self.en_outputs_1, 3, name='encoder_dense')
        self.en_outputs_1 = tf.reshape(self.en_outputs_1, (-1, self.batch_size, 3))
        # Y -> X
        self.de_outputs_1, _ = self.decoder(self.en_outputs_1, self.len_X)
        self.de_outputs_1 = tf.reshape(self.de_outputs_1, (-1, self.de_outputs_1.get_shape().as_list()[-1]))
        self.de_outputs_1 = dense(self.de_outputs_1, 6, name='decoder_dense')
        self.de_outputs_1 = tf.reshape(self.de_outputs_1, (-1, self.batch_size, 6))

        # Y -> X
        self.de_outputs_2, _ = self.decoder(self.source_Y, self.len_Y)
        self.de_outputs_2 = tf.reshape(self.de_outputs_2, (-1, self.de_outputs_2.get_shape().as_list()[-1]))
        self.de_outputs_2 = dense(self.de_outputs_2, 6, name="decoder_dense", reuse=True)
        self.de_outputs_2 = tf.reshape(self.de_outputs_2, (-1, self.batch_size, 6))
        # X -> Y
        self.en_outputs_2, _ = self.encoder(self.de_outputs_2, self.len_Y)
        self.en_outputs_2 = tf.reshape(self.en_outputs_2, (-1, self.en_outputs_2.get_shape().as_list()[-1]))
        self.en_outputs_2 = dense(self.en_outputs_2, 3, name='encoder_dense', reuse=True)
        self.en_outputs_2 = tf.reshape(self.en_outputs_2, (-1, self.batch_size, 3))
    
    def build_optimizer(self):
        # cycle consistency
        self.ae_loss_1 = tf.sqrt(tf.reduce_mean(tf.square(self.target_X - self.de_outputs_1)))
        self.ae_loss_2 = tf.sqrt(tf.reduce_mean(tf.square(self.target_Y - self.en_outputs_2)))
        self.ae_loss = self.ae_loss_1 + self.ae_loss_2

        # solvers 
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_solver_1 = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.ae_loss_1, var_list=self.encoder.vars + self.decoder.vars)
            self.ae_solver_2 = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.ae_loss_2, var_list=self.encoder.vars + self.decoder.vars)
            
    def build_summary(self):
        ae_loss_sum = tf.summary.scalar("ae_loss", self.ae_loss)
        ae_loss_1_sum = tf.summary.scalar("ae_loss_1", self.ae_loss_1)
        ae_loss_2_sum = tf.summary.scalar("ae_loss_2", self.ae_loss_2)
        self.ae_sum = tf.summary.merge([ae_loss_sum, ae_loss_1_sum, ae_loss_2_sum])

    def sample(self, epoch):
        print "sample after epoch {}".format(epoch)
        data = datamanager(seed=19931028)
        data_X = data(self.batch_size, phase='train', var_list=["AccGyo", "lens", "labels", "XYZ"])
        feed_dict={
            self.source_X:data_X['AccGyo'],
            self.len_X:data_X['lens']
        }

        data_X['enX_outputs'], data_X['deX_outputs'] = self.sess.run([self.en_outputs_1, self.de_outputs_1], feed_dict=feed_dict)
        np.savez(os.path.join(self.fig_dir, "sample_epoch_{}.npz".format(epoch)), **data_X)

    def train(self, epoches=10):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        batches_per_epoch = self.data_X.train_num // self.batch_size

        # restore checkpoint
        cound_load, checkpoint_counter = self.load_model(self.model_dir)
        if cound_load:
            start_epoch = (int)(checkpoint_counter / batches_per_epoch)
            start_batch_id = checkpoint_counter - start_epoch * batches_per_epoch
            cnt = checkpoint_counter
        else:
            start_epoch, start_batch_id, cnt = 0,0,0
        
        start_time = time.time()
        for epoch in range(start_epoch, epoches + start_epoch):
            self.data_X.shuffle_train(seed=epoch)
            self.data_Y.shuffle_train(seed=epoch)

            for iteration in range(start_batch_id, batches_per_epoch):

                data_X = self.data_X(self.batch_size, phase='train', var_list=["AccGyo", "lens"])
                data_Y = self.data_Y(self.batch_size, phase='train', var_list=["XYZ", "lens"])

                feed_dict = {
                    self.source_X:data_X["AccGyo"],
                    self.len_X:data_X["lens"],
                    self.target_X:data_X["AccGyo"],
                    self.source_Y:data_Y["XYZ"],
                    self.len_Y:data_Y["lens"],
                    self.target_Y:data_Y["XYZ"]
                }

                # update cycle consistency
                self.sess.run(self.ae_solver_1, feed_dict=feed_dict)
                self.sess.run(self.ae_solver_2, feed_dict=feed_dict)

                if (cnt - 1) % 5 == 0:
                    ae_sum, ae_loss_1, ae_loss_2 = self.sess.run([self.ae_sum, self.ae_loss_1, self.ae_loss_2], feed_dict=feed_dict)
                    self.writer.add_summary(ae_sum, cnt)
                    h,m,s = logging_time(time.time() - start_time)
                    print "Epoch: [%2d] [%4d/%4d] time: %3d:%3d:%3d, ae_loss_1:%.5f ae_loss_2:%.5f" \
                        % (epoch, iteration, batches_per_epoch, h,m,s, ae_loss_1, ae_loss_2)
                
                cnt += 1
            
            start_batch_id = 0

            self.saver.save(self.sess, os.path.join(self.model_dir, 'DoubleCycle.ckpt'), global_step=cnt)
            self.sample(epoch)  


class Double_CycleGAN(BasicTrainFramework):
    def __init__(self, 
        encoder, en_hidden_units,
        decoder, de_hidden_units, 
        discriminator, D_hidden_units,
        data_X, data_Y,
        batch_size,
        version):
        super(Double_CycleGAN, self).__init__(batch_size, version)

        self.enX_hidden_units = en_hidden_units
        self.deX_hidden_units = de_hidden_units
        self.D_hidden_units = D_hidden_units

        self.encoder = encoder(en_hidden_units, name='encoder')
        self.decoder = decoder(de_hidden_units, name='decoder')
        self.discriminator = discriminator(D_hidden_units, name='discriminator')

        self.data_X = data_X
        self.data_Y = data_Y

        self.build_network()

        self.build_sess()
        self.build_dirs()
    
    def build_network(self):
        self.build_placeholder()
        self.build_ae()
        self.build_gan()
        self.build_optimizer()
        self.build_summary()

    def build_placeholder(self):
        with tf.variable_scope("placeholders"):
            self.source_X = tf.placeholder(shape=(None, self.batch_size, 6), dtype=tf.float32)
            self.len_X = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
            self.target_X = tf.placeholder(shape=(None, self.batch_size, 6), dtype=tf.float32)
            self.label_X = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)

            self.source_Y = tf.placeholder(shape=(None, self.batch_size, 3), dtype=tf.float32)
            self.len_Y = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
            self.target_Y = tf.placeholder(shape=(None, self.batch_size, 3), dtype=tf.float32)
            self.label_Y = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)
    
    def build_ae(self):
        # X -> Y
        self.en_outputs_1, _ = self.encoder(self.source_X, self.len_X)
        self.en_outputs_1 = tf.reshape(self.en_outputs_1, (-1, self.en_outputs_1.get_shape().as_list()[-1]))
        self.en_outputs_1 = dense(self.en_outputs_1, 3, name='encoder_dense')
        self.en_outputs_1 = tf.reshape(self.en_outputs_1, (-1, self.batch_size, 3))
        # Y -> X
        self.de_outputs_1, _ = self.decoder(self.en_outputs_1, self.len_X)
        self.de_outputs_1 = tf.reshape(self.de_outputs_1, (-1, self.de_outputs_1.get_shape().as_list()[-1]))
        self.de_outputs_1 = dense(self.de_outputs_1, 6, name='decoder_dense')
        self.de_outputs_1 = tf.reshape(self.de_outputs_1, (-1, self.batch_size, 6))

        # Y -> X
        self.de_outputs_2, _ = self.decoder(self.source_Y, self.len_Y)
        self.de_outputs_2 = tf.reshape(self.de_outputs_2, (-1, self.de_outputs_2.get_shape().as_list()[-1]))
        self.de_outputs_2 = dense(self.de_outputs_2, 6, name="decoder_dense", reuse=True)
        self.de_outputs_2 = tf.reshape(self.de_outputs_2, (-1, self.batch_size, 6))
        # X -> Y
        self.en_outputs_2, _ = self.encoder(self.de_outputs_2, self.len_Y)
        self.en_outputs_2 = tf.reshape(self.en_outputs_2, (-1, self.en_outputs_2.get_shape().as_list()[-1]))
        self.en_outputs_2 = dense(self.en_outputs_2, 3, name='encoder_dense', reuse=True)
        self.en_outputs_2 = tf.reshape(self.en_outputs_2, (-1, self.batch_size, 3))
    
    def build_gan(self):
        _, D_state_real = self.discriminator(self.source_Y, self.len_Y, reuse=False)
        _, D_state_fake = self.discriminator(self.en_outputs_1, self.len_X, reuse=True)

        self.D_logit_real = dense(D_state_real[-1].c, 1, name='D_dense')
        self.D_logit_fake = dense(D_state_fake[-1].c, 1, name='D_dense', reuse=True)
    
    def build_optimizer(self):
        # cycle consistency
        self.ae_loss_1 = tf.sqrt(tf.reduce_mean(tf.square(self.target_X - self.de_outputs_1)))
        self.ae_loss_2 = tf.sqrt(tf.reduce_mean(tf.square(self.target_Y - self.en_outputs_2)))
        self.ae_loss = self.ae_loss_1 + self.ae_loss_2

        # wgan
        self.D_loss_real = - tf.reduce_mean(self.D_logit_real)
        self.D_loss_fake = tf.reduce_mean(self.D_logit_fake)
        self.D_loss = self.D_loss_real + self.D_loss_fake 
        self.G_loss = - self.D_loss_fake 

        # weights clipping
        self.D_clip = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in self.discriminator.vars]

        # solvers 
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(self.ae_loss, var_list=self.encoder.vars + self.decoder.vars)
            self.D_solver = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(self.G_loss, var_list=self.encoder.vars)
    
    def build_summary(self):
        ae_loss_sum = tf.summary.scalar("ae_loss", self.ae_loss)
        ae_loss_1_sum = tf.summary.scalar("ae_loss_1", self.ae_loss_1)
        ae_loss_2_sum = tf.summary.scalar("ae_loss_2", self.ae_loss_2)
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.D_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.D_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.D_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)

        self.ae_sum = tf.summary.merge([ae_loss_sum, ae_loss_1_sum, ae_loss_2_sum])
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
    
    def sample(self, epoch):
        print "sample after epoch {}".format(epoch)
        data = datamanager(seed=19931028)
        data_X = data(self.batch_size, phase='train', var_list=["AccGyo", "lens", "labels", "XYZ"])
        feed_dict={
            self.source_X:data_X['AccGyo'],
            self.len_X:data_X['lens']
        }

        data_X['enX_outputs'], data_X['deX_outputs'] = self.sess.run([self.en_outputs_1, self.de_outputs_1], feed_dict=feed_dict)
        np.savez(os.path.join(self.fig_dir, "sample_epoch_{}.npz".format(epoch)), **data_X)

    def train(self, epoches=10):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        batches_per_epoch = self.data_X.train_num // self.batch_size

        # restore checkpoint
        cound_load, checkpoint_counter = self.load_model(self.model_dir)
        if cound_load:
            start_epoch = (int)(checkpoint_counter / batches_per_epoch)
            start_batch_id = checkpoint_counter - start_epoch * batches_per_epoch
            cnt = checkpoint_counter
        else:
            start_epoch, start_batch_id, cnt = 0,0,0
        
        start_time = time.time()
        for epoch in range(start_epoch, epoches + start_epoch):
            self.data_X.shuffle_train(seed=epoch)
            self.data_Y.shuffle_train(seed=epoch)

            for iteration in range(start_batch_id, batches_per_epoch):

                data_X = self.data_X(self.batch_size, phase='train', var_list=["AccGyo", "lens"])
                data_Y = self.data_Y(self.batch_size, phase='train', var_list=["XYZ", "lens"])

                feed_dict = {
                    self.source_X:data_X["AccGyo"],
                    self.len_X:data_X["lens"],
                    self.target_X:data_X["AccGyo"],
                    self.source_Y:data_Y["XYZ"],
                    self.len_Y:data_Y["lens"],
                    self.target_Y:data_Y["XYZ"]
                }

                # update cycle consistency & D network
                self.sess.run([self.ae_solver, self.D_solver, self.D_clip], feed_dict=feed_dict)

                # update G network
                if (cnt - 1) % 1 == 0:
                    self.sess.run(self.G_solver, feed_dict=feed_dict)
                
                if (cnt - 1) % 5 == 0:
                    ae_sum, d_sum, g_sum, g_loss, d_loss, ae_loss_1, ae_loss_2 = self.sess.run([self.ae_sum, self.d_sum, self.g_sum, self.G_loss, self.D_loss, self.ae_loss_1, self.ae_loss_2], feed_dict=feed_dict)
                    self.writer.add_summary(ae_sum, cnt)
                    self.writer.add_summary(d_sum, cnt)
                    self.writer.add_summary(g_sum, cnt)
                    h,m,s = logging_time(time.time() - start_time)
                    print "Epoch: [%2d] [%4d/%4d] time: %3d:%3d:%3d, d_loss:%.5f, g_loss:%.5f, ae_loss_1:%.5f ae_loss_2:%.5f" \
                        % (epoch, iteration, batches_per_epoch, h,m,s, d_loss, g_loss, ae_loss_1, ae_loss_2)
                
                if (cnt - 1) % 300 == 0:
                    pass
                
                cnt += 1
            
            start_batch_id = 0

            self.saver.save(self.sess, os.path.join(self.model_dir, 'CycleGAN.ckpt'), global_step=cnt)
            self.sample(epoch)  


class CycleGAN(BasicTrainFramework):
    def __init__(self, 
        encoder_X, enX_hidden_units,
        decoder_X, deX_hidden_units, 
        discriminator, D_hidden_units,
        data_X, data_Y,
        batch_size,
        version):
        super(CycleGAN, self).__init__(batch_size, version)

        self.enX_hidden_units = enX_hidden_units
        self.deX_hidden_units = deX_hidden_units
        self.D_hidden_units = D_hidden_units

        self.encoder_X = encoder_X(enX_hidden_units, name='encoder_X')
        self.decoder_X = decoder_X(deX_hidden_units, name='decoder_X')
        self.discriminator = discriminator(D_hidden_units, name='discriminator')

        self.data_X = data_X
        self.data_Y = data_Y

        self.build_network()

        self.build_sess()
        self.build_dirs()
    
    def build_network(self):
        self.build_placeholder()
        self.build_ae()
        self.build_gan()
        self.build_optimizer()
        self.build_summary()
    
    def build_placeholder(self):
        with tf.variable_scope("placeholders"):
            self.source_X = tf.placeholder(shape=(None, self.batch_size, 6), dtype=tf.float32)
            self.len_X = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
            self.target_X = tf.placeholder(shape=(None, self.batch_size, 6), dtype=tf.float32)
            self.label_X = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)

            self.source_Y = tf.placeholder(shape=(None, self.batch_size, 3), dtype=tf.float32)
            self.len_Y = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
        
    def build_ae(self):
        self.enX_outputs, _ = self.encoder_X(self.source_X, self.len_X)
        self.enX_outputs = tf.reshape(self.enX_outputs, (-1, self.enX_outputs.get_shape().as_list()[-1]))
        self.enX_outputs = dense(self.enX_outputs, 3, name='encoder_X_output')
        self.enX_outputs = tf.reshape(self.enX_outputs, (-1, self.batch_size, 3))

        self.deX_outputs, _ = self.decoder_X(self.enX_outputs, self.len_X)
        self.deX_outputs = tf.reshape(self.deX_outputs, (-1, self.deX_outputs.get_shape().as_list()[-1]))
        self.deX_outputs = dense(self.deX_outputs, 6, name='decoder_X_output')
        self.deX_outputs = tf.reshape(self.deX_outputs, (-1, self.batch_size, 6))
    
    def build_gan(self):
        print 'source_Y', self.source_Y.get_shape().as_list()
        print 'enX_outputs', self.enX_outputs.get_shape().as_list()
        _, D_state_real = self.discriminator(self.source_Y, self.len_Y, reuse=False)
        _, D_state_fake = self.discriminator(self.enX_outputs, self.len_X, reuse=True)

        self.D_logit_real = dense(D_state_real[-1].c, 1, name="D_dense")
        self.D_logit_fake = dense(D_state_fake[-1].c, 1, name="D_dense", reuse=True)

    def build_optimizer(self):
        # cycle consistency
        self.ae_loss = tf.sqrt(tf.reduce_mean(tf.square(self.target_X - self.deX_outputs)))
        
        # wgan
        self.D_loss_real = - tf.reduce_mean(self.D_logit_real)
        self.D_loss_fake = tf.reduce_mean(self.D_logit_fake) 
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = - self.D_loss_fake

        # weights clipping
        self.D_clip = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in self.discriminator.vars]

        # solvers 
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.ae_loss, var_list=self.encoder_X.vars + self.decoder_X.vars)
            self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(self.G_loss, var_list=self.encoder_X.vars)

    def build_summary(self):
        ae_loss_sum = tf.summary.scalar("ae_loss", self.ae_loss)
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.D_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.D_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.D_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)

        self.ae_sum = tf.summary.merge([ae_loss_sum])
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
    
    def sample(self, epoch):
        print "sample after epoch {}".format(epoch)
        data = datamanager(seed=19931028)
        data_X = data(self.batch_size, phase='train', var_list=["AccGyo", "lens", "labels", "XYZ"])
        feed_dict={
            self.source_X:data_X['AccGyo'],
            self.len_X:data_X['lens']
        }

        data_X['enX_outputs'], data_X['deX_outputs'] = self.sess.run([self.enX_outputs, self.deX_outputs], feed_dict=feed_dict)
        np.savez(os.path.join(self.fig_dir, "sample_epoch_{}.npz".format(epoch)), **data_X)

    def train(self, epoches=10):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        batches_per_epoch = self.data_X.train_num // self.batch_size

        # restore checkpoint
        cound_load, checkpoint_counter = self.load_model(self.model_dir)
        if cound_load:
            start_epoch = (int)(checkpoint_counter / batches_per_epoch)
            start_batch_id = checkpoint_counter - start_epoch * batches_per_epoch
            cnt = checkpoint_counter
        else:
            start_epoch, start_batch_id, cnt = 0,0,0
        
        start_time = time.time()
        for epoch in range(start_epoch, epoches + start_epoch):
            self.data_X.shuffle_train(seed=epoch)
            self.data_Y.shuffle_train(seed=epoch)

            for iteration in range(start_batch_id, batches_per_epoch):

                data_X = self.data_X(self.batch_size, phase='train', var_list=["AccGyo", "lens", "labels"])
                data_Y = self.data_Y(self.batch_size, phase='train', var_list=["XYZ", "lens"])

                feed_dict = {
                    self.source_X:data_X["AccGyo"],
                    self.len_X:data_X["lens"],
                    self.target_X:data_X["AccGyo"],
                    self.source_Y:data_Y["XYZ"],
                    self.len_Y:data_Y["lens"]
                }

                # update cycle consistency & D network
                self.sess.run([self.ae_solver, self.D_solver, self.D_clip], feed_dict=feed_dict)

                # update G network
                if (cnt - 1) % 1 == 0:
                    self.sess.run(self.G_solver, feed_dict=feed_dict)
                
                if (cnt - 1) % 5 == 0:
                    ae_sum, d_sum, g_sum, g_loss, d_loss, ae_loss = self.sess.run([self.ae_sum, self.d_sum, self.g_sum, self.G_loss, self.D_loss, self.ae_loss], feed_dict=feed_dict)
                    self.writer.add_summary(ae_sum, cnt)
                    self.writer.add_summary(d_sum, cnt)
                    self.writer.add_summary(g_sum, cnt)
                    h,m,s = logging_time(time.time() - start_time)
                    print "Epoch: [%2d] [%4d/%4d] time: %3d:%3d:%3d, d_loss:%.5f, g_loss:%.5f, ae_loss:%.5f" \
                        % (epoch, iteration, batches_per_epoch, h,m,s, d_loss, g_loss, ae_loss)
                
                if (cnt - 1) % 300 == 0:
                    pass
                
                cnt += 1
            
            start_batch_id = 0

            self.saver.save(self.sess, os.path.join(self.model_dir, 'CycleGAN.ckpt'), global_step=cnt)
            self.sample(epoch)  

class CycleGAN_CNN(BasicTrainFramework):
    def __init__(self, batch_size, version):
        super(CycleGAN_CNN, self).__init__(batch_size, version)

        self.encoder_X = ConverterA_CNN(name="encoder_cnn")
        self.decoder_X = ConverterB_CNN(name="decoder_cnn")
        self.discriminator = SeqDiscriminator_CNN(class_num=None, mode=1, fixed_length=False, name='discriminator')

        self.data_X = datamanager(time_major=False, seed=19940610)
        self.data_Y = datamanager(time_major=False, seed=19940610)
        self.data_Y.shuffle_train(seed=19931028)

        self.build_network()

        self.build_sess()
        self.build_dirs()
    
    def build_network(self):
        self.build_placeholder()
        self.build_ae()
        self.build_gan()
        self.build_optimizer()
        self.build_summary()
    
    def build_placeholder(self):
        with tf.variable_scope("placeholders"):
            self.source_X = tf.placeholder(shape=(self.batch_size, None, 6, 1), dtype=tf.float32)
            self.len_X = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
            self.target_X = tf.placeholder(shape=(self.batch_size, None, 6, 1), dtype=tf.float32)
            # self.label_X = tf.placeholder(shape=(self.batch_size, 62), dtype=tf.float32)

            self.source_Y = tf.placeholder(shape=(self.batch_size, None, 3, 1), dtype=tf.float32)
            self.len_Y = tf.placeholder(shape=(self.batch_size, ), dtype=tf.int32)
        
    def build_ae(self):
        self.encoder_outputs = self.encoder_X(self.source_X, self.len_X)

        self.decoder_outputs = self.decoder_X(self.encoder_outputs, self.len_X)
        
    def build_gan(self):
        self.D_logit_real = self.discriminator(self.source_Y, self.len_Y, reuse=False)
        self.D_logit_fake = self.discriminator(self.encoder_outputs, self.len_X, reuse=True)
    
    def build_optimizer(self):
        # cycle consistency
        self.ae_loss = tf.reduce_mean(tf.square(self.target_X - self.decoder_outputs))
        
        # wgan
        self.D_loss_real = - tf.reduce_mean(self.D_logit_real)
        self.D_loss_fake = tf.reduce_mean(self.D_logit_fake) 
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = - self.D_loss_fake

        # weights clipping
        self.D_clip = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in self.discriminator.vars]

        # solvers 
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.ae_loss, var_list=self.encoder_X.vars + self.decoder_X.vars)
            self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss, var_list=self.discriminator.vars)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(self.G_loss, var_list=self.encoder_X.vars)

    def build_summary(self):
        ae_loss_sum = tf.summary.scalar("ae_loss", self.ae_loss)
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.D_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.D_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.D_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.G_loss)

        self.ae_sum = tf.summary.merge([ae_loss_sum])
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum]) 
    
    def sample(self, epoch):
        print "sample after epoch {}".format(epoch)
        data = datamanager(time_major=False,seed=19940610)
        data_X = data(self.batch_size, phase='train', var_list=["AccGyo", "lens", "labels", "XYZ"])
        feed_dict={
            self.source_X:data_X['AccGyo'][:,:,:,None],
            self.len_X:data_X['lens']
        }

        data_X['enX_outputs'], data_X['deX_outputs'] = self.sess.run([self.encoder_outputs, self.decoder_outputs], feed_dict=feed_dict)
        np.savez(os.path.join(self.fig_dir, "sample_epoch_{}.npz".format(epoch)), **data_X)

    def train(self, epoches=10):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        batches_per_epoch = self.data_X.train_num // self.batch_size

        start_time = time.time()
        for epoch in range(epoches):
            for idx in range(batches_per_epoch):
                cnt = batches_per_epoch * epoch + idx

                X = self.data_X(self.batch_size, var_list=["lens", "AccGyo"])
                Y = self.data_Y(self.batch_size, var_list=["lens", "XYZ"])

                feed_dict = {
                    self.source_X: X["AccGyo"][:,:,:,None],
                    self.len_X: X["lens"],
                    self.target_X: X["AccGyo"][:,:,:,None],

                    self.source_Y: Y["XYZ"][:,:,:,None],
                    self.len_Y: Y["lens"]
                }
                
                # update cycle consistency & D network
                self.sess.run([self.ae_solver, self.D_solver, self.D_clip], feed_dict=feed_dict)

                # update G network
                if (cnt - 1) % 1 == 0:
                    self.sess.run(self.G_solver, feed_dict=feed_dict)
                
                if (cnt - 1) % 5 == 0:
                    ae_sum, d_sum, g_sum, g_loss, d_loss, ae_loss = self.sess.run([self.ae_sum, self.d_sum, self.g_sum, self.G_loss, self.D_loss, self.ae_loss], feed_dict=feed_dict)
                    self.writer.add_summary(ae_sum, cnt)
                    self.writer.add_summary(d_sum, cnt)
                    self.writer.add_summary(g_sum, cnt)
                    h,m,s = logging_time(time.time() - start_time)
                    print "Epoch: [%2d] [%4d/%4d] time: %3d:%3d:%3d, d_loss:%.5f, g_loss:%.5f, ae_loss:%.5f" \
                        % (epoch, idx, batches_per_epoch, h,m,s, d_loss, g_loss, ae_loss)
            if epoch % 20 == 0:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'CycleGAN_CNN.ckpt'), global_step=cnt)
            if epoch % 10 == 0:
                self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'CycleGAN_CNN.ckpt'), global_step=cnt)

# c = CycleGAN(
#     encoder_naive, [128,],
#     encoder_naive, [128,],
#     encoder_bi, [64,],
#     datamanager(seed=0),
#     datamanager(seed=1),
#     64,
#     "test")
# c.train(100)

# dc = Double_CycleGAN(
#     encoder_naive, [128,],
#     encoder_naive, [128,],
#     encoder_bi, [64,],
#     datamanager(seed=0),
#     datamanager(seed=1),
#     64,
#     "Double_CycleGAN"
# )
# dc.train(epoches=100)

# dc = Double_Cycle(
#     encoder_naive, [128,],
#     encoder_naive, [128,],
#     datamanager(seed=0),
#     datamanager(seed=1),
#     64,
#     "Double_Cycle"
# )
# dc.train(epoches=20)

c = CycleGAN_CNN(64, "CycleGAN_CNN")

c.train(epoches=100)