# coding:utf-8
import tensorflow as tf
import os

def Seq2Seq_RNN(mode):
    from autoencoder import AutoEncoder_RNN, AE
    from encoders import encoder_bi
    from decoders import dynamic_decoder
    from datamanager import datamanager

    tmp = []
    for fold_id in range(5):
        autoencoder = AutoEncoder_RNN(
            encoder_bi, {"hidden_units":[64, 64], "cell_type":"gru"},
            dynamic_decoder, {"hidden_units":[64, 64], "cell_type":"gru"},
            input_depth=6, output_depth=3, embedding_dim=50,
            name="encoderdecoder_gru"
        )
        data = datamanager(time_major=True, expand_dim=False, train_ratio=None, fold_k=5, seed=233)
        ae = AE(autoencoder, data, 64, "Seq2Seq_GRU/fold_{}".format(fold_id))

        if mode=='train':
            ae.train(epoches=100)
        elif mode=='test':
            ae.saver.restore(ae.sess, os.path.join(ae.model_dir, "model.ckpt-10699"))
            a,b = ae.test()
            tmp.append([a,b])

        tf.reset_default_graph()
    print tmp

def Seq2Seq_CNN(mode):
    from autoencoder import ConverterA_CNN, AE
    from encoders import encoder_bi
    from decoders import dynamic_decoder
    from datamanager import datamanager

    for fold_id in range(5):
        autoencoder = ConverterA_CNN(name="encoderdecoder_CNN")
        data = datamanager(time_major=False, expand_dim=True, train_ratio=None, fold_k=5, seed=233)
        ae = AE(autoencoder, data, 64, "Seq2Seq_CNN/fold_{}".format(fold_id))
        if mode=='train':
            ae.train(epoches=100)
        elif mode=='test':
            # ae.saver.restore(ae.sess, os.path.join(ae.model_dir, "model.ckpt-10699"))
            ae.load_model()
            ae.test()
        tf.reset_default_graph()

Seq2Seq_CNN("test")