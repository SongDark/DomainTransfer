# coding:utf-8

'''
    decoders
    map from Latent space to Sequences
    input an embedding and an expected length, return a sequence
'''

import tensorflow as tf
from utils import BasicBlock, gen_rnn_cells, dense

class dynamic_decoder(BasicBlock):
    '''
        Implementation using raw_rnn.
        Refering to https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb
    '''

    def __init__(self, hidden_units, output_depth, cell_type='lstm', name=None):
        name = "dynamic_{}_decoder".format(cell_type) if name is None else name
        super(dynamic_decoder, self).__init__(hidden_units=hidden_units, name=name)
        self.output_depth = output_depth
        self.cell_type = cell_type

        # Output Projection
        with tf.variable_scope(self.name):
            self.decoder_rnn_layers = gen_rnn_cells(cell_type, hidden_units)
            self.decoder_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(self.decoder_rnn_layers)

    def __call__(self, embedding, lens, with_state=True):
        # for now just assume that 'lens' is available 
        # embedding [bz, dim]

        # batch_size, _ = encoder_final_state.c.shape
        batch_size = embedding.get_shape().as_list()[0]
        eos_time_slice = tf.ones([batch_size, self.output_depth], dtype=tf.float32, name='EOS')
        pad_time_slice = tf.zeros([batch_size, self.output_depth], dtype=tf.float32, name='PAD')
        
        def loop_fn_initial():
            # used when time = 0
            initial_elements_finished = (lens <= 0) # all False
            initial_input = eos_time_slice
            
            '''here's the key link of decoder's last state and decoder's inital state'''
            initial_cell_state = []
            for i in range(len(self.hidden_units)):
                '''project embedding to each layers initial state'''
                state = dense(embedding, self.hidden_units[i], name='dense_input_{}'.format(i))
                if self.cell_type == 'lstm':
                    initial_cell_state.append(tf.nn.rnn_cell.LSTMStateTuple(
                        c=state, 
                        h=tf.zeros_like(state)))
                else:
                    initial_cell_state.append(state)
            
            return (initial_elements_finished, 
                    initial_input,
                    tuple(initial_cell_state),
                    None, None)

        def loop_fn_transition(time, cell_output, cell_state, loop_state):
            def get_next_input():
                return dense(cell_output, self.output_depth, name='dense_output')
            elements_finished = (lens <= time)
            finished = tf.reduce_all(elements_finished)
            inputs = tf.cond(finished, lambda:pad_time_slice, get_next_input)
            states = cell_state
            outputs = cell_output
            loop_state = None 
            return (elements_finished, 
                    inputs,
                    states,
                    outputs,
                    loop_state)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_state is None:
                assert cell_state is None and cell_output is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, cell_output, cell_state, loop_state)
        
        # core codes
        with tf.variable_scope(self.name):
            decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_multi_rnn_cell, loop_fn)
            decoder_outputs = decoder_outputs_ta.stack()

        with tf.variable_scope(self.name+"/rnn", reuse=True):
            '''raw_rnn return cell_output, still need dense projection'''
            # share the same variables
            _, bz, d = decoder_outputs.get_shape().as_list()
            decoder_outputs = tf.reshape(decoder_outputs, (-1, d))
            decoder_outputs = dense(decoder_outputs, self.output_depth, name='dense_output')
            decoder_outputs = tf.reshape(decoder_outputs, (-1, bz, self.output_depth))

        if with_state:
            return decoder_outputs, decoder_final_state
        else:
            return decoder_outputs
