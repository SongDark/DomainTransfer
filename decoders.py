# coding:utf-8

'''
    decoders
    map from Latent space to Sequences
    input an embedding and an expected length, return a sequence
'''

import tensorflow as tf
from utils import BasicBlock, gen_rnn_cells

class dynamic_decoder(BasicBlock):
    '''
        Implementation using raw_rnn.
        Refering to https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb
    '''

    def __init__(self, hidden_units, output_depth, cell_type='lstm', name=None):
        name = "dynamic_{}_decoder".format(cell_type) if name is None else name
        super(dynamic_decoder, self).__init__(hidden_units=hidden_units, name=name)
        self.output_depth = output_depth

        # Output Projection
        with tf.variable_scope(self.name):
            self.W_out_proj = tf.get_variable('W_out_proj', shape=[self.hidden_units[-1], self.output_depth], initializer=tf.random_uniform_initializer(-1.0, 1.0), dtype=tf.float32)
            self.b_out_proj = tf.get_variable('b_out_proj', shape=[self.output_depth], initializer=tf.zeros_initializer(), dtype=tf.float32)

            self.decoder_rnn_layers = gen_rnn_cells(cell_type, hidden_units)
            self.decoder_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(self.decoder_rnn_layers)

    def __call__(self, encoder_final_state, lens):
        # for now just assume that 'lens' is available 
        # 'encoder_final_state' should be either a tuple of LSTMStateTuple or a LSTMStateTuple, element size [bz, H]
        # 'encoder_final_state' should have the same width as hidden_unit[0]

        # batch_size, _ = encoder_final_state.c.shape
        batch_size, _ = tf.unstack(tf.shape(encoder_final_state.c))
        eos_time_slice = tf.ones([batch_size, self.output_depth], dtype=tf.float32, name='EOS')
        pad_time_slice = tf.zeros([batch_size, self.output_depth], dtype=tf.float32, name='PAD')
        
        def loop_fn_initial():
            # used when time = 0
            initial_elements_finished = (lens <= 0) # all False
            initial_input = eos_time_slice
            
            '''here's the key link of decoder's last state and decoder's inital state'''
            initial_cell_state = []
            initial_cell_state.append(encoder_final_state) # [bz, H1]
            for i in range(1, len(self.hidden_units)):
                initial_cell_state.append(self.decoder_rnn_layers[i].zero_state(batch_size, dtype=tf.float32))
            
            return (initial_elements_finished, 
                    initial_input,
                    tuple(initial_cell_state),
                    None, None)

        def loop_fn_transition(time, cell_output, cell_state, loop_state):
            def get_next_input():
                return tf.add(tf.matmul(cell_output, self.W_out_proj), self.b_out_proj)
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

        return decoder_outputs, decoder_final_state
