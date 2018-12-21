# coding:utf-8

'''
    encoders
    map from Sequences to Latent space
    input a sequence and return a fixed-size embedding
'''

import tensorflow as tf
from utils import BasicBlock, gen_rnn_cells

class encoder_naive(BasicBlock):
    '''naive implementation using dynamic_rnn'''

    def __init__(self, hidden_units, cell_type='lstm', name=None):        
        name = 'naive_encoder' if name is None else name
        super(encoder_naive, self).__init__(hidden_units=hidden_units, name=name)

        with tf.variable_scope(self.name, reuse=False):
            encoder_rnn_layers = gen_rnn_cells(cell_type, hidden_units)
            self.encoder_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_rnn_layers, state_is_tuple=True)
    
    def __call__(self, x, lens, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            encoder_outputs, encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.encoder_multi_rnn_cell, 
                time_major=True,
                inputs=x, 
                sequence_length=lens, 
                dtype=tf.float32)

        return encoder_outputs, encoder_last_state

class encoder_bi(BasicBlock):
    '''implementation of bi-lstm using bidirectional_dynamic_rnn'''

    def __init__(self, hidden_units, cell_type='lstm', name=None):
        name = 'bi_encoder' if name is None else name
        super(encoder_bi, self).__init__(hidden_units=hidden_units, name=name)
        self.cell_type = cell_type
        with tf.variable_scope(self.name, reuse=False):
            encoder_fw_cells = gen_rnn_cells(cell_type, hidden_units)
            encoder_bw_cells = gen_rnn_cells(cell_type, hidden_units)
            self.encoder_fw_stacked_cells = tf.nn.rnn_cell.MultiRNNCell(encoder_fw_cells)
            self.encoder_bw_stacked_cells = tf.nn.rnn_cell.MultiRNNCell(encoder_bw_cells)
        
    def __call__(self, x, lens, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            [encoder_fw_outputs, encoder_bw_outputs], [encoder_fw_last_states, encoder_bw_last_states] = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_fw_stacked_cells,
                                                cell_bw=self.encoder_bw_stacked_cells,
                                                inputs=x,
                                                sequence_length=lens,
                                                time_major=True,
                                                dtype=tf.float32)
            
            encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], axis=-1) # [T,bz,H] + [T,bz,H] -> [T,bz,2H]
            encoder_last_state = []
            for i in range(len(self.hidden_units)):
                if self.cell_type == 'lstm':
                    encoder_last_state_c = tf.concat((encoder_fw_last_states[i].c, encoder_bw_last_states[i].c), axis=1)
                    encoder_last_state_h = tf.concat((encoder_fw_last_states[i].h, encoder_bw_last_states[i].h), axis=1)
                    encoder_last_state.append(tf.nn.rnn_cell.LSTMStateTuple(
                                                c=encoder_last_state_c,
                                                h=encoder_last_state_h))
                else:
                    encoder_last_state.append(tf.concat([encoder_fw_last_states[i], encoder_bw_last_states[i]], axis=1))

            return encoder_outputs, tuple(encoder_last_state)

class dynamic_encoder(BasicBlock):
    '''an implementation of dynamic_rnn using raw_rnn'''

    def __init__(self, hidden_units, name=None):
        name = 'dynamic_encoder' if name is None else name
        super(dynamic_encoder, self).__init__(hidden_units=hidden_units, name=name)

        with tf.variable_scope(self.name):
            encoder_rnn_layers = [tf.nn.rnn_cell.LSTMCell(size, use_peepholes=False) for size in self.hidden_units]
            self.encoder_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_rnn_layers, state_is_tuple=True)
            
    def __call__(self, x, lens):
        # x must be time-major, i.e. [max_time, batch_size, dim]
        batch_size, input_depth = x.shape[1:]
        max_time = tf.unstack(tf.shape(x))[0]

        x_ta = tf.TensorArray(size=max_time, dtype=tf.float32)
        x_ta = x_ta.unstack(x)
        
        def loop_fn(time, cell_output, cell_state, loop_state):
            # 'output' would has the size of final layer output [bz, Hlast]
            emit_output = cell_output 
            # 'state' would be a tuple of LSTMStateTuple
            if cell_output is None: # time = 0
                next_cell_state = self.encoder_multi_rnn_cell.zero_state(batch_size, tf.float32)
            else:
                next_cell_state = cell_state

            elements_finished = (lens <= time)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda:tf.zeros([batch_size, input_depth], dtype=tf.float32),
                lambda:x_ta.read(time)
            )
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state, 
                    emit_output, next_loop_state)

        with tf.variable_scope(self.name):
            outputs_ta, final_state, _ = tf.nn.raw_rnn(self.encoder_multi_rnn_cell, loop_fn)
            outputs = outputs_ta.stack()

        return outputs, final_state