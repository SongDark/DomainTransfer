import tensorflow as tf 

class BasicBlock(object):
    def __init__(self, hidden_units, name):
        self.name = name
        self.hidden_units = hidden_units
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

def gen_rnn_cells(cell_type, hidden_units):
    if cell_type == 'rnn':
        return [tf.nn.rnn_cell.RNNCell(size) for size in hidden_units]
    elif cell_type == 'lstm':
        return [tf.nn.rnn_cell.LSTMCell(size, use_peepholes=False) for size in hidden_units]
    elif cell_type == 'gru':
        return [tf.nn.rnn_cell.GRUCell(size) for size in hidden_units]