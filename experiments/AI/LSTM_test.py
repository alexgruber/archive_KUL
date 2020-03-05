
import numpy as np
from scipy.special import expit as sigmoid



def forget_gate(x, h, Weights_hf, Bias_hf, Weights_xf, Bias_xf, prev_cell_state):
    forget_hidden  = np.dot(Weights_hf, h) + Bias_hf
    forget_eventx  = np.dot(Weights_xf, x) + Bias_xf
    return np.multiply( sigmoid(forget_hidden + forget_eventx), prev_cell_state )



def input_gate(x, h, Weights_hi, Bias_hi, Weights_xi, Bias_xi, Weights_hl, Bias_hl, Weights_xl, Bias_xl):
    ignore_hidden  = np.dot(Weights_hi, h) + Bias_hi
    ignore_eventx  = np.dot(Weights_xi, x) + Bias_xi
    learn_hidden   = np.dot(Weights_hl, h) + Bias_hl
    learn_eventx   = np.dot(Weights_xl, x) + Bias_xl
    return np.multiply( sigmoid(ignore_eventx + ignore_hidden), np.tanh(learn_eventx + learn_hidden) )



def update_cell_state(forget_gate_output, input_gate_output):
    return forget_gate_output + input_gate_output


def output_gate(x, h, Weights_ho, Bias_ho, Weights_xo, Bias_xo, cell_state):
    out_hidden = np.dot(Weights_ho, h) + Bias_ho
    out_eventx = np.dot(Weights_xo, x) + Bias_xo
    return np.multiply( sigmoid(out_eventx + out_hidden), np.tanh(cell_state) )

#Set Parameters for a small LSTM network
input_size  = 2 # size of one 'event', or sample, in our batch of data
hidden_dim  = 3 # 3 cells in the LSTM layer
output_size = 1 # desired model output

def model_output(lstm_output, fc_Weight, fc_Bias):
  '''Takes the LSTM output and transforms it to our desired
  output size using a final, fully connected layer'''
  return np.dot(fc_Weight, lstm_output) + fc_Bias



'''
Input dimension: size of input at each time step [x1, x2, ... , xn]
Hidden dimension: shape of hidden state (short-term memory) and cell state (long-term memory) at each time step
Number of layers: # of LSTM layers/cells stacked on top of each other

'''