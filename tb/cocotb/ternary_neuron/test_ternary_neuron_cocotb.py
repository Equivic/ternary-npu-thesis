import sys

sys.path.append("../../golden_model")

from ternary_reference import ternary_neuron

import cocotb
from cocotb.triggers import Timer

def pack_inputs(inputs):
    result = 0
    for i, bit in enumerate(inputs):
        result += bit << i
    return result

weight_code_map = {
    1   : 0b01,
    -1  : 0b00,
    0   : 0b10,
}

def pack_weights(weights):
    result = 0
    for i, w in enumerate(weights):
        result += weight_code_map[w] << (2*i)
    return result
