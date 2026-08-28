import sys

sys.path.append("../../golden_model")

from ternary_reference import ternary_contribution

import cocotb
from cocotb.triggers import Timer

test_cases = [
    (0b01, 1, 1),
    (0b01, 1, 0),
    (0b00, -1, 1),
    (0b00, -1, 0),
    (0b10, 0, 1),
    (0b10, 0, 0),
]
two_compl_map = {
    -1: 0b11,
    1: 0b01,
    0: 0b00,
}


@cocotb.test()
async def test_something(dut):
    for wb, w, i in test_cases:
        dut.weight.value = wb
        dut.input_bit.value = i
        await Timer(1, unit="ns")
        expected = ternary_contribution(w, i)
        assert dut.contribution.value == two_compl_map[expected]
