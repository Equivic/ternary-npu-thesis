TOPLEVEL_LANG = vhdl
SIM = ghdl

VHDL_SOURCES = ../../../rtl/npu/ternary_pe.vhd ../../../rtl/npu/ternary_neuron.vhd
TOPLEVEL = ternary_neuron
ARCH = structural

COCOTB_TEST_MODULES = test_ternary_neuron_cocotb

include $(shell cocotb-config --makefiles)/Makefile.sim
