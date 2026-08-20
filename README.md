# Near-Memory Ternary Neural Network Accelerator

Bachelor's thesis project — Electronics Engineering, Vilnius TECH.

A ternary (−1, 0, +1) neural network accelerator using a near-memory
architecture, where each neuron has its own local weight storage paired
with a dedicated compute lane, instead of a shared centralized compute
block. Implemented entirely in standard digital cells, integrated as a
memory-mapped peripheral into an open-source RISC-V SoC.

Full architecture and motivation: see [`docs/thesis_proposal.pdf`](docs/thesis_proposal.pdf).

## Status

🚧 Early development — started August 2026.

- [ ] Single ternary processing element (Verilog + CocoTB verification)
- [ ] Layer 1 near-memory module (32 parallel PEs)
- [ ] Centralized baseline architecture
- [ ] Synthesis comparison (area / power / timing)
- [ ] PicoRV32 SoC integration
- [ ] FPGA bring-up
- [ ] Validation task (TBD)

## Repository structure

```
rtl/           Verilog source (npu/, soc/)
tb/            Testbenches (CocoTB, Python golden model)
synth/         OpenLane configs and synthesis reports
fpga/          Constraints and bitstream build scripts
docs/          Thesis proposal and reference material
scripts/       Automation / toolchain helpers
```

## Toolchain

- Verilog / SystemVerilog
- [CocoTB](https://www.cocotb.org/) for Python-based verification
- [OpenLane](https://github.com/The-OpenROAD-Project/OpenLane) + Sky130 PDK for synthesis
- KLayout for layout inspection
- FPGA target: TBD

## Supervisor

doc. dr. Marijan Jurgo — Vilnius TECH
