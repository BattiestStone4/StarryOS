# ACT Inference on StarryOS (QEMU) - Design Spec

## Goal

Run ACT (Action Chunking with Transformers) model inference on StarryOS in QEMU riscv64, using tract as the ONNX runtime. This is the first milestone toward running on physical hardware (RK3588, SG2002).

## Background

- **ACT model**: ~20-30M params, ResNet18 backbone + transformer encoder-decoder + CVAE. ONNX export ~80-120MB. Inputs: camera image (480x640) + joint state vector. Output: action chunk sequence.
- **StarryOS**: Monolithic Rust kernel built on ArceOS, supports 100+ Linux syscalls, runs Linux user-space binaries via ext4 rootfs.
- **tract**: Pure Rust ONNX runtime, zero C dependencies, proven on embedded targets (ARM Cortex-M).

## Architecture

```
QEMU riscv64 virt (1GB RAM)
├── StarryOS kernel
│   ├── ext4 rootfs (virtio-blk)
│   │   ├── /bin/act_infer       # statically linked riscv64 musl binary
│   │   ├── /models/act.onnx     # ACT model file
│   │   ├── /input/image.bin     # pre-serialized test image tensor
│   │   ├── /input/state.bin     # pre-serialized joint state tensor
│   │   └── /output/result.bin   # inference output
│   └── Linux syscall compat (mmap, read, write, open, brk, exit, ...)
└── virtio-blk disk image (rootfs)
```

The inference program runs as a regular Linux user-space binary. No kernel modifications needed.

## Components

### 1. ACT Model Preparation

- Source: use an open-source ACT implementation (e.g., TonyZhaoV/ACT or similar)
- Export to ONNX format using PyTorch's `torch.onnx.export`
- Simplify/validate with `onnx-simplifier`
- If model is too large, apply quantization (FP16 or INT8) using ONNX quantization tools
- Target size: ideally under 50MB after optimization

### 2. Test Input Data

- Generate a fixed test input: dummy image tensor (1,3,480,640) and joint state (1,14)
- Serialize as raw binary (f32 array, row-major) for easy loading
- Include a Python script to generate and a Rust struct to parse

### 3. Inference Program (`act_infer`)

Rust program with these responsibilities:
1. Load ONNX model via `tract-onnx`
2. Read pre-serialized input tensors from files
3. Run forward pass
4. Write output tensor to file
5. Print summary (shape, first few values, inference time)

Dependencies:
- `tract-onnx` for model loading and inference
- `std` (not no_std -- runs in user space)

Build: `cargo build --target riscv64gc-unknown-linux-musl --release`

### 4. Cross-Compilation Toolchain

- Install `riscv64-unknown-linux-musl` target via rustup
- May need a musl cross-linker for riscv64
- All dependencies must compile cleanly for riscv64

### 5. Rootfs Integration

- Mount the ext4 rootfs image
- Copy `act_infer` binary to `/bin/`
- Copy model and input data to `/models/` and `/input/`
- Unmount and boot QEMU
- Run: `act_infer --model /models/act.onnx --input /input --output /output`

### 6. QEMU Configuration

- Default: `make run` with `ARCH=riscv64 MEM=1G`
- May need to increase rootfs image size to accommodate model file
- No kernel changes required for QEMU baseline

## Memory Budget (QEMU 1GB)

| Component | Estimated Size |
|-----------|---------------|
| StarryOS kernel | ~5-10MB |
| tract runtime + binary | ~5-10MB |
| ACT ONNX model (file, loaded via mmap) | ~80-120MB |
| Inference intermediate tensors | ~50-100MB |
| Rootfs overhead | ~20-50MB |
| **Total** | **~160-290MB** |

1GB is sufficient for QEMU. SG2002 (256MB) will require quantization or model pruning.

## Success Criteria

1. `act_infer` binary runs on StarryOS QEMU without crashes
2. Model loads successfully via tract
3. Forward pass completes with valid output (non-NaN, correct shape)
4. Output matches reference Python inference within tolerance

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| tract missing ONNX ops for ACT | Medium | Test op compatibility early; fallback to ONNX Runtime if needed |
| StarryOS syscall gaps | Medium | Add missing syscalls as discovered; tract is pure Rust so fewer syscalls needed than C++ ORT |
| Model too large for rootfs | Low | Increase rootfs image size; apply quantization |
| Numerical accuracy issues | Low | Validate against Python reference output |

## Future Steps (after QEMU baseline works)

- **RK3588 (Task 2)**: Port StarryOS to AArch64, use same user-space approach
- **SG2002 (Task 1)**: Quantize model to INT8, reduce resolution, possibly use TFLite Micro
- **Performance optimization**: Threading support, SIMD acceleration

## Why This Approach

1. **No kernel changes needed**: tract runs entirely in user space, leveraging StarryOS's existing Linux binary compatibility
2. **Lowest risk**: Pure Rust, no C++ runtime, minimal syscall surface
3. **Iterative**: Start with QEMU, then adapt for real hardware
4. **Practical**: Model + binary go into rootfs, no complex deployment pipeline
