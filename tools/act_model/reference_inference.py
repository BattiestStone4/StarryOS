#!/usr/bin/env python3
"""Run reference inference with SimpleACT model and save output."""

import torch
import numpy as np
import os
import struct

# Re-use the same model definition
from export_act_onnx import SimpleACT


def load_tensor(path: str) -> np.ndarray:
    """Load tensor from custom binary format: ndim(u32) + shape(u32 each) + f32 data."""
    with open(path, 'rb') as f:
        ndim = struct.unpack('<I', f.read(4))[0]
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('<I', f.read(4))[0])
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


def save_tensor(tensor: np.ndarray, path: str):
    """Save tensor in custom binary format."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', tensor.ndim))
        for dim in tensor.shape:
            f.write(struct.pack('<I', dim))
        f.write(tensor.astype(np.float32).tobytes())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    input_dir = os.path.join(project_root, 'rootfs_overlay', 'input')
    output_dir = os.path.join(project_root, 'rootfs_overlay', 'output')

    os.makedirs(output_dir, exist_ok=True)

    # Build model with same seed as export
    torch.manual_seed(42)
    np.random.seed(42)
    model = SimpleACT()
    model.eval()

    # Load test inputs
    image_np = load_tensor(os.path.join(input_dir, 'image.bin'))
    state_np = load_tensor(os.path.join(input_dir, 'state.bin'))

    image = torch.from_numpy(image_np)
    state = torch.from_numpy(state_np)

    print(f"Input image shape: {image.shape}")
    print(f"Input state shape: {state.shape}")

    # Run inference
    with torch.no_grad():
        output = model(image, state)

    output_np = output.numpy()
    print(f"Output shape: {output_np.shape}")
    print(f"Output dtype: {output_np.dtype}")
    print(f"Output min:  {output_np.min():.6f}")
    print(f"Output max:  {output_np.max():.6f}")
    print(f"Output mean: {output_np.mean():.6f}")
    print(f"Output std:  {output_np.std():.6f}")
    print(f"First 7 values of output[0,0]: {output_np[0, 0, :7]}")

    # Save reference output
    ref_path = os.path.join(output_dir, 'reference.bin')
    save_tensor(output_np, ref_path)
    print(f"Reference output saved to: {ref_path}  ({os.path.getsize(ref_path)} bytes)")


if __name__ == "__main__":
    main()
