#!/usr/bin/env python3
"""Generate test input tensors for ACT model inference.

Binary format: [4B ndim][4B per dimension][raw f32 data]
"""

import numpy as np
import os
import struct


def save_tensor(tensor: np.ndarray, path: str):
    """Save tensor in custom binary format: ndim(u32) + shape(u32 each) + f32 data."""
    with open(path, 'wb') as f:
        # Write ndim
        f.write(struct.pack('<I', tensor.ndim))
        # Write shape
        for dim in tensor.shape:
            f.write(struct.pack('<I', dim))
        # Write raw float32 data
        f.write(tensor.astype(np.float32).tobytes())
    print(f"  Saved {path}  shape={tensor.shape}  dtype={tensor.dtype}  "
          f"size={os.path.getsize(path)} bytes")


def main():
    np.random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    input_dir = os.path.join(project_root, 'rootfs_overlay', 'input')

    os.makedirs(input_dir, exist_ok=True)

    # Image tensor: (1, 3, 480, 640)
    image = np.random.randn(1, 3, 480, 640).astype(np.float32)
    save_tensor(image, os.path.join(input_dir, 'image.bin'))

    # State tensor: (1, 14)
    state = np.random.randn(1, 14).astype(np.float32)
    save_tensor(state, os.path.join(input_dir, 'state.bin'))

    print("Test inputs generated.")


if __name__ == "__main__":
    main()
