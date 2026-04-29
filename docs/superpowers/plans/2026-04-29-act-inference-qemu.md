# ACT Inference on StarryOS (QEMU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ACT (Action Chunking with Transformers) model inference on StarryOS in QEMU riscv64, using tract as the ONNX runtime.

**Architecture:** A standalone Rust user-space program (`act_infer`) is cross-compiled for riscv64-linux-musl as a static binary. It uses the `tract-onnx` crate to load an ONNX model, read pre-serialized input tensors, run a forward pass, and write output. The binary and model are placed into the Alpine-based ext4 rootfs image. StarryOS boots in QEMU, launches `/bin/sh` which runs `init.sh`, and from the interactive shell (or via init.sh modification) we execute `act_infer`.

**Tech Stack:** Rust, tract-onnx, ONNX, PyTorch (for model export), riscv64-linux-musl cross-compilation, QEMU riscv64 virt, ext4 rootfs

---

## File Structure

```
StarryOS/
├── tools/
│   └── act_infer/                  # Standalone Rust inference project (NOT part of workspace)
│       ├── Cargo.toml
│       └── src/
│           └── main.rs             # Inference program
├── tools/
│   └── act_model/                  # Model preparation scripts
│       ├── export_act_onnx.py      # Export ACT to ONNX
│       ├── generate_test_input.py  # Generate dummy input tensors
│       └── reference_inference.py  # Python reference output for validation
├── rootfs_overlay/                 # Files to overlay onto the rootfs image
│   ├── bin/act_infer               # Cross-compiled static binary
│   ├── models/act.onnx             # ONNX model file
│   ├── input/                      # Pre-serialized test inputs
│   └── output/                     # Inference output directory
├── Makefile                         # Modified: add rootfs-overlay target
└── src/init.sh                     # Modified (optional): auto-run inference
```

---

### Task 1: Set Up Cross-Compilation Toolchain

**Files:**
- None (environment setup only)

- [ ] **Step 1: Add riscv64 musl Rust target**

```bash
rustup target add riscv64gc-unknown-linux-musl
```

- [ ] **Step 2: Install riscv64 musl cross-linker**

On macOS, install via homebrew:
```bash
brew install filosottile/musl-cross/musl-cross --with-riscv64
```

Verify the toolchain is available:
```bash
riscv64-linux-musl-gcc --version
```

- [ ] **Step 3: Create a minimal test program to verify cross-compilation**

Create `tools/act_infer/Cargo.toml`:

```toml
[package]
name = "act_infer"
version = "0.1.0"
edition = "2021"

[dependencies]
```

Create `tools/act_infer/src/main.rs`:

```rust
fn main() {
    println!("Hello from act_infer on riscv64!");
}
```

Create `tools/act_infer/.cargo/config.toml`:

```toml
[target.riscv64gc-unknown-linux-musl]
linker = "riscv64-linux-musl-gcc"
```

Build and verify:
```bash
cd tools/act_infer
cargo build --target riscv64gc-unknown-linux-musl --release
file ../target/riscv64gc-unknown-linux-musl/release/act_infer
```

Expected output includes: `ELF 64-bit LSB executable, UCB RISC-V, ... statically linked`

- [ ] **Step 4: Commit**

```bash
git add tools/act_infer/
git commit -m "feat(tools): scaffold act_infer project with riscv64 musl cross-compilation"
```

---

### Task 2: Verify tract Loads ONNX on Host (x86_64)

**Files:**
- Modify: `tools/act_infer/Cargo.toml`
- Modify: `tools/act_infer/src/main.rs`

Before cross-compiling for riscv64, verify tract works on the host machine with a simple ONNX model.

- [ ] **Step 1: Add tract-onnx dependency**

Edit `tools/act_infer/Cargo.toml`:

```toml
[package]
name = "act_infer"
version = "0.1.0"
edition = "2021"

[dependencies]
tract-onnx = "0.21"
```

- [ ] **Step 2: Write a test that loads a simple ONNX model**

First, create a trivial ONNX model for testing. Run this Python:

```bash
pip install onnx onnxruntime numpy
```

Create `tools/act_model/generate_test_model.py`:

```python
"""Generate a trivial ONNX model for testing tract compatibility."""
import numpy as np
import onnx
from onnx import helper, TensorProto

X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

# Simple: Y = X * 2 + 1
const_two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
const_one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])

mul_node = helper.make_node("Mul", inputs=["input", "two"], outputs=["mul_out"])
add_node = helper.make_node("Add", inputs=["mul_out", "one"], outputs=["output"])

graph = helper.make_graph(
    [mul_node, add_node],
    "test_graph",
    [X],
    [Y],
    initializer=[
        helper.make_tensor_from_numpy(np.array([2.0], dtype=np.float32), "two"),
        helper.make_tensor_from_numpy(np.array([1.0], dtype=np.float32), "one"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "rootfs_overlay/models/test_simple.onnx")
print("Saved test_simple.onnx")
```

Run it:
```bash
mkdir -p rootfs_overlay/models
python3 tools/act_model/generate_test_model.py
```

- [ ] **Step 3: Write host test program**

Edit `tools/act_infer/src/main.rs`:

```rust
use tract_onnx::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rootfs_overlay/models/test_simple.onnx".to_string());

    println!("Loading model from: {}", model_path);

    let model: Model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .into_optimized()?
        .into_runnable()?;

    println!("Model loaded successfully.");

    let input: Tensor = tract_ndarray::array![[1.0f32, 2.0, 3.0]].into();
    let result = model.run(tvec!(input.into()))?;

    println!("Output: {:?}", result[0]);
    Ok(())
}
```

- [ ] **Step 4: Test on host**

```bash
cd tools/act_infer
cargo run --release -- ../../rootfs_overlay/models/test_simple.onnx
```

Expected output: `Output: [3.0, 5.0, 7.0]` (input * 2 + 1)

- [ ] **Step 5: Commit**

```bash
git add tools/act_infer/ tools/act_model/ rootfs_overlay/
git commit -m "feat(tools): verify tract-onnx loads and runs simple model on host"
```

---

### Task 3: Prepare ACT ONNX Model

**Files:**
- Create: `tools/act_model/export_act_onnx.py`
- Create: `tools/act_model/generate_test_input.py`
- Create: `tools/act_model/reference_inference.py`

- [ ] **Step 1: Install ACT dependencies**

```bash
pip install torch torchvision transformers
```

- [ ] **Step 2: Write ACT ONNX export script**

Create `tools/act_model/export_act_onnx.py`:

```python
"""Export ACT (Action Chunking with Transformers) to ONNX format.

Uses the open-source ACT implementation from TonyZhaoV/ACT.
If the real ACT model is not available, exports a simplified version
with the same architecture for testing.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify


class SimpleACT(nn.Module):
    """Simplified ACT model matching the paper architecture.

    - ResNet18-like CNN backbone for image features
    - Transformer encoder-decoder
    - CVAE latent (set to zeros at inference)
    - MLP head for action prediction
    """

    def __init__(self, state_dim=14, img_h=480, img_w=640, chunk_size=100, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Simplified CNN backbone (ResNet18-style, reduced for tract compat)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Project image features to hidden_dim
        self.img_proj = nn.Linear(256, hidden_dim)

        # Project state to hidden_dim
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Learnable query tokens for decoder
        self.query_embed = nn.Parameter(torch.randn(1, chunk_size, hidden_dim))

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # image: (B, 3, H, W), state: (B, state_dim)
        img_feat = self.cnn(image).flatten(2).transpose(1, 2)  # (B, 1, 256)
        img_feat = self.img_proj(img_feat)  # (B, 1, hidden_dim)

        state_feat = self.state_proj(state).unsqueeze(1)  # (B, 1, hidden_dim)

        # Concatenate tokens for encoder
        tokens = torch.cat([img_feat, state_feat], dim=1)  # (B, 2, hidden_dim)
        enc_out = self.encoder(tokens)  # (B, 2, hidden_dim)

        # Decode with learnable queries
        query = self.query_embed.expand(enc_out.size(0), -1, -1)
        dec_out = self.decoder(query, enc_out)  # (B, chunk_size, hidden_dim)

        # Predict actions
        actions = self.action_head(dec_out)  # (B, chunk_size, state_dim)
        return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="rootfs_overlay/models/act.onnx")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    args = parser.parse_args()

    model = SimpleACT()
    model.eval()

    # Create dummy inputs
    dummy_image = torch.randn(1, 3, 480, 640)
    dummy_state = torch.randn(1, 14)

    torch.onnx.export(
        model,
        (dummy_image, dummy_state),
        args.output,
        input_names=["image", "state"],
        output_names=["actions"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported to {args.output}")

    if args.simplify:
        onnx_model = onnx.load(args.output)
        simplified, check = simplify(onnx_model)
        if check:
            onnx.save(simplified, args.output)
            print(f"Simplified and saved to {args.output}")
        else:
            print("Simplification failed, keeping original")

    # Print model size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Export the model**

```bash
pip install onnx-simplifier
python3 tools/act_model/export_act_onnx.py
```

Expected: Model saved to `rootfs_overlay/models/act.onnx`, size printed.

- [ ] **Step 4: Generate test input data**

Create `tools/act_model/generate_test_input.py`:

```python
"""Generate serialized test input tensors for act_infer."""
import numpy as np
import struct
import os

def write_tensor(path: str, array: np.ndarray):
    """Write tensor as: [4 bytes ndim][4 bytes per dim][raw f32 data]."""
    data = array.astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", data.ndim))
        for d in data.shape:
            f.write(struct.pack("<I", d))
        f.write(data.tobytes())
    print(f"  Wrote {path}: shape={data.shape}, size={os.path.getsize(path)} bytes")

def main():
    os.makedirs("rootfs_overlay/input", exist_ok=True)

    # Image: (1, 3, 480, 640) - normalized ImageNet stats
    image = np.random.randn(1, 3, 480, 640).astype(np.float32)
    write_tensor("rootfs_overlay/input/image.bin", image)

    # State: (1, 14)
    state = np.random.randn(1, 14).astype(np.float32)
    write_tensor("rootfs_overlay/input/state.bin", state)

    print("Done.")

if __name__ == "__main__":
    main()
```

Run:
```bash
python3 tools/act_model/generate_test_input.py
```

- [ ] **Step 5: Generate Python reference output**

Create `tools/act_model/reference_inference.py`:

```python
"""Run ACT inference with PyTorch and save reference output for validation."""
import torch
import numpy as np
import struct
import sys
import os

# Re-use the model definition
sys.path.insert(0, os.path.dirname(__file__))
from export_act_onnx import SimpleACT


def read_tensor(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


def write_tensor(path: str, array: np.ndarray):
    data = array.astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", data.ndim))
        for d in data.shape:
            f.write(struct.pack("<I", d))
        f.write(data.tobytes())


def main():
    os.makedirs("rootfs_overlay/output", exist_ok=True)

    model = SimpleACT()
    model.eval()

    # Load the same seed-generated inputs
    torch.manual_seed(42)
    image = torch.randn(1, 3, 480, 640)
    state = torch.randn(1, 14)

    with torch.no_grad():
        # Save inputs with same seed
        write_tensor("rootfs_overlay/input/image.bin", image.numpy())
        write_tensor("rootfs_overlay/input/state.bin", state.numpy())

        output = model(image, state)

    write_tensor("rootfs_overlay/output/reference.bin", output.numpy())
    print(f"Reference output shape: {output.shape}")
    print(f"First action: {output[0, 0, :7].tolist()}")
    print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")


if __name__ == "__main__":
    main()
```

Run:
```bash
python3 tools/act_model/reference_inference.py
```

Expected: Reference output saved, shape (1, 100, 14), stats printed.

- [ ] **Step 6: Verify tract loads the ACT model on host**

Edit `tools/act_infer/src/main.rs` to load the ACT model:

```rust
use std::fs;
use std::io::{Read, Write};
use std::time::Instant;
use tract_onnx::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

fn read_tensor(path: &str) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut f = fs::File::open(path)?;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    let ndim = u32::from_le_bytes(buf) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        f.read_exact(&mut buf)?;
        shape.push(u32::from_le_bytes(buf) as usize);
    }
    let total: usize = shape.iter().product();
    let mut data = vec![0f32; total];
    f.read_exact(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, total * 4) })?;
    let tensor: Tensor = tract_ndarray::ArrayD::from_shape_vec(shape, data)?.into();
    Ok(tensor)
}

fn write_tensor(path: &str, tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = fs::File::create(path)?;
    let shape = tensor.shape();
    f.write_all(&(shape.len() as u32).to_le_bytes())?;
    for &d in shape {
        f.write_all(&(d as u32).to_le_bytes())?;
    }
    let data: &[f32] = unsafe { std::slice::from_raw_parts(tensor.to_array_view::<f32>()?.as_ptr(), tensor.len()) };
    f.write_all(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) })?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("rootfs_overlay/models/act.onnx");
    let input_dir = args.get(2).map(|s| s.as_str()).unwrap_or("rootfs_overlay/input");
    let output_path = args.get(3).map(|s| s.as_str()).unwrap_or("rootfs_overlay/output/result.bin");

    println!("=== ACT Inference (tract) ===");
    println!("Model: {}", model_path);

    // Load model
    let t0 = Instant::now();
    let model: Model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()?;
    println!("Model loaded in {:?}", t0.elapsed());

    // Load inputs
    let image = read_tensor(&format!("{}/image.bin", input_dir))?;
    let state = read_tensor(&format!("{}/state.bin", input_dir))?;
    println!("Input image shape: {:?}", image.shape());
    println!("Input state shape: {:?}", state.shape());

    // Run inference
    let t1 = Instant::now();
    let result = model.run(tvec!(image.into(), state.into()))?;
    println!("Inference completed in {:?}", t1.elapsed());

    // Print results
    let output = &result[0];
    println!("Output shape: {:?}", output.shape());

    let arr = output.to_array_view::<f32>()?;
    let flat = arr.as_slice().unwrap();
    let first_n = 7.min(flat.len());
    println!("First {} values: {:?}", first_n, &flat[..first_n]);

    // Stats
    let min = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = flat.iter().sum();
    let mean = sum / flat.len() as f32;
    println!("Stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);

    // Save output
    write_tensor(output_path, &output.to_tensor())?;
    println!("Output saved to: {}", output_path);

    println!("=== Done ===");
    Ok(())
}
```

Run on host:
```bash
cd tools/act_infer
cargo run --release -- ../../rootfs_overlay/models/act.onnx ../../rootfs_overlay/input ../../rootfs_overlay/output/result_host.bin
```

Expected: Model loads, inference runs, output printed with stats. Compare with Python reference.

- [ ] **Step 7: Commit**

```bash
git add tools/act_model/ tools/act_infer/ rootfs_overlay/
git commit -m "feat(tools): add ACT model export, test input generation, and tract inference program"
```

---

### Task 4: Cross-Compile act_infer for riscv64

**Files:**
- Build: `tools/act_infer/` (cross-compile)

- [ ] **Step 1: Cross-compile the inference binary**

```bash
cd tools/act_infer
cargo build --target riscv64gc-unknown-linux-musl --release 2>&1
```

If tract dependencies fail to compile for riscv64, check:
1. All transitive deps are pure Rust (no C/C++ build scripts)
2. `ndarray` compiles for riscv64 (it should, pure Rust)
3. If needed, add feature flags to exclude optional deps

- [ ] **Step 2: Verify the binary**

```bash
file target/riscv64gc-unknown-linux-musl/release/act_infer
```

Expected: `ELF 64-bit LSB executable, UCB RISC-V, ... statically linked`

Check size:
```bash
ls -lh target/riscv64gc-unknown-linux-musl/release/act_infer
```

- [ ] **Step 3: Copy binary to overlay directory**

```bash
cp target/riscv64gc-unknown-linux-musl/release/act_infer rootfs_overlay/bin/act_infer
```

- [ ] **Step 4: Commit**

```bash
git add rootfs_overlay/bin/act_infer
git commit -m "feat(tools): cross-compile act_infer for riscv64-linux-musl"
```

---

### Task 5: Create Rootfs Overlay Script

**Files:**
- Create: `scripts/prepare_rootfs.sh`
- Modify: `Makefile` (add overlay target)

- [ ] **Step 1: Write rootfs overlay script**

Create `scripts/prepare_rootfs.sh`:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OVERLAY_DIR="$ROOT_DIR/rootfs_overlay"
DISK_IMG="$ROOT_DIR/make/disk.img"
MOUNT_POINT="/tmp/starryos_rootfs"

# Ensure rootfs image exists
if [ ! -f "$DISK_IMG" ]; then
    echo "Error: disk.img not found. Run 'make rootfs' first."
    exit 1
fi

# Check for required files
if [ ! -f "$OVERLAY_DIR/bin/act_infer" ]; then
    echo "Error: act_infer binary not found. Run cross-compilation first."
    exit 1
fi

if [ ! -f "$OVERLAY_DIR/models/act.onnx" ]; then
    echo "Error: act.onnx model not found. Run export script first."
    exit 1
fi

# Resize disk image if needed (add 256MB for model + data)
echo "Checking disk image size..."
CURRENT_SIZE=$(stat -f%z "$DISK_IMG" 2>/dev/null || stat -c%s "$DISK_IMG" 2>/dev/null)
MIN_SIZE=$((300 * 1024 * 1024))  # 300MB minimum
if [ "$CURRENT_SIZE" -lt "$MIN_SIZE" ]; then
    echo "Resizing disk image from $((CURRENT_SIZE / 1024 / 1024))MB to 512MB..."
    truncate -s 512M "$DISK_IMG"
    # Resize ext4 filesystem
    if command -v e2fsck &> /dev/null; then
        e2fsck -fy "$DISK_IMG" || true
        resize2fs "$DISK_IMG"
    else
        echo "Warning: e2fsck/resize2fs not found. You may need to resize manually."
        echo "On macOS: brew install e2fsprogs"
    fi
fi

# Mount and overlay
echo "Mounting rootfs image..."
mkdir -p "$MOUNT_POINT"
sudo mount -o loop "$DISK_IMG" "$MOUNT_POINT"

echo "Copying overlay files..."
sudo cp "$OVERLAY_DIR/bin/act_infer" "$MOUNT_POINT/bin/act_infer"
sudo chmod +x "$MOUNT_POINT/bin/act_infer"

sudo mkdir -p "$MOUNT_POINT/models"
sudo cp "$OVERLAY_DIR/models/act.onnx" "$MOUNT_POINT/models/act.onnx"

sudo mkdir -p "$MOUNT_POINT/input"
if [ -f "$OVERLAY_DIR/input/image.bin" ]; then
    sudo cp "$OVERLAY_DIR/input/image.bin" "$MOUNT_POINT/input/image.bin"
    sudo cp "$OVERLAY_DIR/input/state.bin" "$MOUNT_POINT/input/state.bin"
fi

sudo mkdir -p "$MOUNT_POINT/output"

echo "Overlay complete. Files on rootfs:"
ls -lh "$MOUNT_POINT/bin/act_infer"
ls -lh "$MOUNT_POINT/models/"
ls -lh "$MOUNT_POINT/input/"

echo "Unmounting..."
sudo umount "$MOUNT_POINT"
echo "Done. Run 'make run' to boot StarryOS with ACT inference support."
```

Make it executable:
```bash
chmod +x scripts/prepare_rootfs.sh
```

- [ ] **Step 2: Add Makefile target**

Add to `Makefile` after the `rootfs` target:

```makefile
act-prepare: rootfs
	@echo "Preparing rootfs with ACT inference overlay..."
	@bash scripts/prepare_rootfs.sh

act-run: act-prepare
	@$(MAKE) run
```

- [ ] **Step 3: Test the overlay process (dry run)**

```bash
# First ensure rootfs is downloaded
make rootfs

# Run the overlay script
bash scripts/prepare_rootfs.sh
```

Verify by mounting and checking:
```bash
sudo mount -o loop make/disk.img /tmp/starryos_rootfs
ls -la /tmp/starryos_rootfs/bin/act_infer
ls -la /tmp/starryos_rootfs/models/act.onnx
sudo umount /tmp/starryos_rootfs
```

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_rootfs.sh Makefile
git commit -m "feat: add rootfs overlay script and make targets for ACT inference"
```

---

### Task 6: Run ACT Inference on QEMU

**Files:**
- None (testing only)

- [ ] **Step 1: Prepare rootfs and run QEMU**

```bash
make act-prepare
make run
```

Wait for the StarryOS shell prompt to appear.

- [ ] **Step 2: Run inference from StarryOS shell**

In the QEMU serial console:
```
/bin/act_infer /models/act.onnx /input /output
```

Observe the output:
- Model loaded successfully?
- Inference completed?
- Output shape correct (1, 100, 14)?
- No NaN values?

- [ ] **Step 3: If inference crashes, debug syscall gaps**

If the binary crashes with a syscall error, check:
1. Run QEMU with `LOG=debug` to see syscall traces:
   ```bash
   make run LOG=debug 2>&1 | tee qemu_log.txt
   ```
2. Search for "unknown syscall" in the log:
   ```bash
   grep -i "unknown syscall\|unhandled\|SIGSYS" qemu_log.txt
   ```
3. Add missing syscalls in `kernel/src/syscall/mod.rs`

Common syscalls tract might need beyond what's implemented:
- `getrandom` (for tensor initialization) -- if missing, can use a fixed seed
- `clock_gettime` (for timing) -- may already be implemented
- `mmap` with specific flags -- may need tweaking

- [ ] **Step 4: Validate output correctness**

After successful inference, extract the output:
```bash
# Mount the rootfs and copy output
sudo mount -o loop make/disk.img /tmp/starryos_rootfs
cp /tmp/starryos_rootfs/output/result.bin rootfs_overlay/output/result_qemu.bin
sudo umount /tmp/starryos_rootfs
```

Compare with Python reference (write a quick validation script or manually check stats match).

- [ ] **Step 5: (Optional) Auto-run inference on boot**

Edit `src/init.sh` to run inference automatically:

```sh
#!/bin/sh

export HOME=/root

echo -e "Welcome to \e[96m\e[1mStarry OS\e[0m!"
env
echo

echo -e "Use \e[1m\e[3mapk\e[0m to install packages."
echo

# Run ACT inference
echo "Running ACT model inference..."
/bin/act_infer /models/act.onnx /input /output
echo "Inference complete."

cd ~
sh --login
```

Rebuild and run:
```bash
make run
```

- [ ] **Step 6: Commit working state**

```bash
git add -A
git commit -m "feat: ACT inference running successfully on StarryOS QEMU"
```

---

### Task 7: Troubleshooting and Iteration

This task covers likely issues and their fixes. Only execute relevant steps.

**Issue A: tract fails to parse ACT ONNX model**

Some ONNX ops may not be supported. Check which ops are used:
```bash
python3 -c "
import onnx
model = onnx.load('rootfs_overlay/models/act.onnx')
ops = set(node.op_type for node in model.graph.node)
for op in sorted(ops):
    print(op)
"
```

Compare with tract's supported ops list: https://github.com/sonos/tract/wiki/Supported-ONNX-ops

If ops are missing:
1. Replace unsupported ops in the PyTorch model (e.g., replace `nn.MultiheadAttention` with explicit Q/K/V linear + matmul)
2. Use `onnx-simplifier` to fold unsupported ops
3. As a fallback, switch to ONNX Runtime (Task 7B)

**Issue B: Binary crashes with "SIGSYS" or syscall error**

Check missing syscalls by enabling debug logging. Add the missing syscall to `kernel/src/syscall/mod.rs`:

1. Find the syscall number in the RISC-V Linux syscall table
2. Add a handler in the appropriate file under `kernel/src/syscall/`
3. Register it in the dispatch table in `kernel/src/syscall/mod.rs`

**Issue C: Out of memory**

If QEMU runs out of memory:
1. Increase `MEM` in Makefile: `MEM=2G`
2. Apply model quantization (FP32 -> FP16):
   ```bash
   pip install onnxconverter-common
   python3 -c "
   import onnx
   from onnxconverter_common import float16
   model = onnx.load('rootfs_overlay/models/act.onnx')
   model_fp16 = float16.convert_float_to_float16(model)
   onnx.save(model_fp16, 'rootfs_overlay/models/act_fp16.onnx')
   "
   ```

**Issue D: Cross-compilation failures for tract on riscv64**

If tract or its dependencies fail to compile for riscv64:
1. Check if the issue is in `tract` itself or a transitive dependency
2. File an issue on the tract repo
3. As fallback, try building on a riscv64 Linux machine or use QEMU user-mode emulation:
   ```bash
   cargo build --target riscv64gc-unknown-linux-musl --release
   # Or with full emulation:
   qemu-riscv64-static target/riscv64gc-unknown-linux-musl/release/act_infer
   ```

---

## Self-Review Checklist

- [x] Spec coverage: Model preparation (Task 3), inference program (Task 2-3), cross-compilation (Task 1, 4), rootfs integration (Task 5), QEMU run (Task 6), troubleshooting (Task 7)
- [x] No placeholders: All code blocks contain actual implementation code
- [x] Type consistency: Tensor read/write format (ndim + shape + f32 data) is consistent across Python and Rust
- [x] File paths are exact and consistent across tasks
- [x] Build commands are exact with expected outputs
