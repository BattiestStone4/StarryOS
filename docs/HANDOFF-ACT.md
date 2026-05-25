# ACT 推理移植到 StarryOS - 交接文档

## TL;DR

我们已经在 **QEMU + StarryOS (riscv64)** 上跑通了 ACT 模型的推理流程（三等奖目标完成）。
现在需要你接手，把模型的输入输出维度改成**适配我们网球抓球小车**的形状，下一步再考虑量化和真实硬件部署。

---

## 已完成的工作

### 1. 整体架构

```
StarryOS (QEMU riscv64)
   └── ext4 rootfs
        ├── /bin/act_infer    ← 我们写的 Rust 推理程序（静态编译）
        ├── /models/act.onnx  ← ACT 模型（13MB）
        ├── /input/           ← 测试输入（图像+状态）
        └── /output/          ← 推理结果

技术栈：
- 推理框架：tract-onnx（纯 Rust，无 C 依赖）
- 编译目标：riscv64gc-unknown-linux-musl（静态链接）
- 构建环境：docker.cnb.cool/starry-os/arceos-build
```

### 2. 关键文件清单

| 路径 | 作用 |
|------|------|
| `tools/act_infer/` | Rust 推理程序（用 tract 加载 ONNX 跑推理）|
| `tools/act_model/export_act_onnx.py` | PyTorch 定义模型 + 导出 ONNX |
| `tools/act_model/generate_test_input.py` | 生成测试用的二进制 tensor 输入 |
| `tools/act_model/reference_inference.py` | PyTorch 参考输出（用于精度对比）|
| `scripts/prepare_rootfs.sh` | 把二进制+模型注入 rootfs（用 Docker 挂载 ext4）|
| `rootfs_overlay/` | 要塞进 rootfs 的所有文件 |
| `Makefile` | 加了 `act-prepare` / `act-run` target |
| `docs/superpowers/specs/` | 设计文档 |
| `docs/superpowers/plans/` | 实施计划 |

### 3. 当前 Tensor 维度（**通用模板，需要你修改**）

```
输入：
  image : (1, 3, 480, 640)    f32
  state : (1, 14)              f32

输出：
  actions: (1, 100, 14)        f32
```

模型大小 13MB，QEMU 上推理一次 **77 秒**（纯软件模拟，真实硬件会快很多）。

### 4. 验证结果

| 指标 | 主机 (macOS arm64) tract | QEMU StarryOS riscv64 tract |
|------|--------------------------|------------------------------|
| 输出 shape | [1, 100, 14] | [1, 100, 14] |
| 输出 mean | 0.088503 | 0.088503 |
| 与 PyTorch 参考 误差 | < 1e-6 | < 1e-6 |

---

## 你要接手的工作

### 任务：把 Tensor 维度适配我们的网球小车

参考我们 `/Users/stone/starry_for_car/aka0` 项目里的硬件：
- **2 轮差速底盘**（左右各 2 个 PWM 通道，但逻辑上左右两边）
- **3 个舵机手臂**（肩、肘、夹爪）
- **VI 摄像头**（硬件可缩放到任意分辨率）
- **YOLOv8 已经能检测球的中心点**（可以作为额外输入）

### 推荐改成

```
输入：
  image : (1, 3, 224, 224)    ← 分辨率从 480×640 降到 224×224，模型变小
  state : (1, 7)               ← [左轮速度, 右轮速度, 肩角度, 肘角度, 夹爪, 球cx, 球cy]

输出：
  actions: (1, 50, 5)          ← 50 步规划窗口，每步 5 个控制量
                                  [左轮速度, 右轮速度, 肩目标角度, 肘目标角度, 夹爪]
```

**为什么这样设计：**
- `state_dim=7`：2 轮速度 + 2 关节角 + 1 夹爪 + 2 球坐标（YOLOv8 检测结果）
- `action_dim=5`：完全对应硬件控制接口
- `chunk_size=50`：50 × 100ms ≈ 5 秒规划窗口，比 100 步更灵活
- `224×224`：模型体积估计降到 5-6MB，对 SG2002 (256MB) 友好

### 具体改 3 个文件

**1. `tools/act_model/export_act_onnx.py`**
```python
# 改 SimpleACT 的构造参数：
state_dim=7          # 原来 14
action_dim=5         # 原来 14
num_queries=50       # 原来 100，对应 chunk_size

# 改 dummy_image 的 shape：
dummy_image = torch.randn(1, 3, 224, 224)   # 原来 (1, 3, 480, 640)
dummy_state = torch.randn(1, 7)              # 原来 (1, 14)
```

**2. `tools/act_model/generate_test_input.py`**
```python
# 改维度：
image = np.random.randn(1, 3, 224, 224).astype(np.float32)  # 原来 480, 640
state = np.random.randn(1, 7).astype(np.float32)             # 原来 14
```

**3. `tools/act_model/reference_inference.py`**
跟着上面两个文件同步改即可。

### 推理程序不用改

`tools/act_infer/src/main.rs` **是维度无关的**，从文件读什么 shape 就跑什么 shape，所以你只需要改 Python 脚本。

---

## 怎么验证你的修改

完整流程（在 macOS / Linux 主机上跑，需要 Python + Docker）：

```bash
# 0. 第一次拉代码后，先生成模型和测试数据（仓库里没有这些大文件）
pip install torch onnx onnxsim numpy
python3 tools/act_model/export_act_onnx.py        # 生成 rootfs_overlay/models/act.onnx
python3 tools/act_model/generate_test_input.py    # 生成 rootfs_overlay/input/*.bin
python3 tools/act_model/reference_inference.py    # 生成 rootfs_overlay/output/reference.bin

# 1. 改维度后重新生成（同样三条命令）

# 2. 主机上验证（不用进 QEMU，先跑通 tract 加载和推理）
cd tools/act_infer
cargo run --release -- ../../rootfs_overlay/models/act.onnx \
                       ../../rootfs_overlay/input \
                       ../../rootfs_overlay/output/result_host.bin

# 3. 看输出，跟 reference.bin 对比，误差应该 < 1e-5

# 4. （可选）跑到 QEMU 上验证
#    需要交叉编译，详见下面"如何交叉编译"
```

---

## 如何交叉编译 act_infer 到 riscv64

我用了 OrbStack VM（也可以换成 Docker），里面装了 musl 交叉编译器：

```bash
# 在 VM 内（已经装好 musl 工具链）：
export PATH=/opt/riscv64-linux-musl-cross/bin:$PATH
cd ~/act_infer
cargo +stable build --target riscv64gc-unknown-linux-musl --release

# 输出： ~/act_infer/target/riscv64gc-unknown-linux-musl/release/act_infer
# file 应该显示：ELF 64-bit LSB executable, UCB RISC-V, statically linked
```

`.cargo/config.toml` 必须配静态链接：
```toml
[target.riscv64gc-unknown-linux-musl]
linker = "riscv64-linux-musl-gcc"
rustflags = ["-C", "target-feature=+crt-static", "-C", "relocation-model=static"]
```

---

## 怎么在 QEMU 上跑

```bash
# 1. 编译 StarryOS 内核（需要 Docker）
docker run --rm -v $(pwd):/workspace -w /workspace \
    docker.cnb.cool/starry-os/arceos-build bash -c 'make build'
ln -sf workspace_riscv64-qemu-virt.bin StarryOS_riscv64-qemu-virt.bin

# 2. 下载 rootfs 镜像（如果还没有）
make rootfs

# 3. 把 act_infer + 模型 + 输入注入 rootfs
bash scripts/prepare_rootfs.sh

# 4. 启动 QEMU
make justrun

# 5. 在 starry:~# shell 里手动执行
/bin/act_infer /models/act.onnx /input /output/result.bin
```

或者改 `src/init.sh` 让它启动时自动跑（已经验证可行）：
```sh
# Do your initialization here!
/bin/act_infer /models/act.onnx /input /output/result.bin
```

---

## 后续可以做的方向

按难度排序：

1. **改维度**（你接手的部分）→ 给真实小车用做准备
2. **量化**（FP32→FP16→INT8）→ 模型变小，推理变快，SG2002 才可能跑
3. **AArch64 适配** → RK3588 部署（二等奖）
4. **录制示教数据** → 让 aka0 自主跑+人工 Y/N 筛选 → 训练自己的 ACT 模型
5. **NPU 硬件加速** → SG2002 的 TPU 或 RK3588 的 RKNN，把推理时间从秒级降到毫秒级（aka0 已经在用 SG2002 TPU 跑 YOLOv8，可以参考）

---

## 重要约束 / 坑

1. **Docker 镜像是 amd64 的**，在 Apple Silicon 上会有 platform warning，能用但慢。
2. **内核 binary 路径名**：Docker 里 `make build` 生成 `workspace_*.bin`，本机生成 `StarryOS_*.bin`，二选一时记得做 symlink。
3. **rootfs 镜像默认只有 25MB**，加上 13MB 模型 + 15MB 二进制 + 3.5MB 输入，刚好够；**未来如果模型变大要扩容**。
4. **prepare_rootfs.sh 用 Docker 挂载 ext4**（macOS 不能直接 mount ext4），改 rootfs 必须经过它。
5. **tract 不支持所有 ONNX 算子**，改模型时注意避开 `MultiheadAttention` 之类的（用 Linear+matmul 拆开写）。我们当前的 `SimpleACT` 已经规避了这个问题，照着改就行。

---

## 联系

有问题问我（邵志航）。所有改动都在 `dev` 分支。
