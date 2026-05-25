#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OVERLAY_DIR="$ROOT_DIR/rootfs_overlay"
DISK_IMG="$ROOT_DIR/make/disk.img"
DOCKER_IMG="docker.cnb.cool/starry-os/arceos-build"

# Ensure disk image exists
if [ ! -f "$DISK_IMG" ]; then
    echo "Error: disk.img not found. Run 'make rootfs' first."
    exit 1
fi

# Check required overlay files
for f in bin/act_infer models/act.onnx input/image.bin input/state.bin; do
    if [ ! -f "$OVERLAY_DIR/$f" ]; then
        echo "Error: $OVERLAY_DIR/$f not found."
        exit 1
    fi
done

echo "=== Preparing rootfs with ACT overlay ==="
echo "Disk image: $DISK_IMG ($(du -h "$DISK_IMG" | cut -f1))"
echo "Overlay files:"
du -h "$OVERLAY_DIR/bin/act_infer" "$OVERLAY_DIR/models/act.onnx" \
   "$OVERLAY_DIR/input/image.bin" "$OVERLAY_DIR/input/state.bin"

# Use Docker to mount ext4, copy files, unmount
docker run --rm \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    --privileged \
    "$DOCKER_IMG" \
    bash -c '
set -e
DISK="/workspace/make/disk.img"
OVERLAY="/workspace/rootfs_overlay"
MNT="/mnt/rootfs"

echo "Mounting ext4 image..."
mkdir -p "$MNT"
mount -o loop "$DISK" "$MNT"

echo "Copying overlay files..."
cp "$OVERLAY/bin/act_infer" "$MNT/bin/act_infer"
chmod +x "$MNT/bin/act_infer"

mkdir -p "$MNT/models"
cp "$OVERLAY/models/act.onnx" "$MNT/models/act.onnx"

mkdir -p "$MNT/input"
cp "$OVERLAY/input/image.bin" "$MNT/input/image.bin"
cp "$OVERLAY/input/state.bin" "$MNT/input/state.bin"

mkdir -p "$MNT/output"

echo "Verifying..."
ls -lh "$MNT/bin/act_infer"
ls -lh "$MNT/models/act.onnx"
ls -lh "$MNT/input/"

echo "Syncing and unmounting..."
sync
umount "$MNT"

echo "Done."
'

echo "=== Rootfs overlay complete ==="
echo "Run 'make run' to boot StarryOS with ACT inference."
