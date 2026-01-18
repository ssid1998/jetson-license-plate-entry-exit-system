#!/usr/bin/env bash
set -euo pipefail
	
echo "=== Jetson Nano System Verification ==="
	
echo "[1] OS / L4T Information"
if [ -f /etc/nv_tegra_release ]; then
cat /etc/nv_tegra_release
else
echo "WARNING: /etc/nv_tegra_release not found."
fi
lsb_release -a 2>/dev/null || true
	
echo
echo "[1b] Kernel"
uname -a || true

echo "[2] CUDA Availability"
if command -v nvcc >/dev/null 2>&1; then
nvcc --version | tail -n 1
else
echo "WARNING: nvcc not found (CUDA toolkit may not be installed)."
fi
	
echo "[3] TensorRT Packages (dpkg)"
dpkg -l | grep -i tensorrt || echo "WARNING: TensorRT packages not found via dpkg."
	
echo
echo "[4] Memory / Swap"
free -h || true
swapon --show || echo "WARNING: no swap devices shown."

echo
echo "[5] Storage"
df -h || true

echo
echo "[6] Camera Devices"
ls -la /dev/video* 2>/dev/null || echo "WARNING: no /dev/video* devices found (camera may be missing)."

echo
echo "[7] Runtime Monitoring Tool"
if command -v tegrastats >/dev/null 2>&1; then
echo "tegrastats found. Run 'tegrastats' in a separate terminal to monitor the device."
else
echo "WARNING: tegrastats not found."
fi
	
echo "=== Verification complete. ==="
