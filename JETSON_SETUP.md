# Jetson Orin Nano — Live Inference Setup

Setup and operating guide for running `app.py`/`live_infer.py` on the Jetson
Orin Nano specifically. See [README.md](README.md) for the general project
docs — pipeline overview, profiles, config reference, workflow, and
[Running on different machines](README.md#running-on-different-machines) for
how the same codebase/configs work unmodified across Windows, Linux, and this
device.

This device (JetPack 6 / L4T 36.4.7, CUDA 12.6, aarch64) has no conda, so it uses a
plain `venv` instead of the conda setup in the main README. Torch has to come from
NVIDIA's JetPack wheel index, not plain PyPI — the standard `pip install torch`
wheel for aarch64 is CPU-only and won't see the GPU.

## 1. Start the environment

Already set up once at `Hengfeng_Patchcore/.venv`. Each new terminal session just
needs:

```bash
cd /home/ir/justin/Hengfeng_Patchcore
source .venv/bin/activate
```

To rebuild it from scratch (e.g. a fresh SD card/SSD):

```bash
sudo apt update && sudo apt install -y python3.10-venv python3-pip

cd /home/ir/justin/Hengfeng_Patchcore
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# JetPack 6 / CUDA 12.6 torch build — NOT plain PyPI
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ torch==2.8.0 torchvision==0.23.0

# opencv-python 4.14+/5.x requires numpy>=2, which torch 2.8's JetPack build
# doesn't like — pin both down together to a compatible pair
pip install "numpy<2" "opencv-python==4.11.0.86"

# rest of requirements.txt (torch/torchvision lines already satisfied above)
pip install click matplotlib pillow pretrainedmodels pyyaml requests timm \
  scikit-image scikit-learn scipy tqdm ultralytics faiss-cpu flask
pip install "numpy<2"   # some of the above re-bump numpy to 2.x — pin back down

pip install -e .          # installs the patchcore package from src/
```

Sanity check after install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# should print: 2.8.0 True Orin
```

## 2. Start the app

**Every boot**, lock max power/clocks first — this does not persist across
reboots (unlike `nvpmodel`'s power mode, which does), so it needs re-running
each time:

```bash
sudo nvpmodel -m 2       # MAXN_SUPER -- persists across reboots, but harmless to re-run
sudo jetson_clocks       # locks clocks to max -- does NOT persist, must re-run every boot
```

Then:

```bash
cd /home/ir/justin/Hengfeng_Patchcore
source .venv/bin/activate
python app.py --port 5000
```

Open `http://127.0.0.1:5000` on the Jetson itself, or `http://192.168.0.210:5000`
from another device on the same LAN. Ctrl-C to stop.

## 3. Free up memory: SSH in from the MacBook instead of using the desktop

This is an 8GB **unified-memory** device — GPU and CPU share the same RAM, and the
GNOME desktop + Chromium + VS Code running locally can eat 3-4GB, which starves
PatchCore's CUDA allocations (shows up as `NVML_SUCCESS == r INTERNAL ASSERT
FAILED` / `NvMapMemAllocInternalTagged ... error 12`). Running headless over SSH
frees that whole desktop stack back up. SSH is already enabled on this Jetson, so
from the MacBook:

```bash
ssh ir@192.168.0.210
```

(same Wi-Fi network, `192.168.0.0/24`; enter the `ir` account password.)

Once connected over SSH, **temporarily** stop the desktop for this session only —
this does *not* persist across reboots, it just kills the currently-running
desktop:

```bash
sudo systemctl stop gdm3
```

Then run steps 1-2 above from that same SSH session. Memory freed this way is
much larger than closing a few browser tabs, since it also drops GNOME Shell,
Xorg, and every local VS Code/Chromium process, not just Chromium's own tabs.

## 4. Bring the desktop screen back

From the same (or a new) SSH session:

```bash
sudo systemctl start gdm3
```

The physical screen/login prompt reappears immediately — no reboot needed. (A
plain reboot also brings it back on its own, since `stop gdm3` never changed the
boot-time default — it only affects the currently running session.)

## 5. Performance tuning

A single capture went from ~23s down to ~12s, with no loss of precision, via:

- **Max power/clocks**: `sudo nvpmodel -m 2` (MAXN_SUPER) then `sudo jetson_clocks`.
- **PatchCore's memory-bank search on GPU instead of CPU**. FAISS's exact
  search ran on CPU only (~15s/image); `TorchGpuFlatNN` in
  `src/patchcore/common.py` reimplements the identical exact search as a
  chunked torch matmul on CUDA instead (~5s/image, validated bit-for-bit
  identical results). Used automatically whenever `inference.device: cuda`.
- **YOLO exported to TensorRT** — cuts YOLO's cold-start (first capture after
  each profile load) from ~2.4s to ~0.3s. Exported at FP32 (no `--half`), so
  detections are unchanged. Already done for all 4 profiles. `live_config.yaml`
  keeps pointing at the portable `.pt` file either way — `resolve_yolo_weights()`
  (`live_infer.py`) auto-detects a sibling `.engine` and prefers it when present,
  so no config edit is needed after (re)building one (see
  [Running on different machines](README.md#running-on-different-machines)). To
  (re)build one, e.g. after retraining a YOLO model:
  ```bash
  python -c "from ultralytics import YOLO; YOLO('path/to/best.pt').export(format='engine', imgsz=1280, half=False, dynamic=False, batch=1, device=0)"
  ```
  Needs `onnx`, `onnxslim`, `onnxruntime-gpu` (from the same JetPack index as
  torch), and TensorRT's Python bindings, which ship with JetPack system-wide
  but not inside the venv — symlink them in once:
  ```bash
  ln -sf /usr/lib/python3.10/dist-packages/tensorrt* .venv/lib/python3.10/site-packages/
  ```
  **Watch for this warning when the app loads an `.engine` file:**
  `Using an engine plan file across different models of devices is not
  recommended and is likely to affect performance or even cause errors.`
  Observed once after a reboot (not confirmed whether every reboot triggers
  it, or something else about that session did) — TensorRT is saying the
  engine's embedded device fingerprint doesn't match what it's loading onto
  now. Rebuilding the `.engine` (command above) resolved it that time,
  along with an intermittent CUDA OOM that came with it. If you see this
  warning, rebuild that profile's engine rather than debugging the OOM
  directly.
- **`torch.backends.cudnn.benchmark = True`** — lets cuDNN cache the fastest
  conv algorithm for the fixed per-profile input shape.
- **`PYTORCH_NO_CUDA_MEMORY_CACHING=1`** — required, not optional. PyTorch's
  CUDA allocator crashes on this Jetson (`NVML_SUCCESS == r INTERNAL ASSERT
  FAILED`) once YOLO's TensorRT engine and PatchCore are both resident,
  because TensorRT reserves its own GPU memory outside PyTorch's allocator.
  Already set by default in `app.py`/`live_infer.py`.

**Known remaining bottleneck, not currently fixed:** `_align_patch_grid` in
`src/patchcore/patchcore.py` (resampling one backbone layer's patch grid onto
another's resolution before merging) is anomalously slow on this Jetson's CUDA
build — ~4.5s to transpose a single ~600MB tensor, regardless of allocation
strategy or chunking (a low-level kernel/bandwidth issue, not an algorithmic
one). `@torch.compile` (Triton backend) fixes the speed (~15-20x, validated
bit-for-bit correct), and was tried, but is not reliable on this device:

- With YOLO's TensorRT engine also resident, the compiled kernel's own fresh
  ~600MB buffer (no caching-allocator reuse survives
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1`) reproducibly OOMs, regardless of load
  order or `PYTORCH_CUDA_ALLOC_CONF` strategy.
- Suspecting a TensorRT-vs-compile conflict specifically, plain PyTorch YOLO
  (no TensorRT) was tried too — same failure, ~50% of fresh-process first
  calls (2 of 4 trials) still OOM'd. So it isn't a conflict between the two
  optimizations; Inductor's own compilation-time memory footprint is just
  borderline on this particular 8GB unified-memory device, independent of
  what else is loaded.

Reverted in favor of reliability (back to plain eager execution, ~11-12s/
capture instead of ~6.5s). This really does look like it needs either more
RAM than this device has (a 16GB+ Jetson would likely have no trouble), or
chunking `_align_patch_grid`'s work into smaller pieces the way `TorchGpuFlatNN`
already does for FAISS's search, so each compiled call needs a much smaller
buffer — untried, since the compile step itself (not just the runtime buffer)
may be the memory-hungry part.
