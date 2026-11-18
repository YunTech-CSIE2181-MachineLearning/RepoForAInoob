# Environment check — Windows / NVIDIA CUDA

A **minimal, isolated** compatibility test for the CISE2181 MVTec AD project.
Use this immediately after cloning the repository on a new Windows PC **before**
installing the full project or downloading the MVTec dataset.

## One-command setup and GPU training test

From the repository root in **PowerShell**:

```powershell
.\environment_check\run.cmd
```

Alternatively, **double-click `environment_check/run.cmd`** from File Explorer.
The command checks for `uv` (installs it locally from the official Astral
installer if missing), downloads Python 3.12 if needed, and installs only
`torch` and `torchvision` into `environment_check/.venv/` using the project's
**CUDA 12.8** versions. It then:

1. Prints NVIDIA driver info if `nvidia-smi` is available.
2. Checks `torch.cuda.is_available()` and reads the GPU name, compute capability, and free VRAM.
3. Runs an actual CUDA matrix multiplication.
4. Trains a randomly initialized **ResNet-18** for **two optimizer steps** on synthetic 224×224 images (batch size 2).
5. Writes **`environment_check/report.json`** containing the result and any error.

No MVTec images are required or opened. No existing 27 experiment results or
`outputs_topic10/splits/` are modified. The default command does **not** touch
the root `.venv` or `uv.lock`.

> **Not a full reproducibility test.** This proves that a real GPU kernel,
> forward pass, backward pass, and optimizer update run in the smoke-test
> environment. It does not guarantee that the MVTec AD files are present or
> that the complete 27-run project will succeed.

## Optional modes

```powershell
# Additionally test that ImageNet ResNet-18 weights can be downloaded/loaded:
.\environment_check\run.cmd -DownloadWeights

# Also build the full repository environment with its locked dependencies:
.\environment_check\run.cmd -Full
```

`-DownloadWeights` needs an internet connection and accesses the standard
TorchVision weight distribution endpoint. The default smoke test does not
need pretrained weights.

`-Full` uses `uv sync --frozen` on the **root project** and builds/updates
`RepoForAInoob/.venv`. Use it only when you want the complete environment for
`prepare_splits.py`, `train_baseline.py`, and the course notebook.

## Files

| File | Purpose |
| --- | --- |
| `run.cmd` | One-click entry point for Windows. |
| `setup_and_test.ps1` | uv/Python installation, minimal environment bootstrap, and optional full setup. |
| `pyproject.toml` | Only CUDA PyTorch and TorchVision, pinned to the root `uv.lock` versions. |
| `smoke_test.py` | GPU compatibility and actual two-step ResNet-18 training test. |
| `.gitignore` | Keeps the local environment, installed tools, lock, and report out of Git. |
| `report.json` (generated) | Detailed test result, including errors. |

## Troubleshooting

- **uv installation blocked:** the machine may restrict downloads, PowerShell
  execution, or user installation. Ask the lab administrator about a supported
  preinstalled uv/Python setup. Installing uv into this folder avoids requiring
  administrator privileges in typical user accounts.
- **`torch.cuda.is_available()` is false:** verify NVIDIA driver installation
  and that the GPU is supported by CUDA 12.8. `nvidia-smi` alone is insufficient.
- **Out of GPU memory:** close other GPU-consuming software and retry; the
  smoke test uses a very small batch, but the complete project needs more.
- **No network:** a completely fresh machine cannot install Python/PyTorch or
  download pretrained weights without internet or a local package cache.

## After a successful test

For a full setup on the same machine, run `run.cmd -Full` (or `uv sync` from
the root project if uv is already installed). Then follow the root README
for MVTec AD placement and the dataset/experiment workflow.

The standard runner uses the official Astral installation URL:
https://astral.sh/uv/install.ps1
