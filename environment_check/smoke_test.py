"""Check PyTorch CUDA and train ResNet-18 on random data, with no dataset access.

Only imported by environment_check/setup_and_test.ps1. Python 3.12 + torch and
TorchVision are the only required Python packages. Saves report.json beside this file.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPORT_PATH = Path(__file__).with_name("report.json")


def run(download_weights: bool = False) -> int:
    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "test": "2 ResNet-18 training steps, 224x224 random images, batch size 2",
        "dataset_used": False,
        "pretrained_weights_checked": download_weights,
        "status": "FAIL",
    }
    try:
        import torch
        import torchvision
        from torch import nn
        from torchvision import models

        report["torch"] = torch.__version__
        report["torchvision"] = torchvision.__version__
        report["pytorch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        print(f"Python: {report['python']} | torch {torch.__version__} | torchvision {torchvision.__version__}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda} | CUDA available: {report['cuda_available']}")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is False. A usable NVIDIA GPU/driver "
                "is required for this project's training script."
            )

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        report["gpu_name"] = gpu_name
        report["gpu_compute_capability"] = ".".join(map(str, capability))
        report["gpu_total_gib"] = round(total_bytes / (1024**3), 2)
        report["gpu_free_gib_before"] = round(free_bytes / (1024**3), 2)
        report["torch_cuda_arch_list"] = torch.cuda.get_arch_list()
        print(f"GPU: {gpu_name} | compute capability {report['gpu_compute_capability']}")
        print(f"GPU VRAM free/total: {report['gpu_free_gib_before']}/{report['gpu_total_gib']} GiB")

        # Validate a genuine GPU kernel before trying a CNN.
        kernel_test = (torch.ones((64, 64), device=device) @ torch.eye(64, device=device)).sum()
        torch.cuda.synchronize()
        if kernel_test.item() != 4096.0:
            raise RuntimeError("CUDA matrix multiplication returned an unexpected value")

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
        reference = model.fc.weight.detach().clone()
        inputs = torch.randn((2, 3, 224, 224), device=device)
        labels = torch.tensor([0, 1], device=device)
        losses = []
        t0 = time.perf_counter()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at step {step + 1}")
            loss.backward()
            if model.fc.weight.grad is None:
                raise RuntimeError("Gradient did not reach ResNet-18 classification head")
            optimizer.step()
            losses.append(round(loss.item(), 6))
        torch.cuda.synchronize()
        delta = (model.fc.weight - reference).abs().max().item()
        if not delta > 0:
            raise RuntimeError("No ResNet-18 classification weight changed during the test")
        report["smoke_training"] = {
            "steps": 2, "batch_size": 2, "input_size": [224, 224],
            "losses": losses, "head_weight_max_abs_change": delta,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }
        print("ResNet-18 CUDA smoke training: PASS")
        print(f"Losses: {losses}; head weight change: {delta:.8f}")
        del model, optimizer, inputs, labels

        if download_weights:
            print("Checking pretrained ResNet-18 weights (may download ~45MB)...")
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            transfer_model = models.resnet18(weights=weights)
            report["pretrained_weights_download"] = "PASS"
            del transfer_model

        report["status"] = "PASS"
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=8)
        print("ERROR:", report["error"], file=sys.stderr)
        return 1
    finally:
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Test report saved: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-weights", action="store_true", help="Also test ImageNet weight downloads")
    args = parser.parse_args()
    sys.exit(run(args.download_weights))
