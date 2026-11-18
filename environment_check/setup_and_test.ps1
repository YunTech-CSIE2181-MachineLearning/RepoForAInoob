<#
.SYNOPSIS
Create a tiny isolated Python environment and verify actual CUDA training.
.DESCRIPTION
No MVTec data required. Does not modify the root project's .venv, uv.lock,
training scripts, experiment CSVs, or checkpoints.
The official uv installer is only downloaded if uv is not already available.
.PARAMETER Full
Also install the repository's full locked project environment (optional).
.PARAMETER DownloadWeights
Also fetch/load ImageNet ResNet-18 pretrained weights (optional; internet required).
#>
param(
    [switch]$Full,
    [switch]$DownloadWeights
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Step([string]$Message) {
    Write-Host "`n[STEP] $Message" -ForegroundColor Cyan
}

function Invoke-Checked([string]$Executable, [string[]]$CommandArguments) {
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable $($CommandArguments -join ' ')"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LocalUvDir = Join-Path $PSScriptRoot ".tools\uv"
$LocalUvExe = Join-Path $LocalUvDir "uv.exe"
$SmokePython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Push-Location $RepoRoot
try {
    Write-Step "Find or install uv"
    if (Test-Path $LocalUvExe) {
        $UvExe = $LocalUvExe
        Write-Host "Using local uv: $UvExe"
    } else {
        $AvailableUv = Get-Command uv -ErrorAction SilentlyContinue
        if ($null -ne $AvailableUv) {
            $UvExe = $AvailableUv.Source
            Write-Host "Using existing uv: $UvExe"
        } else {
            Write-Host "uv is missing. Downloading the official Astral installer."
            Write-Host "Source: https://astral.sh/uv/install.ps1"
            Write-Host "Install target: $LocalUvDir (no PATH/profile changes)"
            New-Item -ItemType Directory -Force -Path $LocalUvDir | Out-Null
            $OldInstallDir = $env:UV_INSTALL_DIR
            $OldNoModifyPath = $env:UV_NO_MODIFY_PATH
            try {
                $env:UV_INSTALL_DIR = $LocalUvDir
                $env:UV_NO_MODIFY_PATH = "1"
                Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression
            } finally {
                $env:UV_INSTALL_DIR = $OldInstallDir
                $env:UV_NO_MODIFY_PATH = $OldNoModifyPath
            }
            if (-not (Test-Path $LocalUvExe)) {
                throw "uv installer did not create $LocalUvExe. Check network or lab restrictions."
            }
            $UvExe = $LocalUvExe
        }
    }
    Invoke-Checked $UvExe @("--version")

    Write-Step "Install/check Python 3.12"
    Invoke-Checked $UvExe @("python", "install", "3.12")

    Write-Step "Install the MINIMAL PyTorch CUDA 12.8 smoke-test environment"
    Write-Host "This uses only torch + torchvision (not Jupyter or MVTec data)."
    Write-Host "The initial CUDA wheel download may be large; existing uv cache is reused."
    Invoke-Checked $UvExe @("sync", "--project", $PSScriptRoot, "--python", "3.12")
    if (-not (Test-Path $SmokePython)) {
        throw "Smoke-test Python was not created: $SmokePython"
    }

    Write-Step "Show driver information (if nvidia-smi is available)"
    $Smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -ne $Smi) {
        try {
            & $Smi.Source "--query-gpu=name,driver_version,memory.total" "--format=csv,noheader"
        } catch {
            Write-Host "nvidia-smi could not report GPU details. Continuing to the PyTorch CUDA test."
        }
    } else {
        Write-Host "nvidia-smi was not found. PyTorch CUDA checks will be authoritative."
    }

    Write-Step "Run a real ResNet-18 CUDA forward/backward/optimizer-step test"
    $TestArgs = @((Join-Path $PSScriptRoot "smoke_test.py"))
    if ($DownloadWeights) { $TestArgs += "--download-weights" }
    Invoke-Checked $SmokePython $TestArgs

    if ($Full) {
        Write-Step "Optional: install FULL repository environment using root uv.lock"
        Invoke-Checked $UvExe @("sync", "--project", $RepoRoot, "--frozen", "--python", "3.12")
        $FullPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path $FullPython)) {
            throw "Full Python environment not created: $FullPython"
        }
        Invoke-Checked $FullPython @("-c", "import torch, torchvision, numpy, pandas, sklearn, PIL; print('Full environment imports: OK')")
    }

    Write-Host "`n[PASS] Environment feasibility test passed." -ForegroundColor Green
    Write-Host "Report: $(Join-Path $PSScriptRoot 'report.json')"
    Write-Host "The 27 existing results and official training environment were not altered (unless -Full was selected)."
    exit 0
} catch {
    Write-Host "`n[FAIL] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "If uv download fails, check internet access or use the official offline/lab installation procedure."
    Write-Host "If CUDA fails, check the NVIDIA driver and the GPU compatibility before training."
    exit 1
} finally {
    Pop-Location
}
