<#
.SYNOPSIS
Create a tiny isolated Python environment and verify actual CUDA training.
.DESCRIPTION
No MVTec data required. Does not modify the root project's .venv, uv.lock,
training scripts, experiment CSVs, or checkpoints.
The official uv installer is only downloaded if uv is not already available.
If uv is only available inside this repository, optionally make it available from
new PowerShell/cmd/VS Code terminals via the current user PATH (no admin rights).
.PARAMETER Full
Also install the repository's full locked project environment (optional).
.PARAMETER DownloadWeights
Also fetch/load ImageNet ResNet-18 pretrained weights (optional; internet required).
.PARAMETER AddUvToPath
Copy uv into a stable per-user folder and add it to the user PATH without prompting.
.PARAMETER NoPathPrompt
Skip the interactive PATH question; useful for unattended setup.
.PARAMETER PathOnly
Configure uv accessibility without installing Python, packages, or running GPU tests.
#>
param(
    [switch]$Full,
    [switch]$DownloadWeights,
    [switch]$AddUvToPath,
    [switch]$NoPathPrompt,
    [switch]$PathOnly
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

# Use the user-level PATH rather than setx or the machine-level PATH.
# Keeping uv outside the cloned repository means moving/deleting the repo will
# not break a PATH entry that the user deliberately opted into.
function Test-PathContains([string]$PathText, [string]$Directory) {
    $Expected = [System.IO.Path]::GetFullPath($Directory).TrimEnd('\', '/')
    foreach ($Part in ($PathText -split ';')) {
        $Item = [Environment]::ExpandEnvironmentVariables($Part.Trim().Trim('"'))
        if (-not $Item) { continue }
        try {
            $Actual = [System.IO.Path]::GetFullPath($Item).TrimEnd('\', '/')
            if ($Actual -ieq $Expected) { return $true }
        } catch {
            # Ignore malformed/legacy PATH entries; never remove them.
        }
    }
    return $false
}

function Offer-UserUvPath([string]$UvExecutable, [bool]$AddWithoutAsking, [bool]$SkipQuestion) {
    $OnPath = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $OnPath) {
        Write-Host "uv is already on PATH: $($OnPath.Source)"
        return
    }

    Write-Host "uv is working, but 'uv' is not a command in new terminals yet."
    Write-Host "Current executable: $UvExecutable"
    Write-Host "An optional permanent setup copies uv/uvx to your local user profile"
    Write-Host "and adds ONLY that folder to your USER PATH (no administrator rights)."
    Write-Host "The repository and system-wide PATH are not modified by this choice."

    if ($SkipQuestion -and -not $AddWithoutAsking) {
        Write-Host "User PATH change skipped (-NoPathPrompt)."
        return
    }

    if (-not $AddWithoutAsking) {
        if ([Console]::IsInputRedirected) {
            Write-Host "Noninteractive terminal: skipping PATH change. Pass -AddUvToPath to opt in."
            return
        }
        try {
            $Answer = Read-Host "Make uv available in future PowerShell/cmd/VS Code terminals? [y/N]"
        } catch {
            Write-Host "Unable to read input; skipping optional PATH change."
            return
        }
        if ($null -eq $Answer -or $Answer.Trim() -notmatch '^(?i:y|yes)$') {
            Write-Host "Skipped. uv remains usable at: $UvExecutable"
            return
        }
    }

    try {
        $LocalAppData = [Environment]::GetFolderPath('LocalApplicationData')
        if ([string]::IsNullOrWhiteSpace($LocalAppData)) {
            throw 'Windows LocalApplicationData folder is unavailable.'
        }
        $PermanentDir = Join-Path $LocalAppData 'Programs\uv'
        New-Item -ItemType Directory -Path $PermanentDir -Force | Out-Null

        $PermanentUv = Join-Path $PermanentDir 'uv.exe'
        if (-not (Test-Path -LiteralPath $PermanentUv)) {
            Copy-Item -LiteralPath $UvExecutable -Destination $PermanentUv -ErrorAction Stop
        } else {
            Write-Host "Reusing existing per-user uv: $PermanentUv"
        }
        $UvXSource = Join-Path (Split-Path -Parent $UvExecutable) 'uvx.exe'
        $PermanentUvx = Join-Path $PermanentDir 'uvx.exe'
        if ((Test-Path -LiteralPath $UvXSource) -and -not (Test-Path -LiteralPath $PermanentUvx)) {
            Copy-Item -LiteralPath $UvXSource -Destination $PermanentUvx -ErrorAction Stop
        }
        Invoke-Checked $PermanentUv @('--version')

        $UserPath = [Environment]::GetEnvironmentVariable('Path', [EnvironmentVariableTarget]::User)
        if (-not (Test-PathContains $UserPath $PermanentDir)) {
            $NewPath = if ([string]::IsNullOrWhiteSpace($UserPath)) {
                $PermanentDir
            } else {
                $UserPath.TrimEnd(';') + ';' + $PermanentDir
            }
            [Environment]::SetEnvironmentVariable('Path', $NewPath, [EnvironmentVariableTarget]::User)
            Write-Host "[PASS] Added to user PATH: $PermanentDir" -ForegroundColor Green
        } else {
            Write-Host "[PASS] User PATH already contains: $PermanentDir" -ForegroundColor Green
        }
        # This applies only to this script's process and its children. The
        # parent PowerShell that launched run.cmd cannot be changed by a child.
        if (-not (Test-PathContains $env:Path $PermanentDir)) {
            $env:Path = "$PermanentDir;$env:Path"
        }
        Write-Host "Open a NEW PowerShell/cmd/VS Code terminal, then run: uv --version"
        Write-Host "Your currently open calling terminal may still need to be restarted."
    } catch {
        Write-Warning "Optional user PATH setup did not succeed: $($_.Exception.Message)"
        Write-Host "The normal GPU/environment setup can continue using: $UvExecutable"
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
        $AvailableUv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
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

    if ($AddUvToPath -and $NoPathPrompt) {
        throw "Choose either -AddUvToPath or -NoPathPrompt, not both."
    }
    Write-Step "Optional: make uv available in future terminals"
    Offer-UserUvPath -UvExecutable $UvExe -AddWithoutAsking ([bool]$AddUvToPath) -SkipQuestion ([bool]$NoPathPrompt)

    if ($PathOnly) {
        Write-Host "`n[PASS] uv PATH configuration step completed." -ForegroundColor Green
        Write-Host "Open a NEW terminal before running: uv --version"
        exit 0
    }

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
