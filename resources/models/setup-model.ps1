param(
    [string]$ModelId = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    [string]$OutputDir = ".\resources\models\it-mini-quant",
    [string]$TempDir = ".\tmp_model",
    [switch]$SkipVenv,
    [ValidateSet("reliable", "compact")]
    [string]$Mode = "reliable"
)

$ErrorActionPreference = "Stop"

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Comando non trovato: $Name"
    }
}

Write-Host "[1/6] Verifica prerequisiti..."
Assert-Command "python"

$repoRoot = (Get-Location).Path
$outputFull = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
$tempFull = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $TempDir))

if (-not $SkipVenv) {
    Write-Host "[2/6] Seleziono/creo virtualenv locale..."
    $venvCandidates = @(
        (Join-Path $repoRoot ".venv-1"),
        (Join-Path $repoRoot ".venv")
    )

    $venvPath = $null
    $venvActivate = $null

    foreach ($candidate in $venvCandidates) {
        if (-not (Test-Path $candidate)) {
            continue
        }

        $activateUpper = Join-Path $candidate "Scripts\Activate.ps1"
        $activateLower = Join-Path $candidate "Scripts\activate.ps1"

        if (Test-Path $activateUpper) {
            $venvPath = $candidate
            $venvActivate = $activateUpper
            break
        }
        if (Test-Path $activateLower) {
            $venvPath = $candidate
            $venvActivate = $activateLower
            break
        }
    }

    if (-not $venvPath) {
        $venvPath = Join-Path $repoRoot ".venv"
        python -m venv $venvPath

        $activateUpper = Join-Path $venvPath "Scripts\Activate.ps1"
        $activateLower = Join-Path $venvPath "Scripts\activate.ps1"

        if (Test-Path $activateUpper) {
            $venvActivate = $activateUpper
        } elseif (Test-Path $activateLower) {
            $venvActivate = $activateLower
        }
    }

    if (-not $venvActivate) {
        throw "Impossibile trovare lo script di attivazione venv sotto: $venvPath"
    }

    Write-Host "Uso virtualenv: $venvPath"
    . $venvActivate
}

Write-Host "[3/6] Installo dipendenze Python (optimum + sentence-transformers)..."
python -m pip install --upgrade pip
python -m pip install "optimum[onnxruntime]" sentence-transformers

Write-Host "[4/6] Export ONNX del modello: $ModelId"
if (Test-Path $tempFull) {
    Remove-Item -Recurse -Force $tempFull
}
optimum-cli export onnx --model $ModelId $tempFull

Write-Host "[5/6] Quantizzazione ONNX (int8)..."
if (-not (Test-Path $outputFull)) {
    New-Item -ItemType Directory -Path $outputFull | Out-Null
}
$onnxInputDir = $tempFull
$onnxInput = Join-Path $onnxInputDir "model.onnx"
if (-not (Test-Path $onnxInput)) {
    throw "File ONNX non trovato dopo export: $onnxInput"
}

optimum-cli onnxruntime quantize --avx2 --per_channel --onnx_model $onnxInputDir --output $outputFull

if ($LASTEXITCODE -ne 0) {
    throw "Quantizzazione ONNX fallita (exit code: $LASTEXITCODE)"
}

Write-Host "[6/6] Uniformo i nomi file richiesti da ilbersaglio..."
$quantizedModel = Join-Path $outputFull "model_quantized.onnx"
$finalModel = Join-Path $outputFull "model.onnx"
$tokenizerJson = Join-Path $outputFull "tokenizer.json"

$fallbackModel = Join-Path $tempFull "model.onnx"

if ($Mode -eq "compact") {
    if (-not (Test-Path $quantizedModel)) {
        throw "Modalita compact richiesta, ma model_quantized.onnx non trovato in $outputFull"
    }
    Copy-Item -Force $quantizedModel $finalModel
} else {
    # Nota compatibilita: alcuni modelli quantizzati possono non essere eseguibili da tract-onnx.
    # In modalita reliable, model.onnx punta al modello non quantizzato esportato.
    if (-not (Test-Path $fallbackModel)) {
        throw "Model fallback non trovato dopo export: $fallbackModel"
    }
    Copy-Item -Force $fallbackModel $finalModel
}

$tempTokenizer = Join-Path $tempFull "tokenizer.json"
if ((-not (Test-Path $tokenizerJson)) -and (Test-Path $tempTokenizer)) {
    Copy-Item -Force $tempTokenizer $tokenizerJson
}

if (-not (Test-Path $finalModel)) {
    throw "model.onnx non trovato in $outputFull"
}
if (-not (Test-Path $tokenizerJson)) {
    throw "tokenizer.json non trovato in $outputFull"
}

Write-Host "Modello pronto in: $outputFull"
Write-Host "File trovati:"
Write-Host " - model.onnx"
Write-Host " - tokenizer.json"
Write-Host "Modalita selezionata: $Mode"
Write-Host ""
Write-Host "Esempio esecuzione CLI:"
Write-Host "cargo run --release --bin ilbersaglio-cli -- OMERO ODISSEA --model-dir $OutputDir"
