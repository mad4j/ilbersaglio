# Modelli ONNX consigliati

Questa applicazione richiede una directory modello con almeno:

- `model.onnx`
- `tokenizer.json`

## Setup automatico (PowerShell)

Da root del repository:

```powershell
./resources/models/setup-model.ps1
```

Selezione modalita:

```powershell
./resources/models/setup-model.ps1 -Mode reliable
./resources/models/setup-model.ps1 -Mode compact
```

Lo script:

- installa dipendenze Python (`optimum`, `sentence-transformers`)
- esporta il modello ONNX
- quantizza in int8
- garantisce i file finali richiesti:
	- `model.onnx`
	- `tokenizer.json`

Nota:
- `reliable` (default): `model.onnx` viene impostato alla versione non quantizzata per massima compatibilita con `tract-onnx`.
- `compact`: `model.onnx` viene impostato a `model_quantized.onnx` (dimensione ridotta, compatibilita non garantita su tutti i modelli).

## Trade-off dimensione / affidabilita

Consigliato per italiano: **paraphrase-multilingual-MiniLM-L12-v2** in formato ONNX quantizzato.

- Affidabilita: buona su parole/frasi in molte lingue (incluso italiano)
- Dimensione: molto inferiore alla versione FP32 grazie a quantizzazione int8

## Esempio esportazione + quantizzazione (Python)

```bash
pip install optimum[onnxruntime] sentence-transformers
optimum-cli export onnx --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ./tmp_model
optimum-cli onnxruntime quantize --avx2 --per-channel --model ./tmp_model ./resources/models/it-mini-quant
```

Dopo il processo, copia/rinomina i file in modo che la cartella finale contenga:

- `model.onnx`
- `tokenizer.json`

Esempio:

```bash
cp ./resources/models/it-mini-quant/model_quantized.onnx ./resources/models/it-mini-quant/model.onnx
```
