# ilbersaglio

Applicazione Rust onnx-only per verificare la correlazione tra parole italiane e, con tre o piu input, trovare la catena di relazioni che collega la prima parola all'ultima usando:

- anagramma
- sostituzione di una lettera
- aggiunta o rimozione di una lettera
- segnale semantico da modello AI in formato ONNX

Il modello ONNX e obbligatorio per l'esecuzione.
Puoi passarlo come directory oppure come singolo file ZIP che contiene `model.onnx` e `tokenizer.json`.

Il progetto include:

- una libreria Rust riusabile
- una CLI pronta all'uso

## Obiettivo: dimensione ridotta + buona compatibilita

L'approccio consigliato e usare un modello sentence-transformer multilingua esportato in ONNX e quantizzato int8.

Raccomandazione pratica:

- modello base: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- esportazione ONNX + quantizzazione con `optimum-cli`

Dettagli operativi in [resources/models/README.md](resources/models/README.md).

### Setup rapido modello ONNX (Windows/PowerShell)

```powershell
./resources/models/setup-model.ps1
```

Opzioni utili:

```powershell
./resources/models/setup-model.ps1 -OutputDir .\resources\models\it-mini-quant
./resources/models/setup-model.ps1 -ModelId sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
./resources/models/setup-model.ps1 -SkipVenv
./resources/models/setup-model.ps1 -Mode reliable
./resources/models/setup-model.ps1 -Mode compact
```

`-Mode reliable` (default): massima compatibilita con runtime Rust `tract-onnx`.

`-Mode compact`: usa il modello quantizzato come `model.onnx` (piu piccolo, ma possibile incompatibilita runtime su alcuni grafi ONNX).

## Build

```bash
cargo build --release
```

## Test

```bash
cargo test
```

I casi di regressione per le catene sono definiti in `tests/fixtures/chain_cases.jsonl`.
Ogni riga JSONL e un array JSON che descrive la sequenza canonica completa: i test usano il modello ONNX reale in `resources/models/it-mini-quant`, verificano che la sequenza configurata sia una catena valida e che qualunque permutazione delle parole intermedie converga sempre alla stessa catena restituita dal motore.

## Uso CLI

### 1) Correlazione diretta tra due parole con modello ONNX da directory

```bash
cargo run --release --bin ilbersaglio-cli -- OMERO ODISSEA --model-dir resources/deploy
```

Output testuale:

```text
Parola A      : OMERO
Parola B      : ODISSEA
Correlazione  : 0.8421
Esito         : positiva
Metodo/i      : in relazione semantica
```

### 2) Ricerca di una catena con piu di due parole

```bash
cargo run --release --bin ilbersaglio-cli -- ROMA AMOR ONDA --model-dir resources/deploy
```

Output testuale:

```text
Parole input  : ROMA, AMOR, ONDA
Partenza      : ROMA
Arrivo        : ONDA
Esito         : positiva
Catena        : ROMA -> AMOR -> ONDA
Passo  1      : ROMA -> AMOR | 0.9900 | anagrammi
Passo  2      : AMOR -> ONDA | 0.8421 | in relazione semantica
```

### 3) Con modello ONNX da file ZIP

```bash
cargo run --release --bin ilbersaglio-cli -- OMERO ODISSEA --model-dir dist/model-bundle.zip
```

Note:

- nel file ZIP devono essere presenti `model.onnx` e `tokenizer.json` (anche dentro sottocartelle)
- il parametro resta `--model-dir` per retrocompatibilita, ma ora accetta anche un percorso a `.zip`
- se `--model-dir` punta a una directory senza `model.onnx`/`tokenizer.json`, la CLI prova automaticamente a usare uno ZIP compatibile trovato nella directory (es. `dist/model-bundle.zip`)

### 4) Output JSON

```bash
cargo run --release --bin ilbersaglio-cli -- OMERO ODISSEA --model-dir resources/deploy --json
```

Esempio di output JSON:

```json
{
    "word_a": "OMERO",
    "word_b": "ODISSEA",
    "score": 0.91260815,
    "is_correlated": true,
    "matched_methods": [
        "semantic_relation"
    ]
}
```

Con piu di due parole, l'output JSON cambia e include `input_words`, `path`, `is_correlated` e `steps`, dove ogni elemento di `steps` descrive un collegamento della catena trovata.

Il modello ONNX e obbligatorio: senza `--model-dir` (o senza variabile `ILBERSAGLIO_MODEL_DIR`) la CLI termina con errore.

## API libreria

```rust
use ilbersaglio::{CorrelationCalculator, CorrelationConfig};

fn main() -> anyhow::Result<()> {
    let cfg = CorrelationConfig {
        model_dir: Some("resources/models/it-mini-quant".into()), // oppure "dist/model-bundle.zip".into()
    };

    let calc = CorrelationCalculator::new(cfg)?;
    let result = calc.calculate("OMERO", "ODISSEA")?;
    println!("correlazione = {:.4}", result.score);
    println!("metodi positivi = {:?}", result.matched_methods);

    let chain = calc.calculate_chain(&[
        "ROMA".to_string(),
        "AMOR".to_string(),
        "ONDA".to_string(),
    ])?;
    println!("catena trovata = {:?}", chain.path);
    Ok(())
}
```

## Note tecniche

- La correlazione finale e positiva se almeno uno dei controlli disponibili ha esito positivo.
- Con piu di due parole, la CLI cerca una catena dalla prima all'ultima scegliendo ogni volta, tra i candidati in relazione, quello con indice di correlazione piu alto.
- Se tra i candidati esiste almeno una relazione lessicale (anagramma, differenza di una lettera, aggiunta/rimozione), queste hanno priorita rispetto ai candidati solo semantici.
- Tutte le parole fornite in input devono comparire nella catena; l'ultima parola viene raggiunta solo dopo aver incluso le altre.
- Se non esiste alcun candidato valido in uno step, la catena non viene trovata.
- I controlli lessicali sono implementati in funzioni distinte, separate dal controllo semantico via ONNX.
- I controlli lessicali hanno priorita sul controllo semantico: se almeno uno di essi e positivo, `semantic_relation` non viene aggiunto ai metodi positivi.
- Il modulo ONNX applica mean pooling sull'output token-level e poi normalizzazione L2.
- Il punteggio semantico e derivato da cosine similarity rimappata in `[0, 1]`; il metodo `semantic_relation` e positivo da `0.80` in su.
- Il modello ONNX e obbligatorio per il calcolo: non esiste fallback lessicale.
