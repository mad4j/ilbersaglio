use clap::{ArgAction, Parser};
use ilbersaglio::{CorrelationCalculator, CorrelationConfig, CorrelationMethod};

#[derive(Debug, Parser)]
#[command(name = "ilbersaglio-cli")]
#[command(about = "Calcola la correlazione semantica tra due parole italiane con ONNX")]
struct Cli {
    /// Prima parola.
    word_a: String,

    /// Seconda parola.
    word_b: String,

    /// Percorso a directory o file ZIP contenente model.onnx e tokenizer.json.
    #[arg(long)]
    model_dir: Option<std::path::PathBuf>,

    /// Stampa l'output in JSON.
    #[arg(long, action = ArgAction::SetTrue)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let cfg = CorrelationConfig {
        model_dir: cli.model_dir,
    };

    let calculator = CorrelationCalculator::new(cfg)?;
    let result = calculator.calculate(&cli.word_a, &cli.word_b)?;

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Parola A      : {}", result.word_a);
        println!("Parola B      : {}", result.word_b);
        println!("Correlazione  : {:.4}", result.score);
        println!(
            "Esito         : {}",
            if result.is_correlated { "positiva" } else { "negativa" }
        );
        println!(
            "Metodo/i      : {}",
            format_methods(&result.matched_methods)
        );
    }

    Ok(())
}

fn format_methods(methods: &[CorrelationMethod]) -> String {
    if methods.is_empty() {
        return "nessun metodo positivo".to_string();
    }

    methods
        .iter()
        .map(|method| method.description())
        .collect::<Vec<_>>()
        .join(", ")
}
