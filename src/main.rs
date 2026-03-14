use clap::{ArgAction, Parser};
use ilbersaglio::{CorrelationCalculator, CorrelationConfig, CorrelationMethod};

#[derive(Debug, Parser)]
#[command(name = "ilbersaglio-cli")]
#[command(about = "Calcola correlazioni tra parole italiane e cerca catene di relazione con ONNX")]
struct Cli {
    /// Parole da analizzare; la prima e l'ultima definiscono gli estremi della catena.
    #[arg(required = true, num_args = 2..)]
    words: Vec<String>,

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

    if cli.words.len() == 2 {
        let result = calculator.calculate(&cli.words[0], &cli.words[1])?;

        if cli.json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("Parola A      : {}", result.word_a);
            println!("Parola B      : {}", result.word_b);
            println!("Correlazione  : {:.4}", result.score);
            println!(
                "Esito         : {}",
                if result.is_correlated {
                    "positiva"
                } else {
                    "negativa"
                }
            );
            println!(
                "Metodo/i      : {}",
                format_methods(&result.matched_methods)
            );
        }
    } else {
        let result = calculator.calculate_chain(&cli.words)?;

        if cli.json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("Parole input  : {}", result.input_words.join(", "));
            println!("Partenza      : {}", result.input_words.first().unwrap());
            println!("Arrivo        : {}", result.input_words.last().unwrap());
            println!(
                "Esito         : {}",
                if result.is_correlated {
                    "positiva"
                } else {
                    "negativa"
                }
            );

            if result.is_correlated {
                println!("Catena        : {}", result.path.join(" -> "));

                for (index, step) in result.steps.iter().enumerate() {
                    println!(
                        "Passo {:>2}     : {} -> {} | {:.4} | {}",
                        index + 1,
                        step.word_a,
                        step.word_b,
                        step.score,
                        format_methods(&step.matched_methods)
                    );
                }
            } else {
                println!("Catena        : nessuna catena trovata");
            }
        }
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
