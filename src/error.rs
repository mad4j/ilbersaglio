use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum CorrelationError {
    #[error("argomento non valido: {0}")]
    InvalidInput(String),

    #[error("file modello ONNX non trovato: {0}")]
    ModelNotFound(PathBuf),

    #[error("file tokenizer non trovato: {0}")]
    TokenizerNotFound(PathBuf),

    #[error("errore ONNX runtime: {0}")]
    Ort(String),

    #[error("errore tokenizer: {0}")]
    Tokenizer(String),

    #[error("errore interno: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, CorrelationError>;
