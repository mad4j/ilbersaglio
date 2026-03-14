pub mod correlation;
pub mod error;
pub mod onnx_embedder;

pub use correlation::{
    CorrelationCalculator, CorrelationChainResult, CorrelationChainStep, CorrelationConfig,
    CorrelationMethod, CorrelationResult,
};
pub use error::{CorrelationError, Result};
