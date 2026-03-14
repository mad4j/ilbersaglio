use std::path::PathBuf;

use serde::Serialize;

use crate::{
    error::{CorrelationError, Result},
    onnx_embedder::OnnxEmbedder,
};

#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    pub model_dir: Option<PathBuf>,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            model_dir: std::env::var("ILBERSAGLIO_MODEL_DIR").ok().map(PathBuf::from),
        }
    }
}

const SEMANTIC_RELATION_THRESHOLD: f32 = 0.80;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrelationMethod {
    Anagram,
    OneLetterDifference,
    OneLetterAdditionOrRemoval,
    SemanticRelation,
}

impl CorrelationMethod {
    pub fn description(self) -> &'static str {
        match self {
            Self::Anagram => "anagrammi",
            Self::OneLetterDifference => "differiscono per una lettera",
            Self::OneLetterAdditionOrRemoval => {
                "si ottengono aggiungendo o rimuovendo una lettera"
            }
            Self::SemanticRelation => "in relazione semantica",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrelationResult {
    pub word_a: String,
    pub word_b: String,
    pub score: f32,
    pub is_correlated: bool,
    pub matched_methods: Vec<CorrelationMethod>,
}

pub struct CorrelationCalculator {
    embedder: OnnxEmbedder,
}

impl CorrelationCalculator {
    pub fn new(config: CorrelationConfig) -> Result<Self> {
        let model_dir = config.model_dir.ok_or_else(|| {
            CorrelationError::InvalidInput(
                "modello ONNX obbligatorio: imposta --model-dir (directory o ZIP) o ILBERSAGLIO_MODEL_DIR".to_string(),
            )
        })?;
        let embedder = OnnxEmbedder::from_model_dir(model_dir)?;

        Ok(Self { embedder })
    }

    pub fn calculate(&self, word_a: &str, word_b: &str) -> Result<CorrelationResult> {
        let word_a = word_a.trim();
        let word_b = word_b.trim();

        if word_a.is_empty() || word_b.is_empty() {
            return Err(CorrelationError::InvalidInput(
                "inserire due parole non vuote".to_string(),
            ));
        }

        let emb_a = self.embedder.encode(word_a)?;
        let emb_b = self.embedder.encode(word_b)?;
        let score = cosine_to_unit_interval(&emb_a, &emb_b);
        let matched_methods = collect_correlation_methods(word_a, word_b, score);

        Ok(CorrelationResult {
            word_a: word_a.to_string(),
            word_b: word_b.to_string(),
            score,
            is_correlated: !matched_methods.is_empty(),
            matched_methods,
        })
    }
}

fn collect_correlation_methods(
    word_a: &str,
    word_b: &str,
    semantic_score: f32,
) -> Vec<CorrelationMethod> {
    let normalized_a = normalize_for_comparison(word_a);
    let normalized_b = normalize_for_comparison(word_b);
    let chars_a: Vec<char> = normalized_a.chars().collect();
    let chars_b: Vec<char> = normalized_b.chars().collect();

    let mut methods = Vec::new();

    if are_anagrams(&chars_a, &chars_b) {
        methods.push(CorrelationMethod::Anagram);
    }

    if differ_by_one_letter(&chars_a, &chars_b) {
        methods.push(CorrelationMethod::OneLetterDifference);
    }

    if differ_by_addition_or_removal(&chars_a, &chars_b) {
        methods.push(CorrelationMethod::OneLetterAdditionOrRemoval);
    }

    if methods.is_empty() && has_semantic_relation(semantic_score) {
        methods.push(CorrelationMethod::SemanticRelation);
    }

    methods
}

fn normalize_for_comparison(word: &str) -> String {
    word.trim().to_lowercase()
}

fn are_anagrams(word_a: &[char], word_b: &[char]) -> bool {
    if word_a.len() != word_b.len() {
        return false;
    }

    let mut sorted_a = word_a.to_vec();
    let mut sorted_b = word_b.to_vec();
    sorted_a.sort_unstable();
    sorted_b.sort_unstable();
    sorted_a == sorted_b
}

fn differ_by_one_letter(word_a: &[char], word_b: &[char]) -> bool {
    if word_a.len() != word_b.len() {
        return false;
    }

    word_a
        .iter()
        .zip(word_b.iter())
        .filter(|(left, right)| left != right)
        .count()
        == 1
}

fn differ_by_addition_or_removal(word_a: &[char], word_b: &[char]) -> bool {
    match word_a.len().cmp(&word_b.len()) {
        std::cmp::Ordering::Less if word_b.len() - word_a.len() == 1 => {
            is_single_char_insertion(word_a, word_b)
        }
        std::cmp::Ordering::Greater if word_a.len() - word_b.len() == 1 => {
            is_single_char_insertion(word_b, word_a)
        }
        _ => false,
    }
}

fn is_single_char_insertion(shorter: &[char], longer: &[char]) -> bool {
    let mut shorter_index = 0;
    let mut longer_index = 0;
    let mut skipped = false;

    while shorter_index < shorter.len() && longer_index < longer.len() {
        if shorter[shorter_index] == longer[longer_index] {
            shorter_index += 1;
            longer_index += 1;
            continue;
        }

        if skipped {
            return false;
        }

        skipped = true;
        longer_index += 1;
    }

    true
}

fn has_semantic_relation(score: f32) -> bool {
    score >= SEMANTIC_RELATION_THRESHOLD
}

fn cosine_to_unit_interval(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    ((dot + 1.0) / 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_is_required_when_not_configured() {
        let cfg = CorrelationConfig::default();
        let result = CorrelationCalculator::new(cfg);
        assert!(matches!(result, Err(CorrelationError::InvalidInput(_))));
    }

    #[test]
    fn detects_anagrams_case_insensitively() {
        let methods = collect_correlation_methods("Roma", "Amor", 0.2);
        assert_eq!(methods, vec![CorrelationMethod::Anagram]);
    }

    #[test]
    fn detects_one_letter_difference() {
        let methods = collect_correlation_methods("cane", "pane", 0.2);
        assert_eq!(methods, vec![CorrelationMethod::OneLetterDifference]);
    }

    #[test]
    fn detects_one_letter_addition_or_removal() {
        let methods = collect_correlation_methods("casa", "casae", 0.2);
        assert_eq!(
            methods,
            vec![CorrelationMethod::OneLetterAdditionOrRemoval]
        );
    }

    #[test]
    fn detects_semantic_relation_from_threshold() {
        let methods = collect_correlation_methods("sole", "stella", SEMANTIC_RELATION_THRESHOLD);
        assert_eq!(methods, vec![CorrelationMethod::SemanticRelation]);
    }

    #[test]
    fn semantic_relation_is_ignored_when_lexical_method_matches() {
        let methods = collect_correlation_methods("Roma", "Amor", 0.9);
        assert_eq!(methods, vec![CorrelationMethod::Anagram]);
    }
}
