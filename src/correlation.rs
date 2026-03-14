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
            model_dir: std::env::var("ILBERSAGLIO_MODEL_DIR")
                .ok()
                .map(PathBuf::from),
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
            Self::OneLetterAdditionOrRemoval => "si ottengono aggiungendo o rimuovendo una lettera",
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

#[derive(Debug, Clone, Serialize)]
pub struct CorrelationChainStep {
    pub word_a: String,
    pub word_b: String,
    pub score: f32,
    pub matched_methods: Vec<CorrelationMethod>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrelationChainResult {
    pub input_words: Vec<String>,
    pub path: Vec<String>,
    pub is_correlated: bool,
    pub steps: Vec<CorrelationChainStep>,
}

#[derive(Debug, Clone)]
struct PreparedWord {
    word: String,
    embedding: Vec<f32>,
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
        let word_a = normalize_word(word_a);
        let word_b = normalize_word(word_b);

        if word_a.is_empty() || word_b.is_empty() {
            return Err(CorrelationError::InvalidInput(
                "inserire due parole non vuote".to_string(),
            ));
        }

        let emb_a = self.embedder.encode(&word_a)?;
        let emb_b = self.embedder.encode(&word_b)?;
        let score = cosine_to_unit_interval(&emb_a, &emb_b);
        let matched_methods = collect_correlation_methods(&word_a, &word_b, score);

        Ok(CorrelationResult {
            word_a,
            word_b,
            score,
            is_correlated: !matched_methods.is_empty(),
            matched_methods,
        })
    }

    pub fn calculate_chain(&self, words: &[String]) -> Result<CorrelationChainResult> {
        let normalized_words = normalize_chain_words(words)?;
        let prepared_words = self.prepare_words(&normalized_words)?;

        Ok(build_chain_result(&prepared_words))
    }

    fn prepare_words(&self, words: &[String]) -> Result<Vec<PreparedWord>> {
        let mut prepared_words = Vec::with_capacity(words.len());

        for word in words {
            prepared_words.push(PreparedWord {
                word: word.clone(),
                embedding: self.embedder.encode(word)?,
            });
        }

        Ok(prepared_words)
    }
}

fn normalize_word(word: &str) -> String {
    word.trim()
        .chars()
        .flat_map(|character| character.to_uppercase())
        .collect()
}

fn normalize_chain_words(words: &[String]) -> Result<Vec<String>> {
    if words.len() < 2 {
        return Err(CorrelationError::InvalidInput(
            "inserire almeno due parole non vuote".to_string(),
        ));
    }

    let normalized_words = words
        .iter()
        .map(|word| normalize_word(word))
        .collect::<Vec<_>>();

    if normalized_words.iter().any(|word| word.is_empty()) {
        return Err(CorrelationError::InvalidInput(
            "inserire almeno due parole non vuote".to_string(),
        ));
    }

    Ok(normalized_words)
}

fn build_chain_result(prepared_words: &[PreparedWord]) -> CorrelationChainResult {
    let input_words = prepared_words
        .iter()
        .map(|prepared_word| prepared_word.word.clone())
        .collect::<Vec<_>>();

    let target_idx = prepared_words.len() - 1;
    let mut current_idx = 0usize;
    let mut visited = vec![false; prepared_words.len()];
    let mut path = vec![prepared_words[current_idx].word.clone()];
    let mut steps = Vec::new();

    visited[current_idx] = true;

    while current_idx != target_idx {
        let candidates = collect_related_candidates(
            prepared_words,
            current_idx,
            target_idx,
            &visited,
            has_unvisited_non_target(&visited, target_idx),
        );

        let Some((next_idx, step)) = select_best_candidate(candidates) else {
            return CorrelationChainResult {
                input_words,
                path: Vec::new(),
                is_correlated: false,
                steps,
            };
        };

        if next_idx != target_idx {
            visited[next_idx] = true;
        }

        path.push(prepared_words[next_idx].word.clone());
        steps.push(step);
        current_idx = next_idx;
    }

    CorrelationChainResult {
        input_words,
        path,
        is_correlated: true,
        steps,
    }
}

fn collect_related_candidates(
    prepared_words: &[PreparedWord],
    current_idx: usize,
    target_idx: usize,
    visited: &[bool],
    must_delay_target: bool,
) -> Vec<(usize, CorrelationChainStep)> {
    let mut candidates = Vec::new();

    for candidate_idx in 0..prepared_words.len() {
        if candidate_idx == current_idx {
            continue;
        }

        if must_delay_target && candidate_idx == target_idx {
            continue;
        }

        if candidate_idx != target_idx && visited[candidate_idx] {
            continue;
        }

        let score = cosine_to_unit_interval(
            &prepared_words[current_idx].embedding,
            &prepared_words[candidate_idx].embedding,
        );
        let matched_methods = collect_correlation_methods(
            &prepared_words[current_idx].word,
            &prepared_words[candidate_idx].word,
            score,
        );

        if matched_methods.is_empty() {
            continue;
        }

        candidates.push((
            candidate_idx,
            CorrelationChainStep {
                word_a: prepared_words[current_idx].word.clone(),
                word_b: prepared_words[candidate_idx].word.clone(),
                score,
                matched_methods,
            },
        ));
    }

    candidates
}

fn has_unvisited_non_target(visited: &[bool], target_idx: usize) -> bool {
    visited
        .iter()
        .enumerate()
        .any(|(idx, is_visited)| idx != target_idx && !is_visited)
}

fn select_best_candidate(
    candidates: Vec<(usize, CorrelationChainStep)>,
) -> Option<(usize, CorrelationChainStep)> {
    let has_lexical_candidate = candidates
        .iter()
        .any(|(_, step)| is_lexical_candidate(step));

    candidates
        .into_iter()
        .filter(|(_, step)| !has_lexical_candidate || is_lexical_candidate(step))
        .max_by(|(idx_a, step_a), (idx_b, step_b)| {
            step_a
                .score
                .total_cmp(&step_b.score)
                .then_with(|| idx_b.cmp(idx_a))
        })
}

fn is_lexical_candidate(step: &CorrelationChainStep) -> bool {
    step.matched_methods
        .iter()
        .any(|method| *method != CorrelationMethod::SemanticRelation)
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
    word.chars()
        .filter(|character| !character.is_whitespace() && !character.is_ascii_punctuation())
        .map(strip_accent)
        .flat_map(|character| character.to_uppercase())
        .collect()
}

fn strip_accent(character: char) -> char {
    match character {
        'à' | 'á' | 'â' | 'ä' | 'À' | 'Á' | 'Â' | 'Ä' => 'a',
        'è' | 'é' | 'ê' | 'ë' | 'È' | 'É' | 'Ê' | 'Ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' | 'Ì' | 'Í' | 'Î' | 'Ï' => 'i',
        'ò' | 'ó' | 'ô' | 'ö' | 'Ò' | 'Ó' | 'Ô' | 'Ö' => 'o',
        'ù' | 'ú' | 'û' | 'ü' | 'Ù' | 'Ú' | 'Û' | 'Ü' => 'u',
        _ => character,
    }
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

    fn prepared_with_embedding(word: &str, embedding: Vec<f32>) -> PreparedWord {
        PreparedWord {
            word: word.to_string(),
            embedding,
        }
    }

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
    fn normalize_word_uses_uppercase() {
        assert_eq!(normalize_word("  Roma mia  "), "ROMA MIA");
    }

    #[test]
    fn lexical_methods_ignore_spaces_and_punctuation() {
        let methods = collect_correlation_methods("Roma!", "A mor", 0.2);
        assert_eq!(methods, vec![CorrelationMethod::Anagram]);
    }

    #[test]
    fn normalize_for_comparison_removes_accents() {
        assert_eq!(normalize_for_comparison("perché!"), "PERCHE");
    }

    #[test]
    fn detects_one_letter_difference() {
        let methods = collect_correlation_methods("cane", "pane", 0.2);
        assert_eq!(methods, vec![CorrelationMethod::OneLetterDifference]);
    }

    #[test]
    fn one_letter_difference_uses_normalized_words() {
        let methods = collect_correlation_methods("ca-ne", "pane ", 0.2);
        assert_eq!(methods, vec![CorrelationMethod::OneLetterDifference]);
    }

    #[test]
    fn one_letter_difference_ignores_accents() {
        let methods = collect_correlation_methods("pèsca", "pesco", 0.2);
        assert_eq!(methods, vec![CorrelationMethod::OneLetterDifference]);
    }

    #[test]
    fn detects_one_letter_addition_or_removal() {
        let methods = collect_correlation_methods("casa", "casae", 0.2);
        assert_eq!(methods, vec![CorrelationMethod::OneLetterAdditionOrRemoval]);
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

    #[test]
    fn normalize_chain_words_requires_at_least_two_non_empty_words() {
        let error = normalize_chain_words(&[" ".to_string()]).unwrap_err();
        assert!(matches!(error, CorrelationError::InvalidInput(_)));
    }

    #[test]
    fn build_chain_result_selects_highest_semantic_candidate() {
        let prepared_words = vec![
            prepared_with_embedding("SOLE", vec![1.0, 0.0]),
            prepared_with_embedding("MARE", vec![0.70, 0.0]),
            prepared_with_embedding("VENTO", vec![0.98, 0.0]),
            prepared_with_embedding("VETNO", vec![0.98, 0.0]),
        ];

        let result = build_chain_result(&prepared_words);

        assert!(result.is_correlated);
        assert_eq!(result.path, vec!["SOLE", "VENTO", "MARE", "VETNO"]);
        assert_eq!(result.steps.len(), 3);
        assert_eq!(
            result.steps[0].matched_methods,
            vec![CorrelationMethod::SemanticRelation]
        );
        assert_eq!(result.steps[0].word_b, "VENTO");
        assert_eq!(
            result.steps[1].matched_methods,
            vec![CorrelationMethod::SemanticRelation]
        );
        assert_eq!(
            result.steps[2].matched_methods,
            vec![CorrelationMethod::SemanticRelation]
        );
    }

    #[test]
    fn build_chain_result_includes_all_words_before_target() {
        let prepared_words = vec![
            prepared_with_embedding("OMERO", vec![1.0, 0.0]),
            prepared_with_embedding("TROIA", vec![0.93, 0.0]),
            prepared_with_embedding("ODISSEA", vec![0.98, 0.0]),
            prepared_with_embedding("ASSEDIO", vec![0.99, 0.0]),
        ];

        let result = build_chain_result(&prepared_words);

        assert!(result.is_correlated);
        assert_eq!(result.path, vec!["OMERO", "ODISSEA", "TROIA", "ASSEDIO"]);
        assert_eq!(result.steps.len(), 3);
        assert_eq!(result.steps[0].word_b, "ODISSEA");
        assert_eq!(result.steps[1].word_b, "TROIA");
        assert_eq!(result.steps[2].word_b, "ASSEDIO");
    }

    #[test]
    fn build_chain_result_fails_when_no_related_candidate_exists() {
        let prepared_words = vec![
            prepared_with_embedding("SOLE", vec![1.0, 0.0]),
            prepared_with_embedding("MARE", vec![-1.0, 0.0]),
            prepared_with_embedding("VENTO", vec![0.0, 1.0]),
        ];

        let result = build_chain_result(&prepared_words);

        assert!(!result.is_correlated);
        assert!(result.path.is_empty());
        assert!(result.steps.is_empty());
    }

    #[test]
    fn select_best_candidate_prioritizes_lexical_over_semantic() {
        let lexical = (
            1,
            CorrelationChainStep {
                word_a: "ODISSEA".to_string(),
                word_b: "ASSEDIO".to_string(),
                score: 0.88,
                matched_methods: vec![CorrelationMethod::Anagram],
            },
        );
        let semantic = (
            2,
            CorrelationChainStep {
                word_a: "ODISSEA".to_string(),
                word_b: "TROIA".to_string(),
                score: 0.94,
                matched_methods: vec![CorrelationMethod::SemanticRelation],
            },
        );

        let selected = select_best_candidate(vec![semantic, lexical]).unwrap();
        assert_eq!(selected.1.word_b, "ASSEDIO");
        assert_eq!(selected.1.matched_methods, vec![CorrelationMethod::Anagram]);
    }
}
