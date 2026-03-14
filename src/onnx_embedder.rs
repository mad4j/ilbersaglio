use std::{
    ffi::OsStr,
    fs,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tract_onnx::prelude::*;
use zip::ZipArchive;

use crate::error::{CorrelationError, Result};

const DEFAULT_MAX_LEN: usize = 24;

pub struct OnnxEmbedder {
    session: TypedRunnableModel<TypedModel>,
    tokenizer: Tokenizer,
    max_len: usize,
}

impl OnnxEmbedder {
    pub fn from_model_dir(model_source: impl AsRef<Path>) -> Result<Self> {
        let model_source = model_source.as_ref();

        if model_source.is_dir() {
            let model_path = model_source.join("model.onnx");
            let tokenizer_path = model_source.join("tokenizer.json");

            if model_path.exists() && tokenizer_path.exists() {
                return Self::from_model_files(&model_path, &tokenizer_path);
            }

            if let Some(zip_path) = find_zip_archive_candidate(model_source)? {
                return Self::from_zip_archive(&zip_path);
            }

            return Self::from_model_files(&model_path, &tokenizer_path);
        }

        if model_source
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
        {
            return Self::from_zip_archive(model_source);
        }

        Err(CorrelationError::InvalidInput(format!(
            "percorso modello non valido: {model_source:?}. Usa una directory oppure un file .zip"
        )))
    }

    fn from_zip_archive(zip_path: &Path) -> Result<Self> {
        if !zip_path.exists() {
            return Err(CorrelationError::ModelNotFound(zip_path.to_path_buf()));
        }

        let archive_file = File::open(zip_path).map_err(|e| CorrelationError::Internal(e.to_string()))?;
        let mut archive = ZipArchive::new(archive_file).map_err(|e| CorrelationError::Internal(e.to_string()))?;

        let temp_dir = tempfile::Builder::new()
            .prefix("ilbersaglio-model-")
            .tempdir()
            .map_err(|e| CorrelationError::Internal(e.to_string()))?;

        let model_path = temp_dir.path().join("model.onnx");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        extract_zip_entry_by_basename(&mut archive, "model.onnx", &model_path)?;
        extract_zip_entry_by_basename(&mut archive, "tokenizer.json", &tokenizer_path)?;

        Self::from_model_files(&model_path, &tokenizer_path)
    }

    fn from_model_files(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {

        if !model_path.exists() {
            return Err(CorrelationError::ModelNotFound(model_path.to_path_buf()));
        }
        if !tokenizer_path.exists() {
            return Err(CorrelationError::TokenizerNotFound(tokenizer_path.to_path_buf()));
        }

        let mut tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| CorrelationError::Tokenizer(e.to_string()))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(DEFAULT_MAX_LEN),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        tokenizer.with_truncation(Some(TruncationParams {
            max_length: DEFAULT_MAX_LEN,
            ..Default::default()
        }))
        .map_err(|e| CorrelationError::Tokenizer(e.to_string()))?;

        let session = tract_onnx::onnx()
            .model_for_path(&model_path)
            .map_err(|e: TractError| CorrelationError::Ort(e.to_string()))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(i64::datum_type(), tvec!(1, DEFAULT_MAX_LEN as i64)),
            )
            .map_err(|e: TractError| CorrelationError::Ort(e.to_string()))?
            .with_input_fact(
                1,
                InferenceFact::dt_shape(i64::datum_type(), tvec!(1, DEFAULT_MAX_LEN as i64)),
            )
            .map_err(|e: TractError| CorrelationError::Ort(e.to_string()))?
            .into_optimized()
            .map_err(|e: TractError| CorrelationError::Ort(e.to_string()))?
            .into_runnable()
            .map_err(|e: TractError| CorrelationError::Ort(e.to_string()))?;

        Ok(Self {
            session,
            tokenizer,
            max_len: DEFAULT_MAX_LEN,
        })
    }

    pub fn model_path(model_dir: impl AsRef<Path>) -> PathBuf {
        model_dir.as_ref().join("model.onnx")
    }

    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| CorrelationError::Tokenizer(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|v| *v as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|v| *v as i64)
            .collect();

        let input_ids = tract_ndarray::Array2::from_shape_vec((1, self.max_len), ids)
            .map_err(|e| CorrelationError::Internal(e.to_string()))?;
        let attention_mask = tract_ndarray::Array2::from_shape_vec((1, self.max_len), mask.clone())
            .map_err(|e| CorrelationError::Internal(e.to_string()))?;

        let outputs = self
            .session
            .run(tvec!(input_ids.into_tensor().into(), attention_mask.into_tensor().into()))
            .map_err(|e: TractError| CorrelationError::Ort(e.to_string()))?;

        let tensor = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e: TractError| CorrelationError::Ort(e.to_string()))?;

        let view = tensor;
        if view.ndim() != 3 {
            return Err(CorrelationError::Internal(format!(
                "output atteso a 3 dimensioni, ricevuto {}",
                view.ndim()
            )));
        }

        let shape = view.shape();
        let seq_len = shape[1];
        let hidden = shape[2];

        let mut pooled = vec![0.0f32; hidden];
        let mut valid_tokens = 0.0f32;

        for token_idx in 0..seq_len {
            if mask[token_idx] == 0 {
                continue;
            }
            valid_tokens += 1.0;
            for dim in 0..hidden {
                pooled[dim] += view[[0, token_idx, dim]];
            }
        }

        if valid_tokens == 0.0 {
            return Err(CorrelationError::Internal(
                "nessun token valido dopo il pooling".to_string(),
            ));
        }

        for v in &mut pooled {
            *v /= valid_tokens;
        }

        Ok(l2_normalize(pooled))
    }
}

fn extract_zip_entry_by_basename(
    archive: &mut ZipArchive<File>,
    basename: &str,
    output_path: &Path,
) -> Result<()> {
    let mut matched_index = None;

    for idx in 0..archive.len() {
        let is_match = {
            let entry = archive
                .by_index(idx)
                .map_err(|e| CorrelationError::Internal(e.to_string()))?;

            Path::new(entry.name())
                .file_name()
                .is_some_and(|name| name == OsStr::new(basename))
        };

        if is_match {
            matched_index = Some(idx);
            break;
        }
    }

    let idx = matched_index.ok_or_else(|| {
        CorrelationError::InvalidInput(format!(
            "il file ZIP deve contenere {basename}"
        ))
    })?;

    let mut src = archive
        .by_index(idx)
        .map_err(|e| CorrelationError::Internal(e.to_string()))?;
    let mut dst = File::create(output_path).map_err(|e| CorrelationError::Internal(e.to_string()))?;
    io::copy(&mut src, &mut dst).map_err(|e| CorrelationError::Internal(e.to_string()))?;

    Ok(())
}

fn find_zip_archive_candidate(model_dir: &Path) -> Result<Option<PathBuf>> {
    let mut all_zip_paths = Vec::new();
    let mut compatible_zip_paths = Vec::new();

    let dir_entries = fs::read_dir(model_dir).map_err(|e| CorrelationError::Internal(e.to_string()))?;

    for entry in dir_entries {
        let entry = entry.map_err(|e| CorrelationError::Internal(e.to_string()))?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }
        if !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
        {
            continue;
        }

        all_zip_paths.push(path.clone());

        if zip_contains_model_files(&path)? {
            compatible_zip_paths.push(path);
        }
    }

    if compatible_zip_paths.len() == 1 {
        return Ok(compatible_zip_paths.into_iter().next());
    }

    if compatible_zip_paths.len() > 1 {
        let candidates = compatible_zip_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(CorrelationError::InvalidInput(format!(
            "directory modello ambigua: trovati piu ZIP compatibili ({candidates}). Passa direttamente lo ZIP desiderato con --model-dir"
        )));
    }

    if all_zip_paths.len() == 1 {
        return Ok(all_zip_paths.into_iter().next());
    }

    if all_zip_paths.len() > 1 {
        let candidates = all_zip_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(CorrelationError::InvalidInput(format!(
            "directory modello ambigua: trovati piu ZIP ({candidates}) ma nessuno contiene sia model.onnx che tokenizer.json. Passa direttamente lo ZIP corretto con --model-dir"
        )));
    }

    Ok(None)
}

fn zip_contains_model_files(zip_path: &Path) -> Result<bool> {
    let archive_file = File::open(zip_path).map_err(|e| CorrelationError::Internal(e.to_string()))?;
    let mut archive = ZipArchive::new(archive_file).map_err(|e| CorrelationError::Internal(e.to_string()))?;

    let mut has_model = false;
    let mut has_tokenizer = false;

    for idx in 0..archive.len() {
        let entry = archive
            .by_index(idx)
            .map_err(|e| CorrelationError::Internal(e.to_string()))?;

        if Path::new(entry.name())
            .file_name()
            .is_some_and(|name| name == OsStr::new("model.onnx"))
        {
            has_model = true;
        }

        if Path::new(entry.name())
            .file_name()
            .is_some_and(|name| name == OsStr::new("tokenizer.json"))
        {
            has_tokenizer = true;
        }

        if has_model && has_tokenizer {
            return Ok(true);
        }
    }

    Ok(false)
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
