pub mod features;

use fancy_regex::Regex;
use features::merge_pair;
use std::clone;
use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::features::text_to_bytes_converter;

pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
pub const MIN_PAIR_COUNT: usize = 2;
pub const INIT_VOCAB_SIZE: u32 = 256;
//pub const MAX_VOCAB_SIZE: u32 = 50_000;
const TOKENIZER_FILE_VERSION: u32 = 1;

#[derive(Debug)]
pub enum TokenizerError {
    VocabTooSmall(String),
    UnknownTokenId(u32),
    DisallowedTokenSpecial(String),
    InvalidMergePair(usize),
    Regex(Box<fancy_regex::Error>),
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedFileVersion(u32),
    InvalidSpecialToken {token: String, id: u32},
}

pub struct Tokenizer {
    pattern: String,
    regex: Regex,
    merges: Vec<(u32, u32)>,
    merge_map: HashMap<(u32, u32), u32>,
    vocab: Vec<Vec<u8>>,
    special: HashMap<String, u32>,
    special_inv: HashMap<u32, String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TokenizerFile {
    version: u32,
    pattern: String,
    merges: Vec<(u32, u32)>,
    special_tokens: BTreeMap<String, u32>
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::VocabTooSmall(msg) => write!(f, "vocab too small: {}", msg),
            TokenizerError::UnknownTokenId(id) => write!(f, "Unknown token id {}", id),
            TokenizerError::DisallowedTokenSpecial(token) => {
                write!(f, "Disallowed special token: {}", token)
            }
            TokenizerError::InvalidMergePair(index) => {
                write!(f, "merge {index} uses a token id that does not exist yet")
            }
            TokenizerError::Regex(regex) => write!(f, "Invalid Regex: {}", regex),
            TokenizerError::Io(error) => write!(f, "IO Error: {}", error),
            TokenizerError::Json(error) => write!(f, "JSON Error: {}", error),
            TokenizerError::UnsupportedFileVersion(version) => write!(f, "The version {} is invalid", version),
            TokenizerError::InvalidSpecialToken { token, id } => {
                write!(f, "Invalid special token {token:?} with ID {id:?}")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<fancy_regex::Error> for TokenizerError {
    fn from(err: fancy_regex::Error) -> Self {
        TokenizerError::Regex(Box::new(err))
    }
}

impl From<std::io::Error> for TokenizerError {
    fn from(err: std::io::Error) -> Self {
        TokenizerError::Io(err)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(err: serde_json::Error) -> Self {
        TokenizerError::Json(err)
    }
}

impl Tokenizer {
    pub fn from_merges(pattern: &str, merges: Vec<(u32, u32)>) -> Result<Self, TokenizerError> {
        let regex = Regex::new(pattern)?;
        let mut merge_map = HashMap::with_capacity(merges.len());
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(INIT_VOCAB_SIZE as usize + merges.len());
        vocab.extend((0..INIT_VOCAB_SIZE).map(|i| vec![i as u8]));

        for (i, &(first, second)) in merges.iter().enumerate() {
            let new_token_id = INIT_VOCAB_SIZE + i as u32;
            if first >= new_token_id || second >= new_token_id {
                return Err(TokenizerError::InvalidMergePair(i));
            }

            let mut bytes = vocab[first as usize].clone();
            bytes.extend_from_slice(&vocab[second as usize]);

            vocab.push(bytes);

            merge_map.insert((first, second), new_token_id);
        }

        Ok(Tokenizer {
            pattern: pattern.to_string(),
            regex,
            merges,
            merge_map,
            vocab,
            special: HashMap::new(),
            special_inv: HashMap::new(),
        })
    }

    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    pub fn token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.vocab.get(token_id as usize).map(|v| v.as_slice())
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + self.special.len()
    }

    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.special.get(token).copied()
    }

    pub fn split<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, TokenizerError> {
        let mut chunks: Vec<&str> = Vec::new();

        for m in self.regex.find_iter(text) {
            chunks.push(m?.as_str());
        }

        Ok(chunks)
    }

    pub fn train(text: &str, vocab_size: u32, pattern: &str) -> Result<Self, TokenizerError> {
        if vocab_size < INIT_VOCAB_SIZE {
            return Err(TokenizerError::VocabTooSmall(
                "Vocab can not be less than 256".to_string(),
            ));
        }

        let mut regex_chunks: HashMap<&str, usize> = HashMap::new();

        let mut bytes_frequency_mapping: Vec<(Vec<u32>, usize)> = Vec::new();

        let regex = Regex::new(pattern)?;

        for result in regex.find_iter(text) {
            let chunk = result?.as_str();

            *regex_chunks.entry(chunk).or_insert(0) += 1;
        }

        let mut learned_merges: Vec<(u32, u32)> = Vec::new();

        for (chunk, frequency) in regex_chunks {
            let ids = text_to_bytes_converter(chunk);
            bytes_frequency_mapping.push((ids, frequency));
        }

        while INIT_VOCAB_SIZE as usize + learned_merges.len() < vocab_size as usize {
            let mut pair_frequencies: HashMap<(u32, u32), usize> = HashMap::new();

            for (ids, frequency) in bytes_frequency_mapping.iter() {
                for w in ids.windows(2) {
                    let pair = (w[0], w[1]);

                    *pair_frequencies.entry(pair).or_insert(0) += *frequency;
                }
            }

            let best_pair = pair_frequencies
                .iter()
                .max_by_key(|&(pair, count)| (*count, Reverse(*pair)))
                .map(|(&pair, &count)| (pair, count));

            match best_pair {
                Some((selected_pair, best_count)) => {
                    // Continue processing
                    if best_count < MIN_PAIR_COUNT {
                        break;
                    }

                    let new_token_id = INIT_VOCAB_SIZE + learned_merges.len() as u32;

                    learned_merges.push(selected_pair);

                    for (ids, _freq) in bytes_frequency_mapping.iter_mut() {
                        let merged = merge_pair(ids, selected_pair, new_token_id);

                        *ids = merged;
                    }
                }
                None => {
                    // Stop loop
                    break;
                }
            }
        }

        Self::from_merges(pattern, learned_merges)
    }

    pub fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut chunk_ids = text_to_bytes_converter(chunk);

        while chunk_ids.len() > 1 {
            let best_pair = chunk_ids
                .windows(2)
                .filter_map(|window| {
                    let pair = (window[0], window[1]);

                    self.merge_map
                        .get(&pair)
                        .map(|&new_token_id| (pair, new_token_id))
                })
                .min_by_key(|&(_, new_token_id)| new_token_id);

            let Some((pair, new_token_id)) = best_pair else {
                break;
            };

            chunk_ids = merge_pair(&chunk_ids, pair, new_token_id);
        }

        chunk_ids
    }

    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let chunks = self.split(text)?;

        let mut token_ids: Vec<u32> = Vec::new();

        for chunk in chunks {
            let encoded_chunk = self.encode_chunk(chunk);

            token_ids.extend(encoded_chunk);
        }

        Ok(token_ids)
    }

    pub fn decode_bytes(&self, token_ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut bytes: Vec<u8> = Vec::new();

        for &id in token_ids {

            if let Some(special_text) = self.special_inv.get(&id) {
                bytes.extend_from_slice(special_text.as_bytes());
            } else {
                 let token_bytes = self
                .vocab
                .get(id as usize)
                .ok_or(TokenizerError::UnknownTokenId(id))?;

            bytes.extend_from_slice(token_bytes);
            }
            
        }

        Ok(bytes)
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let bytes = self.decode_bytes(token_ids)?;

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn register_special_tokens(&mut self, tokens: &[&str]) {

        let new_id = self.vocab_size() as u32;

        for &token in tokens {

            if token.is_empty() || self.special.contains_key(token) {
                continue;
            }
            self.special.insert(token.to_string(), new_id);
            self.special_inv.insert(new_id, token.to_string());
        }

    }

    pub fn encode(&self, text: &str, allowed: &[&str]) -> Result<Vec<u32>, TokenizerError>{

        let mut remaining = text;

        let mut token_ids: Vec<u32> = Vec::new();

        loop {

            let mut best_match: Option<(usize, &str, u32)> = None;
            
            for ( special_text, &special_id) in &self.special {

                // We first prevent special tokens that are empty to match forever

                if special_text.is_empty() {
                    continue;
                }

                // Look for the special registered token
                
                if let  Some(position) = remaining.find(special_text.as_str()) {

                    let should_replace = match best_match {

                        None => true,
                        
                        Some((best_position, best_text, _)) => {
                            position < best_position || (position == best_position && special_text.len() > best_text.len())
                        }
                        
                    };

                    // Retain the earliest match and also in the case of ties prefer the longer token

                    if should_replace {

                        best_match = Some(
                            (
                                position, 
                                special_text.as_str(),
                                special_id
                            )
                        )
                    }

            }
        }

        let Some((position, special_text, special_token_id)) = best_match else {

            token_ids.extend(self.encode_ordinary(remaining)?);

            return  Ok(token_ids);
        };

        // Rejected tokens that are not explicitly allowed

        if !allowed.contains(&special_text) {
            return  Err(
                TokenizerError::DisallowedTokenSpecial(
                    special_text.to_string(),
                )
            );
        }

        let prefix = &remaining[..position];

        token_ids.extend(self.encode_ordinary(prefix)?);

        token_ids.push(special_token_id);

        remaining = &remaining[position + special_text.len()..];


    }
}

  pub fn save(&self, path: impl AsRef<Path>) -> Result<(), TokenizerError> {

    let file = TokenizerFile {
        version: TOKENIZER_FILE_VERSION,
        pattern: self.pattern.clone(),
        merges: self.merges.clone(),
        special_tokens: self
                    .special
                    .iter()
                    .map(|(token, &id)| (token.clone(), id))
                    .collect(),
    };

    let json = serde_json::to_vec_pretty(&file)?;
    std::fs::write(path, json)?;

    Ok(())


  }

  pub fn  load(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
      let json = std::fs::read(path)?;
      let file: TokenizerFile = serde_json::from_slice(&json)?;

      if file.version != TOKENIZER_FILE_VERSION {
        return Err(
            TokenizerError::UnsupportedFileVersion(file.version)
        )
      }

      let mut tokenizer = Self::from_merges(&file.pattern, file.merges)?;

      let oridinary_vocab_len = tokenizer.vocab.len();

      let special_token_count = file.special_tokens.len();

      let valid_special_ids = oridinary_vocab_len..(oridinary_vocab_len + special_token_count);

      for (token, id) in file.special_tokens {

       if token.is_empty() 
                || !valid_special_ids.contains(&(id as usize))
                || tokenizer.special_inv.contains_key(&id) {

                    return Err(
                        TokenizerError::InvalidSpecialToken { token, id }
                    );
                }

        tokenizer.special.insert(token.clone(), id);
        tokenizer.special_inv.insert(id, token.clone());
      }

      Ok(tokenizer)

  }
}
