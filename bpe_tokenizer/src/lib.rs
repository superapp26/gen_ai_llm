pub mod features;

use fancy_regex::Regex;
use features::merge_pair;
use std::cmp::Reverse;
use std::collections::HashMap;

use crate::features::text_to_bytes_converter;

pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
pub const MIN_PAIR_COUNT: usize = 2;
pub const INIT_VOCAB_SIZE: u32 = 256;
//pub const MAX_VOCAB_SIZE: u32 = 50_000;

#[derive(Debug)]
pub enum TokenizerError {
    VocabTooSmall(String),
    UnknownTokenId(u32),
    DisallowedTokenSpecial(String),
    InvalidMergePair(usize),
    Regex(Box<fancy_regex::Error>),
    Io(std::io::Error),
    Json(serde_json::Error),
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
            let token_bytes = self
                .vocab
                .get(id as usize)
                .ok_or(TokenizerError::UnknownTokenId(id))?;

            bytes.extend_from_slice(token_bytes);
        }

        Ok(bytes)
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let bytes = self.decode_bytes(token_ids)?;

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
