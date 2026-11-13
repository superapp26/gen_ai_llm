pub mod features;

use fancy_regex::Regex;
use features::{merge_pair, text_to_bytes_converter};
use std::{collections::HashMap, hash::Hash};

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
}
