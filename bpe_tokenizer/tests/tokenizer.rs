use bpe_tokenizer::features::{train, vocab_builder};
use bpe_tokenizer::{GPT4_PATTERN, Tokenizer, TokenizerError};

const SAMPLE: &str = "the cat sat on the mat. the dog sat on the log. \
don't stop: 1234567 cats & 89 dogs!\n    indented line\n";

// ---------- Step 6: Tokenizer type and errors ----------

#[test]
fn step6_from_merges_builds_vocab() {
    // 256 = "a" + "a" = "aa", 257 = "aa" + "b" = "aab"
    let tokenizer = Tokenizer::from_merges(GPT4_PATTERN, vec![(97, 97), (256, 98)]).unwrap();

    assert_eq!(tokenizer.vocab_size(), 258);
    assert_eq!(tokenizer.merges(), &[(97, 97), (256, 98)]);

    // the 256 single bytes come first
    assert_eq!(tokenizer.token_bytes(0), Some(&[0u8][..]));
    assert_eq!(tokenizer.token_bytes(97), Some(&b"a"[..]));
    assert_eq!(tokenizer.token_bytes(255), Some(&[255u8][..]));

    // then one token per merge: bytes, not ids
    assert_eq!(tokenizer.token_bytes(256), Some(&b"aa"[..]));
    assert_eq!(tokenizer.token_bytes(257), Some(&b"aab"[..]));

    // an id past the end is None, not a panic
    assert_eq!(tokenizer.token_bytes(258), None);
    assert_eq!(tokenizer.token_bytes(u32::MAX), None);
}

#[test]
fn step6_no_merges_just_bytes() {
    let tokenizer = Tokenizer::from_merges(GPT4_PATTERN, vec![]).unwrap();

    assert_eq!(tokenizer.vocab_size(), 256);
    assert!(tokenizer.merges().is_empty());
    assert_eq!(tokenizer.token_bytes(255), Some(&[255u8][..]));
    assert_eq!(tokenizer.token_bytes(256), None);
}
