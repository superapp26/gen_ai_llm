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

#[test]
fn step7_split() {
    let tokenizer = Tokenizer::from_merges(GPT4_PATTERN, vec![]).unwrap();

    assert_eq!(
        tokenizer.split("Hello world!! 123456 don't").unwrap(),
        vec!["Hello", " world", "!!", " ", "123", "456", " don", "'t"]
    );

    assert_eq!(
        tokenizer.split("    return 1").unwrap(),
        vec!["   ", " return", " ", "1"]
    );
}

#[test]
fn step8_train_on_chunks() {
    let tokenizer = Tokenizer::train(SAMPLE, 300, GPT4_PATTERN).unwrap();
    assert!(tokenizer.vocab_size() <= 300);

    for id in 256..tokenizer.vocab_size() as u32 {
        let bytes = tokenizer.token_bytes(id).unwrap();

        assert!(
            !bytes.windows(2).any(|w| w[0] != b' ' && w[1] == b' '),
            "{:?}",
            String::from_utf8_lossy(bytes)
        )
    }
}

#[test]
fn test_training_updates_chunk_sequences() {
    let tokenizer = Tokenizer::train("abc abc", 258, GPT4_PATTERN).unwrap();

    assert_eq!(tokenizer.merges(), &[(97, 98), (256, 99)]);
}

#[test]
fn step10_encode_special_tokens() {
    let mut tok = Tokenizer::from_merges(
        GPT4_PATTERN,
        vec![],
    ).unwrap();

    tok.register_special_tokens(&[
        "<s>",
        "<|endoftext|>",
    ]);

    let start_id = tok.special_token_id("<s>").unwrap();
    let end_id = tok.special_token_id("<|endoftext|>").unwrap();

    // Test 1: Disallowed token must be rejected.
    assert!(matches!(
        tok.encode("<s>Hello", &[]),
        Err(TokenizerError::DisallowedTokenSpecial(_))
    ));

    // Test 2: Explicitly allowed special token.
    let ids = tok.encode(
        "<s>Hello",
        &["<s>"],
    ).unwrap();

    assert_eq!(ids.first(), Some(&start_id));

    // Test 3: Multiple allowed special tokens.
    let ids = tok.encode(
        "<s>Hello<|endoftext|>",
        &["<s>", "<|endoftext|>"],
    ).unwrap();

    assert_eq!(ids.first(), Some(&start_id));
    assert_eq!(ids.last(), Some(&end_id));

    // Test 4: Ordinary encoding never emits special IDs.
    let ordinary_ids = tok.encode_ordinary("<s>Hello").unwrap();

    assert!(!ordinary_ids.contains(&start_id));
}

