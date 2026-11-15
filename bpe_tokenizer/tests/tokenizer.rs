use bpe_tokenizer::{GPT4_PATTERN, Tokenizer};

const SAMPLE: &str = "the cat sat on the mat. the dog sat on the log. \
don't stop: 1234567 cats & 89 dogs!\n    indented line\n";

fn nasty() -> Vec<&'static str> {
    vec![
        "",
        " ",
        "  two  spaces ",
        "tab\there",
        "line\r\nbreak",
        "e.g. 3.14",
        "def f():\n    return 1",
        "my_var_name",
        "Ġ",
        "<s>",
        "café 👋 🤗",
        "안녕하세요",
        "\u{0}",
    ]
}

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
fn step9_encode_roundtrip() {
    let tok = Tokenizer::train(SAMPLE, 300, GPT4_PATTERN).unwrap();
    for s in nasty().into_iter().chain([SAMPLE]) {
        let ids = tok.encode_ordinary(s).unwrap();
        assert!(ids.len() <= s.len());
        assert_eq!(tok.decode(&ids).unwrap(), s);
    }
}
