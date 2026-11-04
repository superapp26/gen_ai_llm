use std::collections::HashMap;

use bpe_tokenizer::features::*;

#[test]
fn step1_stats_and_merge() {
    assert_eq!(common_pairs_counter(&[1, 2, 3, 1, 2])[&(1, 2)], 2);
    assert_eq!(merge_pair(&[1, 2, 3, 1, 2], (1, 2), 4), vec![4, 3, 4]);
    assert_eq!(merge_pair(&[1, 1, 1], (1, 1), 2), vec![2, 1]);
    assert_eq!(merge_pair(&[], (1, 2), 4), Vec::<u32>::new());
    assert_eq!(most_common_pair(&[7]), None);
    assert_eq!(most_common_pair(&[5, 4, 3]), Some(((4, 3), 1)));
}

#[test]
fn step2_train() {
    let merges = train("aaabdaaabac", 258);
    assert_eq!(merges[0], ((97, 97), 256));
    assert_eq!(merges[1], ((97, 98), 257));
    assert!(train("Hello", 256).is_empty());
    assert_eq!(train("ab", 1000).len(), 1);
}

#[test]
fn step3_vocab_builder() {
    let vocab = vocab_builder(&train("aaabdaaabac", 258));
    assert_eq!(vocab.len(), 258);
    assert_eq!(vocab[256], b"aa");
    assert_eq!(vocab[257], b"ab");

    // a merge built from two merged tokens
    let vocab_2 = vocab_builder(&train("aaabdaaabac", 259));
    assert_eq!(vocab_2[258], b"aaab");
    // no merges → just the 256 bytes
    assert_eq!(vocab_builder(&[]).len(), 256);
}

#[test]
fn step4_decode() {
    let vocab = vocab_builder(&train("aaabdaaabac", 257));
    assert_eq!(decode(&[256, 98], &vocab), "aab");

    assert_eq!(decode(&[0xE2], &vocab), "\u{FFFD}");

    // raw byte ids are decoded back to the exact original text
    let text = "héllo €";
    assert_eq!(decode(&text_to_bytes_converter(text), &vocab), text);

    // vocab[256] here is [97, 195] = "a" + half of "é"
    let v = vocab_builder(&train("aéaé", 257));
    assert_eq!(decode(&[256, 169, 256, 169], &v), "aéaé");

    assert_eq!(decode(&[], &vocab), "");
}

#[test]

fn step5_encode() {
    let merges = train("aaabdaaabac", 257);
    let map: HashMap<(u32, u32), u32> = merges.iter().cloned().collect();
    assert_eq!(
        encode("aaabdaaabac", &map),
        vec![256, 97, 98, 100, 256, 97, 98, 97, 99]
    );

    let vocab = vocab_builder(&merges);

    for s in ["", "a", "aaaa", "안녕하세요 👋 hello world 🤗"] {
        assert_eq!(decode(&encode(s, &map), &vocab), s);
    }

    // Testing which one wins between the smallest id "aa" (256) wins over "ab" (257). The smallest normally wins
    let merges3 = train("aaabdaaabac", 259);
    let map3: HashMap<(u32, u32), u32> = merges3.iter().copied().collect();
    assert_eq!(encode("aab", &map3), vec![256, 98]);

    assert_eq!(encode("aaab", &map3), vec![258]);
    // encoding the training text reproduces training's final ids
    assert_eq!(encode("aaabdaaabac", &map3), vec![258, 100, 258, 97, 99]);
}
