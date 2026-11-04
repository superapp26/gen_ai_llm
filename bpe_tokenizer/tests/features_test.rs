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
