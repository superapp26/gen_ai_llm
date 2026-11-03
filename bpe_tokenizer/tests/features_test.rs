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