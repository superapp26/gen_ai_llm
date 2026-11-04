use std::cmp::Reverse;
use std::collections::HashMap;

pub fn text_to_bytes_converter(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

// This function is called to count the frequency of each of pair and return the pair with the frequency of their occurence

pub fn common_pairs_counter(ids: &[u32]) -> HashMap<(u32, u32), usize> {
    let mut pairs_count = HashMap::new();

    for w in ids.windows(2) {
        *pairs_count.entry((w[0], w[1])).or_insert(0) += 1;
    }

    pairs_count
}

pub fn most_common_pair(ids: &[u32]) -> Option<((u32, u32), usize)> {
    common_pairs_counter(ids)
        .into_iter()
        .max_by_key(|&(pair, count)| (count, Reverse(pair)))
}

pub fn merge_pair(ids: &[u32], pair: (u32, u32), new_token_id: u32) -> Vec<u32> {
    let ids_len = ids.len();

    let mut merged_ids = Vec::with_capacity(ids_len);

    let mut i = 0;

    while i < ids_len {
        if i + 1 < ids_len && (ids[i], ids[i + 1]) == pair {
            merged_ids.push(new_token_id);
            i += 2;
        } else {
            merged_ids.push(ids[i]);

            i += 1;
        }
    }

    merged_ids
}

pub fn train(text: &str, vocab_size: u32) -> Vec<((u32, u32), u32)> {
    assert!(
        vocab_size >= 256,
        "vocab_size must be at least 256: \
      ids 0-255 are the raw byte values"
    );

    let mut merges = Vec::with_capacity((vocab_size - 256) as usize);

    let mut ids = text_to_bytes_converter(text);

    for new_token_id in 256..vocab_size {
        let Some((pair, _count)) = most_common_pair(&ids) else {
            break;
        };

        ids = merge_pair(&ids, pair, new_token_id);

        merges.push((pair, new_token_id));
    }

    merges
}
