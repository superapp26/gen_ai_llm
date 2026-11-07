use bpe_tokenizer::features::*;
use std::collections::HashMap;
use std::time::Instant;
use std::{env, fs};

fn main() {
    let gutenberg_book_path = env::args()
        .nth(1)
        .expect("Please provide a path to the book file");

    let file_content =
        fs::read_to_string(&gutenberg_book_path).expect("Unable to read the content of the file");

    let start_time = Instant::now();

    let merges = train(&file_content, 512);

    let duration = start_time.elapsed();

    println!("The total duration of the training is {duration:.2?}");

    let vocab = vocab_builder(&merges);

    let map: HashMap<(u32, u32), u32> = merges.iter().copied().collect();

    let start_encode_time = Instant::now();

    let tokens_ids = encode(&file_content, &map);

    let encode_time_duration = start_encode_time.elapsed();

    println!("The total duration of the encoding is {encode_time_duration:.2?}");

    assert_eq!(decode(&tokens_ids, &vocab), file_content);

    let bytes_count = file_content.len();
    let token_count = tokens_ids.len();

    println!(
        "{} bytes -> {} tokens, compression ratio: {:.2}",
        bytes_count,
        token_count,
        (bytes_count as f64 / token_count as f64)
    );

    let first_merges = merges.iter().take(20);

    for &((first_token_id, second_token_id), new_token_id) in first_merges {
        let token = &vocab[new_token_id as usize];

        println!(
            "{}: {:?} + {:?} → {:?} {:?}",
            new_token_id,
            String::from_utf8_lossy(&vocab[first_token_id as usize]),
            String::from_utf8_lossy(&vocab[second_token_id as usize]),
            String::from_utf8_lossy(token),
            token,
        );
    }
}
