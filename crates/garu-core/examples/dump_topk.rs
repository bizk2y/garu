//! Dump the exact top-K reranker input candidates per sentence as JSON lines.
//!
//! Each output line is one sentence:
//!   {"text":"...","candidates":[{"score":<f32>,"tokens":[["form","POS",start,end],...]}, ...]}
//!
//! Context POS corrections and display normalization are excluded from training.
use garu_core::model::Analyzer;
use garu_core::types::AnalyzeResult;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

fn write_candidates(out: &mut impl Write, text: &str, candidates: &[AnalyzeResult]) {
    let candidates: Vec<_> = candidates.iter().map(|candidate| {
        let tokens: Vec<_> = candidate.tokens.iter()
            .map(|token| (token.text.as_str(), token.pos.as_str(), token.start, token.end))
            .collect();
        serde_json::json!({"score": candidate.score, "tokens": tokens})
    }).collect();
    serde_json::to_writer(&mut *out, &serde_json::json!({"text": text, "candidates": candidates}))
        .expect("failed to write candidates");
    writeln!(out).expect("failed to write newline");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dump_topk <input.txt> [top_k=5]");
        std::process::exit(2);
    }
    let input_path = &args[1];
    let top_k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let model_path = env::var("GARU_MODEL").unwrap_or_else(|_| "models/codebook.gmdl".to_string());
    let model_data = fs::read(&model_path).expect("failed to read model");
    let analyzer = Analyzer::from_bytes(&model_data).expect("failed to load");

    let file = fs::File::open(input_path).expect("failed to open input");
    let reader = io::BufReader::new(file);
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in reader.lines() {
        let line = line.expect("read line").trim().to_string();
        if line.is_empty() {
            write_candidates(&mut out, "", &[]);
            continue;
        }
        let candidates = analyzer.analyze_topn_for_rerank(&line, top_k);
        write_candidates(&mut out, &line, &candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use garu_core::types::{Pos, Token};

    #[test]
    fn controls_and_score_roundtrip() {
        let text = "한글\\\"\n\r\t\u{0000}\u{0008}\u{000c}\u{001f}";
        let score = -1.2345678f32;
        let candidates = vec![AnalyzeResult {
            tokens: vec![Token { text: text.to_string(), pos: Pos::SW,
                                start: 3, end: 10, score: None }],
            score, elapsed_ms: 0.0,
        }];
        let mut output = Vec::new();
        write_candidates(&mut output, text, &candidates);
        let decoded = serde_json::from_slice::<serde_json::Value>(&output);
        assert!(decoded.is_ok(), "invalid JSON: {output:?}");
        let decoded = decoded.unwrap();
        assert_eq!(decoded["text"], text);
        assert_eq!(decoded["candidates"][0]["tokens"][0], serde_json::json!([text, "SW", 3, 10]));
        let restored = decoded["candidates"][0]["score"].as_f64().unwrap() as f32;
        assert_eq!(restored.to_bits(), score.to_bits());
        assert_eq!(output.iter().filter(|&&byte| byte == b'\n').count(), 1);
    }
}
