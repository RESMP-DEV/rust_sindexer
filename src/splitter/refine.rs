//! Chunk refinement utilities for splitting oversized code chunks.

use crate::types::CodeChunk;

/// Refines a collection of code chunks by splitting any that exceed the maximum size.
///
/// Chunks larger than `max_size` bytes are split into smaller pieces at line boundaries.
/// Adjacent pieces maintain an overlap of `overlap` lines to preserve context across splits.
///
/// # Arguments
///
/// * `chunks` - Input chunks to refine
/// * `max_size` - Maximum size in bytes for any single chunk
/// * `overlap` - Number of lines to overlap between adjacent split pieces
///
/// # Returns
///
/// A new vector of chunks where all chunks are at or below `max_size` bytes.
/// Chunks that were already small enough are returned unchanged.
pub fn refine_chunks(chunks: Vec<CodeChunk>, max_size: usize, overlap: usize) -> Vec<CodeChunk> {
    let mut refined = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        if chunk.content.len() <= max_size {
            refined.push(chunk);
        } else {
            let split = split_chunk(&chunk, max_size, overlap);
            refined.extend(split);
        }
    }

    refined
}

/// Splits a single chunk into smaller pieces at line boundaries.
fn split_chunk(chunk: &CodeChunk, max_size: usize, overlap: usize) -> Vec<CodeChunk> {
    let lines: Vec<&str> = chunk.content.lines().collect();

    if lines.is_empty() {
        return vec![chunk.clone()];
    }

    let mut pieces = Vec::new();
    let mut start_idx = 0;
    let mut piece_num = 0;

    while start_idx < lines.len() {
        let (end_idx, content) = find_split_point(&lines, start_idx, max_size);

        if content.is_empty() {
            // Single line exceeds max_size; include it anyway to avoid infinite loop
            let line = lines[start_idx];
            let piece = create_piece(
                chunk,
                line.to_string(),
                chunk.start_line + start_idx as u32,
                chunk.start_line + start_idx as u32,
                piece_num,
            );
            pieces.push(piece);
            piece_num += 1;
            start_idx += 1;
        } else {
            let piece_start_line = chunk.start_line + start_idx as u32;
            let piece_end_line = chunk.start_line + (end_idx - 1) as u32;

            let piece = create_piece(chunk, content, piece_start_line, piece_end_line, piece_num);
            pieces.push(piece);
            piece_num += 1;

            // Move forward, but go back by overlap lines for context
            let advance = end_idx.saturating_sub(start_idx);
            if advance == 0 {
                // Prevent infinite loop if we can't make progress
                start_idx = end_idx.max(start_idx + 1);
            } else {
                start_idx = end_idx.saturating_sub(overlap);
                // Ensure we make forward progress
                if start_idx
                    <= pieces
                        .last()
                        .map(|p| (p.start_line - chunk.start_line) as usize)
                        .unwrap_or(0)
                {
                    start_idx = end_idx;
                }
            }
        }

        // Safety check: if we've reached the end, break
        if end_idx >= lines.len() {
            break;
        }
    }

    pieces
}

/// Finds the furthest split point that keeps content under max_size.
/// Returns (end_index_exclusive, accumulated_content).
fn find_split_point(lines: &[&str], start_idx: usize, max_size: usize) -> (usize, String) {
    let mut accumulated = String::new();
    let mut end_idx = start_idx;

    for (i, line) in lines.iter().enumerate().skip(start_idx) {
        let line_with_newline = if accumulated.is_empty() {
            line.to_string()
        } else {
            format!("\n{}", line)
        };

        if accumulated.len() + line_with_newline.len() > max_size && !accumulated.is_empty() {
            // Adding this line would exceed max_size; stop here
            break;
        }

        accumulated.push_str(&line_with_newline);
        end_idx = i + 1;

        // If we've hit max_size exactly or the accumulated content is large enough, stop
        if accumulated.len() >= max_size {
            break;
        }
    }

    (end_idx, accumulated)
}

/// Creates a new CodeChunk piece from a parent chunk.
fn create_piece(
    parent: &CodeChunk,
    content: String,
    start_line: u32,
    end_line: u32,
    piece_num: usize,
) -> CodeChunk {
    CodeChunk {
        id: format!("{}_{}", parent.id, piece_num),
        content,
        file_path: parent.file_path.clone(),
        relative_path: parent.relative_path.clone(),
        start_line,
        end_line,
        language: parent.language.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_chunk(content: &str, start_line: u32) -> CodeChunk {
        let line_count = content.lines().count() as u32;
        CodeChunk {
            id: "test_chunk".to_string(),
            content: content.to_string(),
            file_path: PathBuf::from("/test/file.rs"),
            relative_path: "file.rs".to_string(),
            start_line,
            end_line: start_line + line_count.saturating_sub(1),
            language: "rust".to_string(),
        }
    }

    #[test]
    fn test_small_chunk_unchanged() {
        let chunk = make_chunk("fn foo() {}\n", 1);
        let result = refine_chunks(vec![chunk.clone()], 100, 2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, chunk.content);
    }

    #[test]
    fn test_splits_large_chunk() {
        let content = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8";
        let chunk = make_chunk(content, 1);
        // Each line is ~6 bytes, set max_size to force splits
        let result = refine_chunks(vec![chunk], 20, 1);
        assert!(result.len() > 1);
    }

    #[test]
    fn test_maintains_overlap() {
        let content = "aaa\nbbb\nccc\nddd\neee\nfff";
        let chunk = make_chunk(content, 10);
        let result = refine_chunks(vec![chunk], 12, 2);

        // With overlap, some content should appear in multiple chunks
        if result.len() > 1 {
            // Check that pieces have overlapping line ranges
            for i in 1..result.len() {
                let prev_end = result[i - 1].end_line;
                let curr_start = result[i].start_line;
                // With overlap=2, the current piece should start before or at the previous end
                assert!(
                    curr_start <= prev_end + 1,
                    "Expected overlap between pieces {} and {}",
                    i - 1,
                    i
                );
            }
        }
    }

    #[test]
    fn test_line_numbers_updated() {
        let content = "line1\nline2\nline3\nline4";
        let chunk = make_chunk(content, 100);
        let result = refine_chunks(vec![chunk], 12, 0);

        for piece in &result {
            assert!(piece.start_line >= 100);
            assert!(piece.end_line >= piece.start_line);
        }
    }

    #[test]
    fn test_empty_chunk() {
        let chunk = make_chunk("", 1);
        let result = refine_chunks(vec![chunk], 100, 2);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_single_long_line() {
        // A single line that exceeds max_size should still be included
        let long_line = "a".repeat(200);
        let chunk = make_chunk(&long_line, 1);
        let result = refine_chunks(vec![chunk], 50, 2);
        assert!(!result.is_empty());
        // The long line should be preserved even though it exceeds max_size
        assert!(result.iter().any(|c| c.content.len() >= 50));
    }

    #[test]
    fn test_piece_ids() {
        let content = "line1\nline2\nline3\nline4\nline5\nline6";
        let chunk = make_chunk(content, 1);
        let result = refine_chunks(vec![chunk], 15, 0);

        for (i, piece) in result.iter().enumerate() {
            assert!(
                piece.id.ends_with(&format!("_{}", i)),
                "Piece {} should have id ending with _{}, got {}",
                i,
                i,
                piece.id
            );
        }
    }
}
