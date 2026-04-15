//! AST-based code chunk extraction.
//!
//! Walks tree-sitter AST to extract semantically meaningful code chunks
//! (functions, classes, methods, etc.) for embedding and indexing.

use std::path::Path;

use sha2::{Digest, Sha256};
use tree_sitter::{Node, Tree};

use crate::types::CodeChunk;

/// Extract code chunks from a parsed AST.
///
/// Recursively walks the AST looking for nodes matching `node_types` and extracts
/// their source text as chunks. Handles nested structures by avoiding duplicate
/// extraction of child nodes that are themselves splittable types.
///
/// # Arguments
///
/// * `tree` - The parsed tree-sitter AST
/// * `source` - The original source code text
/// * `node_types` - Node type names to extract (e.g., "function_definition", "class_definition")
/// * `file_path` - Absolute path to the source file
/// * `relative_path` - Path relative to repository root
/// * `language` - Programming language identifier
///
/// # Returns
///
/// A vector of `CodeChunk` instances, each representing a matched AST node.
pub fn extract_chunks(
    tree: &Tree,
    source: &str,
    node_types: &[&str],
    file_path: &Path,
    relative_path: &str,
    language: &str,
) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();
    let root = tree.root_node();
    let mut ctx = ExtractionContext {
        source,
        node_types,
        file_path,
        relative_path,
        language,
        chunks: &mut chunks,
    };

    extract_chunks_recursive(root, &mut ctx, false);

    chunks
}

struct ExtractionContext<'a> {
    source: &'a str,
    node_types: &'a [&'a str],
    file_path: &'a Path,
    relative_path: &'a str,
    language: &'a str,
    chunks: &'a mut Vec<CodeChunk>,
}

/// Recursively walk the AST and extract chunks for matching node types.
///
/// The `inside_splittable` flag prevents extracting nested splittable nodes
/// as separate top-level chunks when they're already part of a parent chunk.
/// For example, methods inside a class are extracted as part of the class,
/// not as separate chunks (unless the class is too large and needs splitting).
fn extract_chunks_recursive(node: Node, ctx: &mut ExtractionContext<'_>, inside_splittable: bool) {
    let node_type = node.kind();
    let is_splittable = ctx.node_types.contains(&node_type);

    if is_splittable && !inside_splittable {
        // Extract this node as a chunk
        if let Some(chunk) = node_to_chunk(
            node,
            ctx.source,
            ctx.file_path,
            ctx.relative_path,
            ctx.language,
        ) {
            ctx.chunks.push(chunk);
        }

        // Continue walking children but mark that we're inside a splittable node
        // This allows us to still find nested splittable types if needed
        for child in node.children(&mut node.walk()) {
            extract_chunks_recursive(child, ctx, true);
        }
    } else {
        // Not a splittable node, or already inside one - continue walking
        for child in node.children(&mut node.walk()) {
            extract_chunks_recursive(child, ctx, inside_splittable);
        }
    }
}

/// Convert a tree-sitter node to a CodeChunk.
///
/// Extracts source text and position information from the node.
/// Returns None if the node has no content or positions are invalid.
fn node_to_chunk(
    node: Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    language: &str,
) -> Option<CodeChunk> {
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();

    // Validate byte range
    if start_byte >= end_byte || end_byte > source.len() {
        return None;
    }

    let content = source.get(start_byte..end_byte)?;

    // Skip empty or whitespace-only chunks
    if content.trim().is_empty() {
        return None;
    }

    // tree-sitter uses 0-indexed lines, we want 1-indexed
    let start_line = node.start_position().row as u32 + 1;
    let end_line = node.end_position().row as u32 + 1;

    let id = generate_chunk_id(relative_path, start_line, content);

    Some(CodeChunk {
        id,
        content: content.to_string(),
        file_path: file_path.to_path_buf(),
        relative_path: relative_path.to_string(),
        start_line,
        end_line,
        language: language.to_string(),
    })
}

/// Generate a unique chunk ID using SHA-256 hash.
///
/// The ID is derived from the relative path, start line, and content,
/// ensuring stability across runs for the same chunk.
fn generate_chunk_id(relative_path: &str, start_line: u32, content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(relative_path.as_bytes());
    hasher.update(start_line.to_le_bytes());
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_rust(source: &str) -> Tree {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("Failed to set Rust language");
        parser.parse(source, None).expect("Failed to parse")
    }

    #[test]
    fn test_extract_function() {
        let source = r#"
fn hello() {
    println!("Hello, world!");
}

fn goodbye() {
    println!("Goodbye!");
}
"#;
        let tree = parse_rust(source);
        let chunks = extract_chunks(
            &tree,
            source,
            &["function_item"],
            Path::new("/test/file.rs"),
            "file.rs",
            "rust",
        );

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("hello"));
        assert!(chunks[1].content.contains("goodbye"));
        assert_eq!(chunks[0].start_line, 2);
        assert_eq!(chunks[0].end_line, 4);
    }

    #[test]
    fn test_extract_impl_block() {
        let source = r#"
struct Foo;

impl Foo {
    fn new() -> Self {
        Foo
    }

    fn method(&self) {
        // do something
    }
}
"#;
        let tree = parse_rust(source);

        // Extract only impl blocks, not individual methods
        let chunks = extract_chunks(
            &tree,
            source,
            &["impl_item"],
            Path::new("/test/file.rs"),
            "file.rs",
            "rust",
        );

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("impl Foo"));
        assert!(chunks[0].content.contains("fn new"));
        assert!(chunks[0].content.contains("fn method"));
    }

    #[test]
    fn test_chunk_id_stability() {
        let id1 = generate_chunk_id("src/main.rs", 10, "fn main() {}");
        let id2 = generate_chunk_id("src/main.rs", 10, "fn main() {}");
        let id3 = generate_chunk_id("src/main.rs", 11, "fn main() {}");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_chunk_id_format() {
        let id = generate_chunk_id("test.rs", 1, "content");
        // SHA-256 produces 64 hex characters
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_empty_source() {
        let source = "";
        let tree = parse_rust(source);
        let chunks = extract_chunks(
            &tree,
            source,
            &["function_item"],
            Path::new("/test/file.rs"),
            "file.rs",
            "rust",
        );

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_no_matching_nodes() {
        let source = "const X: i32 = 42;";
        let tree = parse_rust(source);
        let chunks = extract_chunks(
            &tree,
            source,
            &["function_item"],
            Path::new("/test/file.rs"),
            "file.rs",
            "rust",
        );

        assert!(chunks.is_empty());
    }
}
