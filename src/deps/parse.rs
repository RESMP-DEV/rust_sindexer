//! Import statement parsing for dependency graph construction.
//!
//! This module parses import statements from source files using tree-sitter
//! and extracts dependency relationships.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tree_sitter::{Parser, Tree};
use tracing::debug;

use crate::splitter::parsers::{extension_to_language, get_parser};

/// An import edge representing a dependency from one file to another.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportEdge {
    /// The file that contains the import statement.
    pub from_file: PathBuf,
    /// The relative path of the importing file.
    pub from_relative: String,
    /// The import path/specifier (may not be resolved yet).
    pub import_path: String,
    /// The optional symbol being imported.
    pub import_symbol: Option<String>,
    /// The language of the source file.
    pub language: String,
}

/// Parse imports from a single file.
pub fn parse_imports_from_file(
    file_path: &Path,
    relative_path: &str,
    language: &str,
) -> Result<Vec<ImportEdge>> {
    let source = std::fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

    if source.is_empty() {
        return Ok(vec![]);
    }

    let mut parser = get_parser(language)
        .with_context(|| format!("No parser available for language: {}", language))?;

    let tree = parser
        .parse(&source, None)
        .with_context(|| format!("Failed to parse file: {}", file_path.display()))?;

    Ok(parse_imports_from_tree(&tree, &source, file_path, relative_path, language))
}

/// Parse import statements from a tree-sitter syntax tree.
pub fn parse_imports_from_tree(
    tree: &Tree,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    language: &str,
) -> Vec<ImportEdge> {
    let root = tree.root_node();
    let mut edges = Vec::new();

    match language {
        "rust" => extract_rust_imports(&root, source, file_path, relative_path, &mut edges),
        "python" => extract_python_imports(&root, source, file_path, relative_path, &mut edges),
        "javascript" | "typescript" | "tsx" => {
            extract_js_ts_imports(&root, source, file_path, relative_path, language, &mut edges)
        }
        "go" => extract_go_imports(&root, source, file_path, relative_path, &mut edges),
        "java" => extract_java_imports(&root, source, file_path, relative_path, &mut edges),
        "cpp" | "c" => extract_cpp_imports(&root, source, file_path, relative_path, &mut edges),
        "ruby" => extract_ruby_imports(&root, source, file_path, relative_path, &mut edges),
        "php" => extract_php_imports(&root, source, file_path, relative_path, &mut edges),
        "swift" => extract_swift_imports(&root, source, file_path, relative_path, &mut edges),
        "scala" => extract_scala_imports(&root, source, file_path, relative_path, &mut edges),
        "csharp" => extract_csharp_imports(&root, source, file_path, relative_path, &mut edges),
        _ => {
            debug!("No import parser for language: {}", language);
        }
    }

    edges
}

/// Extract import edges from Rust source.
fn extract_rust_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "use_declaration" {
            if let Some(use_tree) = node.child_by_field_name("argument") {
                let import_path = node_text(&use_tree, source);
                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: extract_rust_use_symbol(&use_tree, source),
                    language: "rust".to_string(),
                });
            }
        } else if node.kind() == "extern_crate_declaration" {
            if let Some(crate_name) = node.child_by_field_name("name") {
                let import_path = node_text(&crate_name, source);
                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: format!("crate:{}", import_path),
                    import_symbol: None,
                    language: "rust".to_string(),
                });
            }
        } else if node.kind() == "mod_item" {
            if let Some(name) = node.child_by_field_name("name") {
                let mod_name = node_text(&name, source);
                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: format!("mod:{}", mod_name),
                    import_symbol: None,
                    language: "rust".to_string(),
                });
            }
        }
    }
}

/// Extract the main symbol from a Rust use tree.
fn extract_rust_use_symbol(node: &tree_sitter::Node, source: &str) -> Option<String> {
    if let Some(ident) = node.child_by_field_name("name") {
        return Some(node_text(&ident, source).to_string());
    }
    // For `use foo::bar`, return "bar"
    if node.kind() == "scoped_identifier" {
        if let Some(name) = node.child_by_field_name("name") {
            return Some(node_text(&name, source).to_string());
        }
    }
    // For `use foo::{a, b}`, return the first identifier
    if node.kind() == "use_wildcard" {
        return Some("*".to_string());
    }
    None
}

/// Extract import edges from Python source.
fn extract_python_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "import_statement" {
            // Handle: import foo, import foo.bar
            if let Some(name) = node.child_by_field_name("name") {
                let import_path = node_text(&name, source);
                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: None,
                    language: "python".to_string(),
                });
            }
        } else if node.kind() == "import_from_statement" {
            // Handle: from foo import bar
            let module_name = node
                .child_by_field_name("module_name")
                .map(|n| node_text(&n, source))
                .unwrap_or("");

            if let Some(names) = node.child_by_field_name("names") {
                let mut names_cursor = names.walk();
                for name_node in pre_order_iter(&names, &mut names_cursor) {
                    if name_node.kind() == "dotted_name" || name_node.kind() == "identifier" {
                        let symbol = node_text(&name_node, source);
                        edges.push(ImportEdge {
                            from_file: file_path.to_path_buf(),
                            from_relative: relative_path.to_string(),
                            import_path: module_name.to_string(),
                            import_symbol: Some(symbol.to_string()),
                            language: "python".to_string(),
                        });
                    }
                }
            }
        }
    }
}

/// Extract import edges from JavaScript/TypeScript source.
fn extract_js_ts_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    language: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "import_statement" {
            // Handle: import foo from 'bar'
            if let Some(source_str) = node.child_by_field_name("source") {
                let import_path = node_text(&source_str, source);
                // Remove quotes
                let import_path = import_path.trim_matches('\'').trim_matches('"');

                let symbol = node
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string());

                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: symbol,
                    language: language.to_string(),
                });
            }
        } else if node.kind() == "require_call" {
            // Handle: const foo = require('bar')
            if let Some(arg) = node.child_by_field_name("arguments") {
                if arg.child_count() > 0 {
                    if let Some(first_arg) = arg.child(0) {
                        if first_arg.kind() == "string" {
                            let import_path = node_text(&first_arg, source);
                            let import_path =
                                import_path.trim_matches('\'').trim_matches('"');
                            edges.push(ImportEdge {
                                from_file: file_path.to_path_buf(),
                                from_relative: relative_path.to_string(),
                                import_path: import_path.to_string(),
                                import_symbol: None,
                                language: language.to_string(),
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Extract import edges from Go source.
fn extract_go_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "import_declaration" {
            if let Some(spec) = node.child_by_field_name("import_spec") {
                if let Some(path) = spec.child_by_field_name("path") {
                    let import_path = node_text(&path, source);
                    let import_path = import_path.trim_matches('"');

                    let symbol = spec
                        .child_by_field_name("name")
                        .map(|n| node_text(&n, source).to_string());

                    edges.push(ImportEdge {
                        from_file: file_path.to_path_buf(),
                        from_relative: relative_path.to_string(),
                        import_path: import_path.to_string(),
                        import_symbol: symbol,
                        language: "go".to_string(),
                    });
                }
            }
        } else if node.kind() == "import_spec_list" {
            // Handle grouped imports: import ( ... )
            let mut spec_cursor = node.walk();
            for child in pre_order_iter(node, &mut spec_cursor) {
                if child.kind() == "import_spec" {
                    if let Some(path) = child.child_by_field_name("path") {
                        let import_path = node_text(&path, source);
                        let import_path = import_path.trim_matches('"');

                        let symbol = child
                            .child_by_field_name("name")
                            .map(|n| node_text(&n, source).to_string());

                        edges.push(ImportEdge {
                            from_file: file_path.to_path_buf(),
                            from_relative: relative_path.to_string(),
                            import_path: import_path.to_string(),
                            import_symbol: symbol,
                            language: "go".to_string(),
                        });
                    }
                }
            }
        }
    }
}

/// Extract import edges from Java source.
fn extract_java_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "import_declaration" {
            if let Some(path) = node.child_by_field_name("value") {
                let import_path = node_text(&path, source);
                let is_static = node.child_by_field_name("static").is_some();

                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: if is_static {
                        import_path.split('.').last().map(|s| s.to_string())
                    } else {
                        None
                    },
                    language: "java".to_string(),
                });
            }
        }
    }
}

/// Extract import edges from C/C++ source.
fn extract_cpp_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "preproc_include" {
            if let Some(path) = node.child_by_field_name("path") {
                let import_path = node_text(&path, source);
                let is_system = import_path.starts_with('<');

                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path
                        .trim_start_matches('<')
                        .trim_end_matches('>')
                        .trim_start_matches('"')
                        .trim_end_matches('"')
                        .to_string(),
                    import_symbol: Some(if is_system {
                        "system".to_string()
                    } else {
                        "local".to_string()
                    }),
                    language: if node.kind() == "preproc_include" {
                        "cpp"
                    } else {
                        "c"
                    }
                    .to_string(),
                });
            }
        }
    }
}

/// Extract import edges from Ruby source.
fn extract_ruby_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "require_relative" || node.kind() == "require" {
            if let Some(arg) = node.child_by_field_name("name") {
                let import_path = node_text(&arg, source);
                let import_path = import_path
                    .trim_start_matches('\'')
                    .trim_start_matches('"')
                    .trim_end_matches('\'')
                    .trim_end_matches('"');

                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: None,
                    language: "ruby".to_string(),
                });
            }
        } else if node.kind() == "include" || node.kind() == "extend" || node.kind() == "prepend" {
            if let Some(arg) = node.child_by_field_name("name") {
                let import_path = node_text(&arg, source);
                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: Some(node.kind().to_string()),
                    language: "ruby".to_string(),
                });
            }
        }
    }
}

/// Extract import edges from PHP source.
fn extract_php_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "use_declaration" {
            if let Some(name) = node.child_by_field_name("name") {
                let import_path = node_text(&name, source);

                let alias = node
                    .child_by_field_name("alias")
                    .map(|n| node_text(&n, source).to_string());

                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: alias,
                    language: "php".to_string(),
                });
            }
        }
    }
}

/// Extract import edges from Swift source.
fn extract_swift_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "import_declaration" {
            if let Some(path) = node.child_by_field_name("name") {
                let import_path = node_text(&path, source);
                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: None,
                    language: "swift".to_string(),
                });
            }
        }
    }
}

/// Extract import edges from Scala source.
fn extract_scala_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "import_declaration" {
            if let Some(path) = node.child_by_field_name("path") {
                let import_path = node_text(&path, source);

                let selector = node
                    .child_by_field_name("selector")
                    .map(|n| node_text(&n, source).to_string());

                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: selector,
                    language: "scala".to_string(),
                });
            }
        }
    }
}

/// Extract import edges from C# source.
fn extract_csharp_imports(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &Path,
    relative_path: &str,
    edges: &mut Vec<ImportEdge>,
) {
    let mut cursor = root.walk();

    for node in pre_order_iter(root, &mut cursor) {
        if node.kind() == "using_directive" {
            if let Some(name) = node.child_by_field_name("name") {
                let import_path = node_text(&name, source);
                edges.push(ImportEdge {
                    from_file: file_path.to_path_buf(),
                    from_relative: relative_path.to_string(),
                    import_path: import_path.to_string(),
                    import_symbol: None,
                    language: "csharp".to_string(),
                });
            }
        }
    }
}

/// Get text from a tree-sitter node.
fn node_text<'a>(node: &tree_sitter::Node, source: &'a str) -> &'a str {
    let range = node.byte_range();
    &source[range.start..range.end]
}

/// Pre-order iterator for tree-sitter nodes.
fn pre_order_iter<'a>(
    root: &'a tree_sitter::Node,
    cursor: &'a mut tree_sitter::TreeCursor,
) -> impl Iterator<Item = tree_sitter::Node<'a>> {
    struct PreOrderIter<'a> {
        cursor: &'a mut tree_sitter::TreeCursor,
        started: bool,
        done: bool,
    }

    impl<'a> Iterator for PreOrderIter<'a> {
        type Item = tree_sitter::Node<'a>;

        fn next(&mut self) -> Option<Self::Item> {
            if !self.started {
                self.started = true;
                return Some(self.cursor.node());
            }

            if self.done {
                return None;
            }

            // Try to go to first child
            if self.cursor.goto_first_child() {
                return Some(self.cursor.node());
            }

            // Try to go to next sibling
            if self.cursor.goto_next_sibling() {
                return Some(self.cursor.node());
            }

            // Go back up and try siblings
            while self.cursor.goto_parent() {
                if self.cursor.goto_next_sibling() {
                    return Some(self.cursor.node());
                }
            }

            self.done = true;
            None
        }
    }

    PreOrderIter {
        cursor,
        started: false,
        done: false,
    }
}

/// Parse imports from multiple files in parallel.
pub fn parse_imports(
    files: &[(PathBuf, String, String)], // (file_path, relative_path, language)
) -> Vec<ImportEdge> {
    let results: Vec<Vec<ImportEdge>> = files
        .par_iter()
        .map(|(file_path, relative_path, language)| {
            parse_imports_from_file(file_path, relative_path, language).unwrap_or_default()
        })
        .collect();

    results.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_rust_imports() {
        let source = r#"
use std::collections::HashMap;
use crate::utils::helper;
use super::types::*;
extern crate serde;
mod config;
"#;
        let mut temp = NamedTempFile::with_suffix(".rs").unwrap();
        temp.write_all(source.as_bytes()).unwrap();

        let edges = parse_imports_from_file(temp.path(), "test.rs", "rust").unwrap();

        assert!(edges.len() >= 3);
        assert!(edges.iter().any(|e| e.import_path.contains("HashMap")));
        assert!(edges.iter().any(|e| e.import_path.contains("helper")));
    }

    #[test]
    fn test_parse_python_imports() {
        let source = r#"
import os
import sys.path
from collections import defaultdict, OrderedDict
from .utils import helper
"#;
        let mut temp = NamedTempFile::with_suffix(".py").unwrap();
        temp.write_all(source.as_bytes()).unwrap();

        let edges = parse_imports_from_file(temp.path(), "test.py", "python").unwrap();

        assert!(edges.len() >= 4);
        assert!(edges.iter().any(|e| e.import_path == "os"));
        assert!(edges.iter().any(|e| e.import_path == "collections"));
    }

    #[test]
    fn test_parse_js_imports() {
        let source = r#"
import React from 'react';
import { useState, useEffect } from './hooks';
const lodash = require('lodash');
"#;
        let mut temp = NamedTempFile::with_suffix(".js").unwrap();
        temp.write_all(source.as_bytes()).unwrap();

        let edges = parse_imports_from_file(temp.path(), "test.js", "javascript").unwrap();

        assert!(edges.len() >= 3);
        assert!(edges.iter().any(|e| e.import_path == "react"));
        assert!(edges.iter().any(|e| e.import_path.contains("hooks")));
    }

    #[test]
    fn test_import_edge_serialization() {
        let edge = ImportEdge {
            from_file: PathBuf::from("src/main.rs"),
            from_relative: "src/main.rs".to_string(),
            import_path: "crate::lib".to_string(),
            import_symbol: Some("helper".to_string()),
            language: "rust".to_string(),
        };

        let json = serde_json::to_string(&edge).unwrap();
        let parsed: ImportEdge = serde_json::from_str(&json).unwrap();

        assert_eq!(edge.from_file, parsed.from_file);
        assert_eq!(edge.import_path, parsed.import_path);
        assert_eq!(edge.import_symbol, parsed.import_symbol);
    }
}
