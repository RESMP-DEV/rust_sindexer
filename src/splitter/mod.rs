pub mod extractor;
pub mod node_types;
pub mod parsers;
pub mod refine;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use tracing::{debug, warn};

use crate::config::SUPPORTED_EXTENSIONS;
use crate::types::CodeChunk;

pub use extractor::extract_chunks;
pub use node_types::{get_splittable_nodes, is_splittable, supported_languages, SPLITTABLE_NODES};
pub use parsers::{extension_to_language, get_parser, get_parser_for_extension, LanguageParser};
pub use refine::refine_chunks;

/// Configuration for the code splitter.
#[derive(Clone, Debug)]
pub struct Config {
    /// Minimum chunk size in lines.
    pub min_chunk_lines: u32,
    /// Maximum chunk size in bytes for refinement.
    pub max_chunk_bytes: usize,
    /// Target chunk size in lines (for splitting large constructs).
    pub target_chunk_lines: u32,
    /// Number of lines to overlap between split chunks.
    pub overlap_lines: usize,
    /// Root path for computing relative paths.
    pub root_path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            min_chunk_lines: 5,
            max_chunk_bytes: 4096,
            target_chunk_lines: 50,
            overlap_lines: 3,
            root_path: PathBuf::from("."),
        }
    }
}

/// Splits source code files into semantic chunks using tree-sitter parsing.
pub struct CodeSplitter {
    config: Config,
}

impl CodeSplitter {
    /// Create a new code splitter with the given configuration.
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Split a single file into code chunks.
    pub fn split_file(&self, path: &Path) -> Result<Vec<CodeChunk>> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let source = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        if source.is_empty() {
            return Ok(vec![]);
        }

        let relative_path = path
            .strip_prefix(&self.config.root_path)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let language = extension_to_language(&extension);
        let raw_chunks = match language {
            Some(language) => match get_parser(language) {
                Some(mut parser) => match parser.parse(&source, None) {
                    Some(tree) => {
                        let node_types = get_splittable_nodes(language).unwrap_or(&[]);
                        extract_chunks(&tree, &source, node_types, path, &relative_path, language)
                    }
                    None => {
                        debug!("Failed to parse AST file {}, using fallback splitter", path.display());
                        self.fallback_chunks(
                            path,
                            &relative_path,
                            &source,
                            &extension,
                            Some(language),
                        )
                    }
                },
                None => self.fallback_chunks(path, &relative_path, &source, &extension, Some(language)),
            },
            None if is_supported_non_ast_extension(&extension) => {
                self.fallback_chunks(path, &relative_path, &source, &extension, None)
            }
            None => {
                debug!("Unsupported file extension: {:?}", path);
                return Ok(vec![]);
            }
        };

        let refined_chunks = refine_chunks(
            raw_chunks,
            self.config.max_chunk_bytes,
            self.config.overlap_lines,
        );

        Ok(refined_chunks)
    }

    /// Split multiple files in parallel using rayon.
    pub fn split_files(&self, paths: &[PathBuf]) -> Result<Vec<CodeChunk>> {
        let results: Vec<Result<Vec<CodeChunk>>> =
            paths.par_iter().map(|path| self.split_file(path)).collect();

        let mut all_chunks = Vec::new();
        for result in results {
            match result {
                Ok(chunks) => all_chunks.extend(chunks),
                Err(e) => {
                    warn!("Failed to split file: {}", e);
                }
            }
        }

        Ok(all_chunks)
    }

    /// Get the splitter configuration used for chunk generation.
    pub fn config(&self) -> &Config {
        &self.config
    }

    fn fallback_chunks(
        &self,
        path: &Path,
        relative_path: &str,
        source: &str,
        extension: &str,
        language: Option<&str>,
    ) -> Vec<CodeChunk> {
        let chunk_language = fallback_language(extension, language);

        if matches!(extension, "md" | "rst") {
            let sections = split_markdown_sections(source);
            if !sections.is_empty() {
                return split_markdown_chunks(
                    &sections,
                    path,
                    relative_path,
                    &chunk_language,
                    self.fallback_chunk_lines(),
                    self.config.overlap_lines,
                );
            }
        }

        split_by_lines(
            source,
            path,
            relative_path,
            &chunk_language,
            self.fallback_chunk_lines(),
            self.config.overlap_lines,
        )
    }

    fn fallback_chunk_lines(&self) -> usize {
        (self.config.max_chunk_bytes / 80).max(1)
    }
}

fn is_supported_non_ast_extension(extension: &str) -> bool {
    SUPPORTED_EXTENSIONS.contains(&extension) && extension_to_language(extension).is_none()
}

fn fallback_language(extension: &str, language: Option<&str>) -> String {
    match language {
        Some(language) => language.to_string(),
        None => match extension {
            "md" => "markdown".to_string(),
            "rst" => "markdown".to_string(),
            "yml" => "yaml".to_string(),
            other => other.to_string(),
        },
    }
}

fn split_markdown_sections(source: &str) -> Vec<(u32, u32, String)> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut boundaries = vec![0usize];
    for (idx, line) in lines.iter().enumerate().skip(1) {
        if is_markdown_heading(line) {
            boundaries.push(idx);
        }
    }
    boundaries.push(lines.len());

    let mut sections = Vec::new();
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        if start >= end {
            continue;
        }

        let content = lines[start..end].join("\n");
        if content.trim().is_empty() {
            continue;
        }

        sections.push((start as u32 + 1, end as u32, content));
    }

    sections
}

fn split_markdown_chunks(
    sections: &[(u32, u32, String)],
    path: &Path,
    relative_path: &str,
    language: &str,
    lines_per_chunk: usize,
    overlap_lines: usize,
) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();

    for (section_idx, (start_line, end_line, content)) in sections.iter().enumerate() {
        let section_lines = content.lines().count();
        if section_lines <= lines_per_chunk.max(1) {
            chunks.push(CodeChunk {
                id: format!("{relative_path}:md:{section_idx}"),
                content: content.clone(),
                file_path: path.to_path_buf(),
                relative_path: relative_path.to_string(),
                start_line: *start_line,
                end_line: *end_line,
                language: language.to_string(),
            });
            continue;
        }

        let section_chunks = split_by_lines(
            content,
            path,
            relative_path,
            language,
            lines_per_chunk,
            overlap_lines,
        );

        chunks.extend(section_chunks.into_iter().enumerate().map(|(sub_idx, mut chunk)| {
            let chunk_line_count = chunk.end_line - chunk.start_line;
            chunk.id = format!("{relative_path}:md:{section_idx}:{sub_idx}");
            chunk.start_line = start_line + chunk.start_line - 1;
            chunk.end_line = (chunk.start_line + chunk_line_count).min(*end_line);
            chunk
        }));
    }

    chunks
}

fn is_markdown_heading(line: &str) -> bool {
    let trimmed = line.trim_start_matches('#');
    trimmed.len() < line.len() && trimmed.starts_with(' ')
}

fn split_by_lines(
    source: &str,
    path: &Path,
    relative_path: &str,
    language: &str,
    lines_per_chunk: usize,
    overlap_lines: usize,
) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let lines_per_chunk = lines_per_chunk.max(1);
    let step = lines_per_chunk.saturating_sub(overlap_lines).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut chunk_idx = 0usize;

    while start < lines.len() {
        let end = (start + lines_per_chunk).min(lines.len());
        let content = lines[start..end].join("\n");

        if !content.trim().is_empty() {
            chunks.push(CodeChunk {
                id: format!("{relative_path}:lines:{chunk_idx}"),
                content,
                file_path: path.to_path_buf(),
                relative_path: relative_path.to_string(),
                start_line: start as u32 + 1,
                end_line: end as u32,
                language: language.to_string(),
            });
            chunk_idx += 1;
        }

        if end >= lines.len() {
            break;
        }

        start += step;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.min_chunk_lines, 5);
        assert_eq!(config.max_chunk_bytes, 4096);
        assert_eq!(config.target_chunk_lines, 50);
        assert_eq!(config.overlap_lines, 3);
    }

    #[test]
    fn test_code_splitter_new() {
        let config = Config::default();
        let _splitter = CodeSplitter::new(config);
    }

    #[test]
    fn test_split_rust_file() {
        let source = r#"
fn hello() {
    println!("Hello, world!");
}

fn goodbye() {
    println!("Goodbye!");
}
"#;
        let mut temp = NamedTempFile::with_suffix(".rs").unwrap();
        temp.write_all(source.as_bytes()).unwrap();

        let config = Config::default();
        let splitter = CodeSplitter::new(config);
        let chunks = splitter.split_file(temp.path()).unwrap();

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("hello"));
        assert!(chunks[1].content.contains("goodbye"));
    }

    #[test]
    fn test_split_unsupported_extension() {
        let mut temp = NamedTempFile::with_suffix(".xyz").unwrap();
        temp.write_all(b"some content").unwrap();

        let config = Config::default();
        let splitter = CodeSplitter::new(config);
        let chunks = splitter.split_file(temp.path()).unwrap();

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_empty_file() {
        let temp = NamedTempFile::with_suffix(".rs").unwrap();

        let config = Config::default();
        let splitter = CodeSplitter::new(config);
        let chunks = splitter.split_file(temp.path()).unwrap();

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_markdown_by_headings() {
        let source = "# Title\nintro\n## Section A\na1\na2\n## Section B\nb1\n";
        let mut temp = NamedTempFile::with_suffix(".md").unwrap();
        temp.write_all(source.as_bytes()).unwrap();

        let splitter = CodeSplitter::new(Config::default());
        let chunks = splitter.split_file(temp.path()).unwrap();

        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].content.starts_with("# Title"));
        assert!(chunks[1].content.starts_with("## Section A"));
        assert!(chunks[2].content.starts_with("## Section B"));
    }

    #[test]
    fn test_split_yaml_by_lines() {
        let source = "root:\n  child1: a\n  child2: b\n  child3: c\n  child4: d\n";
        let mut temp = NamedTempFile::with_suffix(".yaml").unwrap();
        temp.write_all(source.as_bytes()).unwrap();

        let config = Config {
            min_chunk_lines: 1,
            max_chunk_bytes: 160,
            target_chunk_lines: 2,
            overlap_lines: 1,
            root_path: PathBuf::from("."),
        };
        let splitter = CodeSplitter::new(config);
        let chunks = splitter.split_file(temp.path()).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].language, "yaml");
        assert!(chunks.iter().all(|chunk| !chunk.content.trim().is_empty()));
    }

    #[test]
    fn test_ast_fallback_uses_line_chunks() {
        let source = "fn one() {}\nfn two() {}\nfn three() {}\nfn four() {}\n";
        let splitter = CodeSplitter::new(Config {
            min_chunk_lines: 1,
            max_chunk_bytes: 160,
            target_chunk_lines: 2,
            overlap_lines: 1,
            root_path: PathBuf::from("."),
        });

        let chunks =
            splitter.fallback_chunks(Path::new("test.rs"), "test.rs", source, "rs", Some("rust"));

        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].language, "rust");
        assert_eq!(chunks[0].start_line, 1);
    }
}
