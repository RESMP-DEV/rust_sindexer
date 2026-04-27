//! Import resolution for dependency graph construction.
//!
//! This module resolves import paths to actual file paths in the codebase.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::parse::ImportEdge;

/// A resolved dependency edge between two files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ResolvedEdge {
    /// The file that contains the import (absolute path).
    pub from: String,
    /// The file being imported (absolute path).
    pub to: String,
    /// The original import symbol/path.
    pub import_symbol: Option<String>,
}

/// A file index for resolving imports.
#[derive(Debug, Clone, Default)]
pub struct FileIndex {
    /// Map from relative path to absolute path.
    pub by_relative: BTreeMap<String, PathBuf>,
    /// Map from module name (filename without extension) to absolute paths.
    pub by_module: BTreeMap<String, Vec<PathBuf>>,
    /// Map from directory to files in that directory.
    pub by_directory: BTreeMap<PathBuf, Vec<PathBuf>>,
    /// All known file paths.
    pub all_files: BTreeSet<PathBuf>,
}

impl FileIndex {
    /// Build a file index from a list of files.
    pub fn from_files(files: &[PathBuf], root: &Path) -> Self {
        let mut index = Self::default();

        for file in files {
            if let Ok(relative) = file.strip_prefix(root) {
                let relative_str = relative.to_string_lossy().to_string();
                index.by_relative.insert(relative_str.clone(), file.clone());

                // Index by module name
                if let Some(stem) = Path::new(&relative_str).file_stem() {
                    let module_name = stem.to_string_lossy().to_string();
                    index
                        .by_module
                        .entry(module_name)
                        .or_default()
                        .push(file.clone());
                }

                // Index by directory
                if let Some(parent) = file.parent() {
                    index
                        .by_directory
                        .entry(parent.to_path_buf())
                        .or_default()
                        .push(file.clone());
                }

                index.all_files.insert(file.clone());
            }
        }

        index
    }

    /// Find a file by relative path.
    pub fn find_by_relative(&self, path: &str) -> Option<&PathBuf> {
        self.by_relative.get(path)
    }

    /// Find files by module name.
    pub fn find_by_module(&self, name: &str) -> &[PathBuf] {
        self.by_module.get(name).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Find files in a directory.
    pub fn find_in_directory(&self, dir: &Path) -> &[PathBuf] {
        self.by_directory.get(dir).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// Resolve a collection of import edges to file paths.
pub fn resolve_imports(
    edges: Vec<ImportEdge>,
    file_index: &FileIndex,
    root: &Path,
) -> Vec<ResolvedEdge> {
    let mut resolved = Vec::new();

    for edge in &edges {
        if let Some(target) = resolve_import_path(edge, file_index, root) {
            resolved.push(ResolvedEdge {
                from: edge.from_relative.clone(),
                to: target.to_string_lossy().to_string(),
                import_symbol: edge.import_symbol.clone(),
            });
        }
    }

    resolved
}

/// Resolve a single import path to a file path.
fn resolve_import_path(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    match edge.language.as_str() {
        "rust" => resolve_rust_import(edge, file_index, root),
        "python" => resolve_python_import(edge, file_index, root),
        "javascript" | "typescript" | "tsx" => resolve_js_ts_import(edge, file_index, root),
        "go" => resolve_go_import(edge, file_index, root),
        "java" => resolve_java_import(edge, file_index, root),
        "cpp" | "c" => resolve_cpp_import(edge, file_index, root),
        "ruby" => resolve_ruby_import(edge, file_index, root),
        "php" => resolve_php_import(edge, file_index, root),
        "swift" => resolve_swift_import(edge, file_index, root),
        "scala" => resolve_scala_import(edge, file_index, root),
        "csharp" => resolve_csharp_import(edge, file_index, root),
        _ => {
            debug!("No resolver for language: {}", edge.language);
            None
        }
    }
}

/// Resolve Rust import paths.
fn resolve_rust_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;
    let from_file = &edge.from_file;

    // Handle extern crate: crate:serde -> look for serde crate
    if import_path.starts_with("crate:") {
        let crate_name = import_path.strip_prefix("crate:")?;
        // External crates are not in our file index
        return None;
    }

    // Handle mod declarations: mod:config -> config.rs or config/mod.rs
    if import_path.starts_with("mod:") {
        let mod_name = import_path.strip_prefix("mod:")?;
        let from_dir = from_file.parent()?;

        // Try mod_name.rs in same directory
        let mod_rs = from_dir.join(format!("{}.rs", mod_name));
        if file_index.all_files.contains(&mod_rs) {
            return Some(mod_rs);
        }

        // Try mod_name/mod.rs
        let mod_mod_rs = from_dir.join(mod_name).join("mod.rs");
        if file_index.all_files.contains(&mod_mod_rs) {
            return Some(mod_mod_rs);
        }

        return None;
    }

    // Handle use statements
    let from_dir = from_file.parent().unwrap_or(root);

    // Check for relative imports
    if import_path.starts_with("super::") || import_path.starts_with("crate::") {
        // For simplicity, treat super:: and crate:: as project root
        let path_parts: Vec<&str> = import_path
            .trim_start_matches("super::")
            .trim_start_matches("crate::")
            .split("::")
            .collect();

        return resolve_rust_path(&path_parts, root, file_index);
    }

    // Check for self-relative imports
    if import_path.starts_with("self::") {
        let path_parts: Vec<&str> = import_path
            .trim_start_matches("self::")
            .split("::")
            .collect();
        return resolve_rust_path(&path_parts, from_dir, file_index);
    }

    // Absolute path imports (from crate root)
    if !import_path.contains("::") && !import_path.contains('/') {
        // Single identifier - could be a module at root
        if let Some(files) = file_index.by_module.get(import_path) {
            if let Some(first) = files.first() {
                return Some(first.clone());
            }
        }
    }

    // Try to resolve as a path
    let path_parts: Vec<&str> = import_path.split("::").collect();
    resolve_rust_path(&path_parts, root, file_index)
}

/// Resolve a Rust path to a file.
fn resolve_rust_path(
    path_parts: &[&str],
    base_dir: &Path,
    file_index: &FileIndex,
) -> Option<PathBuf> {
    if path_parts.is_empty() {
        return None;
    }

    // Build potential paths
    let mut current_dir = base_dir.to_path_buf();

    for (i, part) in path_parts.iter().enumerate() {
        let is_last = i == path_parts.len() - 1;

        // Try part.rs
        let file_rs = current_dir.join(format!("{}.rs", part));
        if file_index.all_files.contains(&file_rs) {
            if is_last {
                return Some(file_rs);
            }
            if let Some(parent) = file_rs.parent() {
                current_dir = parent.to_path_buf();
                continue;
            }
        }

        // Try part/mod.rs
        let file_mod_rs = current_dir.join(part).join("mod.rs");
        if file_index.all_files.contains(&file_mod_rs) {
            if is_last {
                return Some(file_mod_rs);
            }
            current_dir = current_dir.join(part);
            continue;
        }

        // Try part/lib.rs
        let file_lib_rs = current_dir.join(part).join("lib.rs");
        if file_index.all_files.contains(&file_lib_rs) {
            if is_last {
                return Some(file_lib_rs);
            }
            current_dir = current_dir.join(part);
            continue;
        }

        return None;
    }

    None
}

/// Resolve Python import paths.
fn resolve_python_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;
    let from_file = &edge.from_file;

    // Handle relative imports
    if import_path.starts_with('.') {
        let mut current_dir = from_file.parent()?.to_path_buf();
        let mut dots = 0;

        for ch in import_path.chars() {
            if ch == '.' {
                dots += 1;
            } else {
                break;
            }
        }

        for _ in 1..dots {
            if let Some(parent) = current_dir.parent() {
                current_dir = parent.to_path_buf();
            }
        }

        let rest = &import_path[dots..];
        return resolve_python_path(rest, &current_dir, file_index);
    }

    // Absolute import
    resolve_python_path(import_path, root, file_index)
}

/// Resolve a Python path to a file.
fn resolve_python_path(path: &str, base_dir: &Path, file_index: &FileIndex) -> Option<PathBuf> {
    let parts: Vec<&str> = path.split('.').collect();

    let mut current_dir = base_dir.to_path_buf();

    for (i, part) in parts.iter().enumerate() {
        let is_last = i == parts.len() - 1;

        // Try part/__init__.py
        let init_py = current_dir.join(part).join("__init__.py");
        if file_index.all_files.contains(&init_py) {
            if is_last {
                return Some(init_py);
            }
            current_dir = current_dir.join(part);
            continue;
        }

        // Try part.py
        let py_file = current_dir.join(format!("{}.py", part));
        if file_index.all_files.contains(&py_file) {
            if is_last {
                return Some(py_file);
            }
            if let Some(parent) = py_file.parent() {
                current_dir = parent.to_path_buf();
                continue;
            }
        }

        return None;
    }

    None
}

/// Resolve JavaScript/TypeScript import paths.
fn resolve_js_ts_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    // Skip external packages (not starting with . or ..)
    if !import_path.starts_with('.') && !import_path.starts_with("..") {
        return None;
    }

    let from_file = &edge.from_file;
    let from_dir = from_file.parent()?;

    // Build potential path
    let mut potential = from_dir.join(import_path);

    // Try various extensions
    for ext in [".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js", "/index.jsx"]
    {
        let test_path = if ext.starts_with('/') {
            potential.join(format!("index{}", ext.strip_prefix('/').unwrap()))
        } else {
            potential.with_extension(ext.trim_start_matches('.'))
        };

        if file_index.all_files.contains(&test_path) {
            return Some(test_path);
        }
    }

    // Try without changing extension
    if file_index.all_files.contains(&potential) {
        return Some(potential);
    }

    None
}

/// Resolve Go import paths.
fn resolve_go_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    // Go imports are typically full module paths like "github.com/user/repo/pkg"
    // For local resolution, we look for matching directory structures
    let import_path = &edge.import_path;

    // Try to find matching directory structure
    for file in &file_index.all_files {
        if let Some(rel) = file.strip_prefix(root).ok() {
            let rel_str = rel.to_string_lossy();
            if rel_str.contains(import_path) || rel_str.ends_with(import_path) {
                return Some(file.clone());
            }
        }
    }

    None
}

/// Resolve Java import paths.
fn resolve_java_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    // Java imports are package paths like com.example.MyClass
    // Convert to file path
    let parts: Vec<&str> = import_path.split('.').collect();

    let mut path = root.to_path_buf();
    for part in &parts[..parts.len().saturating_sub(1)] {
        path = path.join(part);
    }

    // Last part is the class name
    if let Some(class_name) = parts.last() {
        let java_file = path.join(format!("{}.java", class_name));
        if file_index.all_files.contains(&java_file) {
            return Some(java_file);
        }
    }

    None
}

/// Resolve C/C++ import paths.
fn resolve_cpp_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    // System includes (from <...>) are not resolved
    if edge.import_symbol.as_deref() == Some("system") {
        return None;
    }

    // Try to find the file
    for file in &file_index.all_files {
        if let Some(file_name) = file.file_name() {
            if file_name.to_string_lossy() == *import_path {
                return Some(file.clone());
            }
        }
        if file.to_string_lossy().contains(import_path.as_str()) {
            return Some(file.clone());
        }
    }

    None
}

/// Resolve Ruby import paths.
fn resolve_ruby_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    // Try to find matching file
    for file in &file_index.all_files {
        if let Some(rel) = file.strip_prefix(root).ok() {
            let rel_str = rel.to_string_lossy();

            // Check various patterns
            if rel_str == format!("{}.rb", import_path)
                || rel_str == format!("{}/index.rb", import_path)
                || rel_str.ends_with(&format!("/{}.rb", import_path))
            {
                return Some(file.clone());
            }
        }
    }

    None
}

/// Resolve PHP import paths.
fn resolve_php_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    // PHP namespaces map to directory structures
    let parts: Vec<&str> = import_path.split('\\').collect();

    let mut path = root.to_path_buf();
    for part in &parts[..parts.len().saturating_sub(1)] {
        path = path.join(part.to_lowercase());
    }

    if let Some(class_name) = parts.last() {
        let php_file = path.join(format!("{}.php", class_name.to_lowercase()));
        if file_index.all_files.contains(&php_file) {
            return Some(php_file);
        }
    }

    None
}

/// Resolve Swift import paths.
fn resolve_swift_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    // Swift imports are module names
    // Look for matching directories with Swift files
    for file in &file_index.all_files {
        if file.extension().map_or(false, |e| e == "swift") {
            if let Some(parent) = file.parent() {
                if let Some(dir_name) = parent.file_name() {
                    if dir_name.to_string_lossy() == *import_path {
                        return Some(file.clone());
                    }
                }
            }
        }
    }

    None
}

/// Resolve Scala import paths.
fn resolve_scala_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    // Scala imports are package paths
    let parts: Vec<&str> = import_path.split('.').collect();

    let mut path = root.to_path_buf();
    for part in &parts[..parts.len().saturating_sub(1)] {
        path = path.join(part);
    }

    if let Some(last) = parts.last() {
        // Try last.scala
        let scala_file = path.join(format!("{}.scala", last));
        if file_index.all_files.contains(&scala_file) {
            return Some(scala_file);
        }

        // Try last/package.scala
        let package_scala = path.join(last).join("package.scala");
        if file_index.all_files.contains(&package_scala) {
            return Some(package_scala);
        }
    }

    None
}

/// Resolve C# import paths.
fn resolve_csharp_import(
    edge: &ImportEdge,
    file_index: &FileIndex,
    root: &Path,
) -> Option<PathBuf> {
    let import_path = &edge.import_path;

    // C# using statements are namespace-based
    // Look for files with matching namespace declarations
    for file in &file_index.all_files {
        if file.extension().map_or(false, |e| e == "cs") {
            if let Ok(content) = std::fs::read_to_string(file) {
                if content.contains(&format!("namespace {}", import_path)) {
                    return Some(file.clone());
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_file_index_from_files() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let src = root.join("src");
        fs::create_dir_all(&src).unwrap();

        let main = src.join("main.rs");
        let lib = src.join("lib.rs");
        fs::write(&main, "fn main() {}").unwrap();
        fs::write(&lib, "pub fn lib() {}").unwrap();

        let files = vec![main.clone(), lib.clone()];
        let index = FileIndex::from_files(&files, root);

        assert!(index.all_files.contains(&main));
        assert!(index.all_files.contains(&lib));
        assert!(index.by_module.contains_key("main"));
        assert!(index.by_module.contains_key("lib"));
    }

    #[test]
    fn test_resolve_rust_mod_import() {
        let temp = TempDir::new().unwrap();
        let root = temp.path();

        let src = root.join("src");
        fs::create_dir_all(&src).unwrap();

        let main = src.join("main.rs");
        let config = src.join("config.rs");
        fs::write(&main, "mod config;").unwrap();
        fs::write(&config, "pub struct Config;").unwrap();

        let files = vec![main.clone(), config.clone()];
        let index = FileIndex::from_files(&files, root);

        let edge = ImportEdge {
            from_file: main.clone(),
            from_relative: "src/main.rs".to_string(),
            import_path: "mod:config".to_string(),
            import_symbol: None,
            language: "rust".to_string(),
        };

        let resolved = resolve_rust_import(&edge, &index, root);
        assert_eq!(resolved, Some(config));
    }

    #[test]
    fn test_resolved_edge_serialization() {
        let edge = ResolvedEdge {
            from: "src/main.rs".to_string(),
            to: "src/lib.rs".to_string(),
            import_symbol: Some("helper".to_string()),
        };

        let json = serde_json::to_string(&edge).unwrap();
        let parsed: ResolvedEdge = serde_json::from_str(&json).unwrap();

        assert_eq!(edge.from, parsed.from);
        assert_eq!(edge.to, parsed.to);
        assert_eq!(edge.import_symbol, parsed.import_symbol);
    }
}
