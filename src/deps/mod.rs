//! Dependency graph construction and resolution.
//!
//! This module provides functionality for parsing import statements from source files
//! and building a dependency graph across the codebase.

pub mod parse;
pub mod resolve;

pub use parse::{parse_imports, ImportEdge};
pub use resolve::{resolve_imports, ResolvedEdge};

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// A dependency graph representing import relationships between files.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DepGraph {
    /// All edges in the graph: (from_file, to_file) pairs.
    pub edges: Vec<ResolvedEdge>,
    /// Map from file path to set of files it depends on (outgoing edges).
    pub outgoing: BTreeMap<String, BTreeSet<String>>,
    /// Map from file path to set of files that depend on it (incoming edges).
    pub incoming: BTreeMap<String, BTreeSet<String>>,
    /// All known file paths in the graph.
    pub files: BTreeSet<String>,
}

impl DepGraph {
    /// Create a new empty dependency graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a dependency graph from a set of resolved edges.
    pub fn from_edges(edges: Vec<ResolvedEdge>) -> Self {
        let mut graph = Self::new();
        for edge in &edges {
            graph.files.insert(edge.from.clone());
            graph.files.insert(edge.to.clone());
            graph
                .outgoing
                .entry(edge.from.clone())
                .or_default()
                .insert(edge.to.clone());
            graph
                .incoming
                .entry(edge.to.clone())
                .or_default()
                .insert(edge.from.clone());
        }
        graph.edges = edges;
        graph
    }

    /// Add a single edge to the graph.
    pub fn add_edge(&mut self, edge: ResolvedEdge) {
        self.files.insert(edge.from.clone());
        self.files.insert(edge.to.clone());
        self.outgoing
            .entry(edge.from.clone())
            .or_default()
            .insert(edge.to.clone());
        self.incoming
            .entry(edge.to.clone())
            .or_default()
            .insert(edge.from.clone());
        self.edges.push(edge);
    }

    /// Remove all edges originating from the given file path.
    pub fn remove_edges_from(&mut self, file_path: &str) {
        if let Some(targets) = self.outgoing.remove(file_path) {
            for target in &targets {
                if let Some(sources) = self.incoming.get_mut(target) {
                    sources.remove(file_path);
                }
            }
            self.edges.retain(|e| e.from != file_path);
        }
        self.files.remove(file_path);
    }

    /// Remove all edges involving the given file path (both incoming and outgoing).
    pub fn remove_file(&mut self, file_path: &str) {
        // Remove outgoing edges
        if let Some(targets) = self.outgoing.remove(file_path) {
            for target in &targets {
                if let Some(sources) = self.incoming.get_mut(target) {
                    sources.remove(file_path);
                }
            }
        }

        // Remove incoming edges
        if let Some(sources) = self.incoming.remove(file_path) {
            for source in &sources {
                if let Some(targets) = self.outgoing.get_mut(source) {
                    targets.remove(file_path);
                }
            }
        }

        self.edges.retain(|e| e.from != file_path && e.to != file_path);
        self.files.remove(file_path);
    }

    /// Get all files that the given file depends on.
    pub fn dependencies_of(&self, file_path: &str) -> BTreeSet<String> {
        self.outgoing
            .get(file_path)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all files that depend on the given file.
    pub fn dependents_of(&self, file_path: &str) -> BTreeSet<String> {
        self.incoming
            .get(file_path)
            .cloned()
            .unwrap_or_default()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of files in the graph.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Clear all data from the graph.
    pub fn clear(&mut self) {
        self.edges.clear();
        self.outgoing.clear();
        self.incoming.clear();
        self.files.clear();
    }
}

/// Thread-safe wrapper for dependency graph operations.
#[derive(Debug, Clone)]
pub struct DepGraphHandle {
    inner: Arc<RwLock<DepGraph>>,
    root_path: PathBuf,
}

impl DepGraphHandle {
    /// Create a new dependency graph handle.
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            inner: Arc::new(RwLock::new(DepGraph::new())),
            root_path,
        }
    }

    /// Get the graph path for persistence.
    pub fn graph_path(&self) -> PathBuf {
        self.root_path.join(".sindexer").join("dep-graph.json")
    }

    /// Load the graph from disk if it exists.
    pub fn load(&self) -> anyhow::Result<bool> {
        let path = self.graph_path();
        if !path.exists() {
            return Ok(false);
        }

        let contents = std::fs::read_to_string(&path)?;
        let graph: DepGraph = serde_json::from_str(&contents)?;
        *self.inner.write() = graph;
        Ok(true)
    }

    /// Persist the graph to disk.
    pub fn save(&self) -> anyhow::Result<()> {
        let path = self.graph_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let graph = self.inner.read();
        let json = serde_json::to_string_pretty(&*graph)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Get a reference to the inner graph.
    pub fn read(&self) -> parking_lot::RwLockReadGuard<DepGraph> {
        self.inner.read()
    }

    /// Get a mutable reference to the inner graph.
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<DepGraph> {
        self.inner.write()
    }

    /// Remove edges for the given files and return the set of affected file paths.
    pub fn remove_files(&self, file_paths: &[String]) {
        let mut graph = self.inner.write();
        for file_path in file_paths {
            graph.remove_file(file_path);
        }
    }

    /// Add edges to the graph.
    pub fn add_edges(&self, edges: Vec<ResolvedEdge>) {
        let mut graph = self.inner.write();
        for edge in edges {
            graph.add_edge(edge);
        }
    }

    /// Rebuild the graph from a complete set of edges.
    pub fn rebuild(&self, edges: Vec<ResolvedEdge>) {
        let mut graph = self.inner.write();
        *graph = DepGraph::from_edges(edges);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dep_graph_from_edges() {
        let edges = vec![
            ResolvedEdge {
                from: "src/main.rs".to_string(),
                to: "src/lib.rs".to_string(),
                import_symbol: Some("crate::lib".to_string()),
            },
            ResolvedEdge {
                from: "src/lib.rs".to_string(),
                to: "src/utils.rs".to_string(),
                import_symbol: Some("crate::utils".to_string()),
            },
        ];

        let graph = DepGraph::from_edges(edges);

        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.file_count(), 3);
        assert_eq!(graph.dependencies_of("src/main.rs").len(), 1);
        assert_eq!(graph.dependents_of("src/lib.rs").len(), 1);
    }

    #[test]
    fn test_dep_graph_remove_file() {
        let edges = vec![
            ResolvedEdge {
                from: "src/main.rs".to_string(),
                to: "src/lib.rs".to_string(),
                import_symbol: None,
            },
            ResolvedEdge {
                from: "src/lib.rs".to_string(),
                to: "src/utils.rs".to_string(),
                import_symbol: None,
            },
        ];

        let mut graph = DepGraph::from_edges(edges);
        graph.remove_file("src/lib.rs");

        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.file_count(), 2);
        assert!(!graph.files.contains("src/lib.rs"));
    }

    #[test]
    fn test_dep_graph_remove_edges_from() {
        let edges = vec![
            ResolvedEdge {
                from: "src/main.rs".to_string(),
                to: "src/lib.rs".to_string(),
                import_symbol: None,
            },
            ResolvedEdge {
                from: "src/main.rs".to_string(),
                to: "src/utils.rs".to_string(),
                import_symbol: None,
            },
        ];

        let mut graph = DepGraph::from_edges(edges);
        graph.remove_edges_from("src/main.rs");

        assert_eq!(graph.edge_count(), 0);
        assert!(graph.outgoing.is_empty());
    }
}
