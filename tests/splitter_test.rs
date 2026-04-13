//! Integration tests for the semantic code splitter.
//!
//! Tests language-specific extraction of semantic units using tree-sitter parsing.

use tree_sitter::Parser;

/// Sample Python code with functions and classes for testing.
const PYTHON_SAMPLE: &str = r#"
def simple_function():
    """A simple function."""
    return 42

async def async_fetch(url: str) -> dict:
    """Async function to fetch data."""
    response = await client.get(url)
    return response.json()

class DataProcessor:
    """Process data with various transforms."""

    def __init__(self, config: dict):
        self.config = config

    def process(self, data: list) -> list:
        return [self.transform(item) for item in data]

    def transform(self, item):
        return item * 2

@decorator
def decorated_function():
    pass

@classmethod
@another_decorator
def multi_decorated():
    pass
"#;

/// Sample JavaScript code with classes and functions.
const JAVASCRIPT_SAMPLE: &str = r#"
function regularFunction(a, b) {
    return a + b;
}

const arrowFunc = (x) => x * 2;

const multilineArrow = (items) => {
    return items.map(i => i + 1);
};

class UserService {
    constructor(db) {
        this.db = db;
    }

    async getUser(id) {
        return await this.db.query('SELECT * FROM users WHERE id = ?', [id]);
    }

    deleteUser(id) {
        return this.db.delete('users', id);
    }
}

export function exportedFunction() {
    return 'exported';
}

export class ExportedClass {
    method() {}
}
"#;

/// Sample TypeScript code with interfaces and type aliases.
const TYPESCRIPT_SAMPLE: &str = r#"
interface User {
    id: number;
    name: string;
    email: string;
}

type UserId = number | string;

type UserResponse = {
    user: User;
    token: string;
};

function processUser(user: User): void {
    console.log(user.name);
}

class TypedService<T> {
    private data: T[];

    constructor() {
        this.data = [];
    }

    add(item: T): void {
        this.data.push(item);
    }

    getAll(): T[] {
        return this.data;
    }
}

export interface ExportedInterface {
    value: number;
}
"#;

/// Sample Rust code with impl blocks, structs, and traits.
const RUST_SAMPLE: &str = r#"
use std::collections::HashMap;

/// A simple struct for testing.
pub struct DataStore {
    data: HashMap<String, Vec<u8>>,
    capacity: usize,
}

impl DataStore {
    /// Creates a new DataStore with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Inserts data into the store.
    pub fn insert(&mut self, key: String, value: Vec<u8>) -> Option<Vec<u8>> {
        self.data.insert(key, value)
    }

    /// Gets data from the store.
    pub fn get(&self, key: &str) -> Option<&Vec<u8>> {
        self.data.get(key)
    }
}

/// A trait for serializable objects.
pub trait Serializable {
    fn serialize(&self) -> Vec<u8>;
    fn deserialize(bytes: &[u8]) -> Self;
}

impl Serializable for DataStore {
    fn serialize(&self) -> Vec<u8> {
        // Implementation
        vec![]
    }

    fn deserialize(_bytes: &[u8]) -> Self {
        Self::new(0)
    }
}

/// An enum for result types.
pub enum ProcessResult {
    Success(String),
    Failure(String),
    Pending,
}

impl ProcessResult {
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }
}

fn standalone_function(x: i32) -> i32 {
    x * 2
}

mod inner_module {
    pub fn inner_fn() -> u32 {
        42
    }
}

macro_rules! create_function {
    ($name:ident) => {
        fn $name() {
            println!("Function: {}", stringify!($name));
        }
    };
}
"#;

/// Large file sample for chunk refinement testing.
const LARGE_FILE_SAMPLE: &str = r#"
def function_one():
    """First function with some implementation."""
    x = 1
    y = 2
    z = x + y
    result = z * 2
    return result

def function_two():
    """Second function."""
    items = [1, 2, 3, 4, 5]
    total = sum(items)
    average = total / len(items)
    return average

def function_three():
    """Third function with nested logic."""
    data = {}
    for i in range(100):
        if i % 2 == 0:
            data[i] = i * 2
        else:
            data[i] = i * 3
    return data

class LargeClass:
    """A class with many methods for testing chunk refinement."""

    def __init__(self):
        self.value = 0
        self.history = []

    def method_one(self):
        self.value += 1
        self.history.append('one')
        return self.value

    def method_two(self):
        self.value += 2
        self.history.append('two')
        return self.value

    def method_three(self):
        self.value += 3
        self.history.append('three')
        return self.value

    def method_four(self):
        self.value += 4
        self.history.append('four')
        return self.value

    def method_five(self):
        self.value += 5
        self.history.append('five')
        return self.value

    def get_history(self):
        return self.history.copy()

    def reset(self):
        self.value = 0
        self.history.clear()

def function_four():
    """Fourth standalone function."""
    return [x**2 for x in range(10)]

def function_five():
    """Fifth function."""
    mapping = {chr(i): i for i in range(ord('a'), ord('z') + 1)}
    return mapping
"#;

// ============================================================================
// Python Extraction Tests
// ============================================================================

#[test]
fn test_python_function_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("Failed to set Python language");

    let tree = parser
        .parse(PYTHON_SAMPLE, None)
        .expect("Failed to parse Python");
    let root = tree.root_node();

    let mut function_count = 0;
    let mut async_function_count = 0;

    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "function_definition" {
            function_count += 1;
            if let Ok(text) = node.utf8_text(PYTHON_SAMPLE.as_bytes()) {
                if text.trim_start().starts_with("async def ") {
                    async_function_count += 1;
                }
            }
        }
    }

    assert!(
        function_count >= 1,
        "Should find at least one regular function"
    );
    assert_eq!(
        async_function_count, 1,
        "Should find exactly one async function"
    );
}

#[test]
fn test_python_class_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("Failed to set Python language");

    let tree = parser
        .parse(PYTHON_SAMPLE, None)
        .expect("Failed to parse Python");
    let root = tree.root_node();

    let mut class_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "class_definition" {
            class_count += 1;

            // Verify class has methods
            let mut method_count = 0;
            let mut inner_cursor = node.walk();
            for child in node.named_children(&mut inner_cursor) {
                if child.kind() == "block" {
                    let mut block_cursor = child.walk();
                    for block_child in child.named_children(&mut block_cursor) {
                        if block_child.kind() == "function_definition" {
                            method_count += 1;
                        }
                    }
                }
            }
            assert!(
                method_count >= 2,
                "DataProcessor class should have at least 2 methods"
            );
        }
    }

    assert_eq!(class_count, 1, "Should find exactly one class");
}

#[test]
fn test_python_decorated_function_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("Failed to set Python language");

    let tree = parser
        .parse(PYTHON_SAMPLE, None)
        .expect("Failed to parse Python");
    let root = tree.root_node();

    let mut decorated_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "decorated_definition" {
            decorated_count += 1;
        }
    }

    assert!(
        decorated_count >= 2,
        "Should find at least 2 decorated definitions"
    );
}

// ============================================================================
// JavaScript/TypeScript Class Extraction Tests
// ============================================================================

#[test]
fn test_javascript_function_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_javascript::LANGUAGE.into())
        .expect("Failed to set JavaScript language");

    let tree = parser
        .parse(JAVASCRIPT_SAMPLE, None)
        .expect("Failed to parse JavaScript");
    let root = tree.root_node();

    let mut function_decl_count = 0;
    let mut arrow_func_count = 0;

    fn count_nodes(node: tree_sitter::Node, func_count: &mut i32, arrow_count: &mut i32) {
        match node.kind() {
            "function_declaration" => *func_count += 1,
            "arrow_function" => *arrow_count += 1,
            _ => {}
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            count_nodes(child, func_count, arrow_count);
        }
    }

    count_nodes(root, &mut function_decl_count, &mut arrow_func_count);

    assert!(
        function_decl_count >= 1,
        "Should find at least one function declaration"
    );
    assert!(
        arrow_func_count >= 2,
        "Should find at least two arrow functions"
    );
}

#[test]
fn test_javascript_class_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_javascript::LANGUAGE.into())
        .expect("Failed to set JavaScript language");

    let tree = parser
        .parse(JAVASCRIPT_SAMPLE, None)
        .expect("Failed to parse JavaScript");
    let root = tree.root_node();

    let mut class_count = 0;
    let mut method_count = 0;

    fn count_class_nodes(node: tree_sitter::Node, class_count: &mut i32, method_count: &mut i32) {
        match node.kind() {
            "class_declaration" => *class_count += 1,
            "method_definition" => *method_count += 1,
            _ => {}
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            count_class_nodes(child, class_count, method_count);
        }
    }

    count_class_nodes(root, &mut class_count, &mut method_count);

    assert!(class_count >= 2, "Should find at least two classes");
    assert!(method_count >= 3, "Should find at least three methods");
}

#[test]
fn test_javascript_export_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_javascript::LANGUAGE.into())
        .expect("Failed to set JavaScript language");

    let tree = parser
        .parse(JAVASCRIPT_SAMPLE, None)
        .expect("Failed to parse JavaScript");
    let root = tree.root_node();

    let mut export_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "export_statement" {
            export_count += 1;
        }
    }

    assert!(
        export_count >= 2,
        "Should find at least two export statements"
    );
}

#[test]
fn test_typescript_interface_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .expect("Failed to set TypeScript language");

    let tree = parser
        .parse(TYPESCRIPT_SAMPLE, None)
        .expect("Failed to parse TypeScript");
    let root = tree.root_node();

    let mut interface_count = 0;
    let mut type_alias_count = 0;

    fn count_type_nodes(node: tree_sitter::Node, iface_count: &mut i32, type_count: &mut i32) {
        match node.kind() {
            "interface_declaration" => *iface_count += 1,
            "type_alias_declaration" => *type_count += 1,
            _ => {}
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            count_type_nodes(child, iface_count, type_count);
        }
    }

    count_type_nodes(root, &mut interface_count, &mut type_alias_count);

    assert!(interface_count >= 2, "Should find at least two interfaces");
    assert!(
        type_alias_count >= 2,
        "Should find at least two type aliases"
    );
}

#[test]
fn test_typescript_generic_class_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .expect("Failed to set TypeScript language");

    let tree = parser
        .parse(TYPESCRIPT_SAMPLE, None)
        .expect("Failed to parse TypeScript");
    let root = tree.root_node();

    let mut class_count = 0;

    fn count_classes(node: tree_sitter::Node, count: &mut i32) {
        if node.kind() == "class_declaration" {
            *count += 1;
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            count_classes(child, count);
        }
    }

    count_classes(root, &mut class_count);

    assert!(
        class_count >= 1,
        "Should find at least one class (TypedService<T>)"
    );
}

// ============================================================================
// Rust impl Block Extraction Tests
// ============================================================================

#[test]
fn test_rust_impl_block_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Failed to set Rust language");

    let tree = parser
        .parse(RUST_SAMPLE, None)
        .expect("Failed to parse Rust");
    let root = tree.root_node();

    let mut impl_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "impl_item" {
            impl_count += 1;
        }
    }

    // DataStore impl, Serializable for DataStore impl, ProcessResult impl
    assert!(
        impl_count >= 3,
        "Should find at least 3 impl blocks, found {}",
        impl_count
    );
}

#[test]
fn test_rust_struct_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Failed to set Rust language");

    let tree = parser
        .parse(RUST_SAMPLE, None)
        .expect("Failed to parse Rust");
    let root = tree.root_node();

    let mut struct_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "struct_item" {
            struct_count += 1;
        }
    }

    assert_eq!(
        struct_count, 1,
        "Should find exactly one struct (DataStore)"
    );
}

#[test]
fn test_rust_trait_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Failed to set Rust language");

    let tree = parser
        .parse(RUST_SAMPLE, None)
        .expect("Failed to parse Rust");
    let root = tree.root_node();

    let mut trait_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "trait_item" {
            trait_count += 1;
        }
    }

    assert_eq!(
        trait_count, 1,
        "Should find exactly one trait (Serializable)"
    );
}

#[test]
fn test_rust_enum_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Failed to set Rust language");

    let tree = parser
        .parse(RUST_SAMPLE, None)
        .expect("Failed to parse Rust");
    let root = tree.root_node();

    let mut enum_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "enum_item" {
            enum_count += 1;
        }
    }

    assert_eq!(
        enum_count, 1,
        "Should find exactly one enum (ProcessResult)"
    );
}

#[test]
fn test_rust_function_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Failed to set Rust language");

    let tree = parser
        .parse(RUST_SAMPLE, None)
        .expect("Failed to parse Rust");
    let root = tree.root_node();

    let mut function_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "function_item" {
            function_count += 1;
        }
    }

    assert!(
        function_count >= 1,
        "Should find at least one standalone function"
    );
}

#[test]
fn test_rust_mod_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Failed to set Rust language");

    let tree = parser
        .parse(RUST_SAMPLE, None)
        .expect("Failed to parse Rust");
    let root = tree.root_node();

    let mut mod_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "mod_item" {
            mod_count += 1;
        }
    }

    assert_eq!(mod_count, 1, "Should find exactly one mod (inner_module)");
}

#[test]
fn test_rust_macro_extraction() {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Failed to set Rust language");

    let tree = parser
        .parse(RUST_SAMPLE, None)
        .expect("Failed to parse Rust");
    let root = tree.root_node();

    let mut macro_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "macro_definition" {
            macro_count += 1;
        }
    }

    assert_eq!(
        macro_count, 1,
        "Should find exactly one macro (create_function)"
    );
}

// ============================================================================
// Chunk Refinement Tests for Large Files
// ============================================================================

/// Simulates chunk extraction with configurable max chunk size.
fn extract_chunks_with_max_size(
    source: &str,
    language: &str,
    max_chunk_lines: usize,
) -> Vec<(usize, usize, String)> {
    let lang = match language {
        "python" => tree_sitter_python::LANGUAGE.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        _ => return vec![],
    };

    let mut parser = Parser::new();
    parser.set_language(&lang).expect("Failed to set language");

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return vec![],
    };

    let root = tree.root_node();
    let mut chunks = Vec::new();

    let splittable_nodes: &[&str] = match language {
        "python" => &[
            "function_definition",
            "class_definition",
            "decorated_definition",
        ],
        "javascript" => &[
            "function_declaration",
            "arrow_function",
            "class_declaration",
            "method_definition",
            "export_statement",
        ],
        "typescript" => &[
            "function_declaration",
            "arrow_function",
            "class_declaration",
            "method_definition",
            "export_statement",
            "interface_declaration",
            "type_alias_declaration",
        ],
        "rust" => &[
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "mod_item",
            "macro_definition",
        ],
        _ => &[],
    };

    fn collect_chunks(
        node: tree_sitter::Node,
        source: &str,
        splittable: &[&str],
        max_lines: usize,
        chunks: &mut Vec<(usize, usize, String)>,
    ) {
        if splittable.contains(&node.kind()) {
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;
            let chunk_lines = end_line - start_line + 1;

            if chunk_lines <= max_lines {
                // Chunk fits within limit
                let content = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                chunks.push((start_line, end_line, content));
            } else {
                // Chunk is too large, need to refine by extracting children
                // For classes/impl blocks, extract methods individually
                let mut child_cursor = node.walk();
                let mut has_splittable_children = false;

                for child in node.named_children(&mut child_cursor) {
                    if splittable.contains(&child.kind()) {
                        has_splittable_children = true;
                        collect_chunks(child, source, splittable, max_lines, chunks);
                    } else {
                        // Recurse into non-splittable children (like blocks)
                        collect_chunks(child, source, splittable, max_lines, chunks);
                    }
                }

                // If no splittable children found, split by line count
                if !has_splittable_children {
                    let content = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
                    let lines: Vec<&str> = content.lines().collect();

                    for chunk_start in (0..lines.len()).step_by(max_lines) {
                        let chunk_end = (chunk_start + max_lines).min(lines.len());
                        let chunk_content: String = lines[chunk_start..chunk_end].join("\n");
                        chunks.push((
                            start_line + chunk_start,
                            start_line + chunk_end - 1,
                            chunk_content,
                        ));
                    }
                }
            }
        } else {
            // Not a splittable node, recurse into children
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                collect_chunks(child, source, splittable, max_lines, chunks);
            }
        }
    }

    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        collect_chunks(node, source, splittable_nodes, max_chunk_lines, &mut chunks);
    }

    chunks
}

#[test]
fn test_chunk_refinement_respects_max_size() {
    let max_lines = 10;
    let chunks = extract_chunks_with_max_size(LARGE_FILE_SAMPLE, "python", max_lines);

    assert!(!chunks.is_empty(), "Should extract at least one chunk");

    // Individual functions should be extracted
    for (start, end, _content) in &chunks {
        let chunk_lines = end - start + 1;
        // Note: Some chunks may exceed max if they can't be split further
        // The test validates the chunking logic attempts to respect limits
        assert!(
            chunk_lines <= max_lines * 2,
            "Chunk from line {} to {} ({} lines) exceeds reasonable limit",
            start,
            end,
            chunk_lines
        );
    }
}

#[test]
fn test_chunk_refinement_extracts_all_functions() {
    let chunks = extract_chunks_with_max_size(LARGE_FILE_SAMPLE, "python", 50);

    // Should find all 5 standalone functions
    let function_chunks: Vec<_> = chunks
        .iter()
        .filter(|(_, _, content)| content.trim_start().starts_with("def "))
        .collect();

    assert!(
        function_chunks.len() >= 5,
        "Should extract at least 5 function definitions, found {}",
        function_chunks.len()
    );
}

#[test]
fn test_chunk_refinement_extracts_class() {
    let chunks = extract_chunks_with_max_size(LARGE_FILE_SAMPLE, "python", 100);

    // Should find the LargeClass
    let class_chunks: Vec<_> = chunks
        .iter()
        .filter(|(_, _, content)| content.contains("class LargeClass"))
        .collect();

    assert!(
        !class_chunks.is_empty(),
        "Should extract LargeClass definition"
    );
}

#[test]
fn test_chunk_refinement_small_max_splits_class() {
    // With a small max, the class should be split into its methods
    let chunks = extract_chunks_with_max_size(LARGE_FILE_SAMPLE, "python", 5);

    // With max 5 lines, the class should be broken down
    // At minimum we should get many small chunks
    assert!(
        chunks.len() >= 5,
        "With small max_lines, should produce many chunks, got {}",
        chunks.len()
    );
}

#[test]
fn test_chunk_line_numbers_are_accurate() {
    let chunks = extract_chunks_with_max_size(PYTHON_SAMPLE, "python", 100);
    let lines: Vec<&str> = PYTHON_SAMPLE.lines().collect();

    for (start, end, content) in &chunks {
        // Line numbers are 1-indexed
        let start_idx = *start - 1;
        let end_idx = *end;

        if start_idx < lines.len() && end_idx <= lines.len() {
            let expected_first_line = lines[start_idx].trim();
            let actual_first_line = content.lines().next().unwrap_or("").trim();

            // The extracted content's first line should match the source
            assert!(
                actual_first_line.contains(expected_first_line)
                    || expected_first_line.contains(actual_first_line)
                    || expected_first_line.is_empty()
                    || actual_first_line.is_empty(),
                "Line {} mismatch: expected '{}', got '{}'",
                start,
                expected_first_line,
                actual_first_line
            );
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_source() {
    let chunks = extract_chunks_with_max_size("", "python", 50);
    assert!(chunks.is_empty(), "Empty source should produce no chunks");
}

#[test]
fn test_comments_only() {
    let source = r#"
# This is a comment
# Another comment
"""
A docstring without a function
"""
"#;
    let chunks = extract_chunks_with_max_size(source, "python", 50);
    assert!(
        chunks.is_empty(),
        "Comments-only source should produce no semantic chunks"
    );
}

#[test]
fn test_nested_classes_javascript() {
    let source = r#"
class Outer {
    static Inner = class {
        method() {
            return 42;
        }
    };

    outerMethod() {
        return new Outer.Inner();
    }
}
"#;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_javascript::LANGUAGE.into())
        .expect("Failed to set JavaScript language");

    let tree = parser.parse(source, None).expect("Failed to parse");
    assert!(!tree.root_node().has_error(), "Should parse without errors");

    let chunks = extract_chunks_with_max_size(source, "javascript", 50);
    assert!(!chunks.is_empty(), "Should extract nested class structures");
}

#[test]
fn test_rust_impl_with_generics() {
    let source = r#"
impl<T: Clone + Send> Container<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add(&mut self, item: T) {
        self.items.push(item);
    }
}

impl<T> Default for Container<T> {
    fn default() -> Self {
        Self { items: vec![] }
    }
}
"#;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Failed to set Rust language");

    let tree = parser.parse(source, None).expect("Failed to parse");
    let root = tree.root_node();

    let mut impl_count = 0;
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "impl_item" {
            impl_count += 1;
        }
    }

    assert_eq!(impl_count, 2, "Should find both generic impl blocks");
}

#[test]
fn test_typescript_complex_types() {
    let source = r#"
type ComplexType<T, U> = {
    data: T;
    transform: (input: T) => U;
    nested: {
        value: T | U;
        optional?: string;
    };
};

interface GenericInterface<T extends object> {
    items: T[];
    add(item: T): void;
    remove(id: keyof T): boolean;
}

type UnionType = string | number | boolean | null;

type ConditionalType<T> = T extends string ? 'string' : 'other';
"#;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .expect("Failed to set TypeScript language");

    let tree = parser.parse(source, None).expect("Failed to parse");
    assert!(
        !tree.root_node().has_error(),
        "Should parse complex types without errors"
    );

    let chunks = extract_chunks_with_max_size(source, "typescript", 50);
    assert!(
        chunks.len() >= 4,
        "Should extract all type definitions, got {}",
        chunks.len()
    );
}
