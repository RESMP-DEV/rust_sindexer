//! Splittable AST node types per language.
//!
//! Defines which tree-sitter node types represent meaningful code units
//! that should be extracted as separate chunks during semantic splitting.

use std::collections::HashMap;
use std::sync::LazyLock;

/// Node types that represent splittable semantic units for each language.
///
/// Keys are language identifiers (matching tree-sitter grammar names).
/// Values are slices of node type names that should be treated as chunk boundaries.
pub static SPLITTABLE_NODES: LazyLock<HashMap<&'static str, &'static [&'static str]>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();

        // Python
        m.insert(
            "python",
            &[
                "function_definition",
                "class_definition",
                "decorated_definition",
            ][..],
        );

        // JavaScript
        m.insert(
            "javascript",
            &[
                "function_declaration",
                "arrow_function",
                "class_declaration",
                "method_definition",
                "export_statement",
            ][..],
        );

        // TypeScript (same as JavaScript plus type declarations)
        m.insert(
            "typescript",
            &[
                "function_declaration",
                "arrow_function",
                "class_declaration",
                "method_definition",
                "export_statement",
                "interface_declaration",
                "type_alias_declaration",
            ][..],
        );

        // TSX (TypeScript with JSX)
        m.insert(
            "tsx",
            &[
                "function_declaration",
                "arrow_function",
                "class_declaration",
                "method_definition",
                "export_statement",
                "interface_declaration",
                "type_alias_declaration",
            ][..],
        );

        // Rust
        m.insert(
            "rust",
            &[
                "function_item",
                "impl_item",
                "struct_item",
                "enum_item",
                "trait_item",
                "mod_item",
                "macro_definition",
            ][..],
        );

        // Go
        m.insert(
            "go",
            &[
                "function_declaration",
                "method_declaration",
                "type_declaration",
            ][..],
        );

        // Java
        m.insert(
            "java",
            &[
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "constructor_declaration",
                "enum_declaration",
            ][..],
        );

        // C++
        m.insert(
            "cpp",
            &[
                "function_definition",
                "class_specifier",
                "namespace_definition",
                "template_declaration",
            ][..],
        );

        // C
        m.insert(
            "c",
            &["function_definition", "struct_specifier", "enum_specifier"][..],
        );

        // Ruby
        m.insert(
            "ruby",
            &["method_definition", "class_definition", "module_definition"][..],
        );

        // PHP
        m.insert(
            "php",
            &[
                "function_definition",
                "class_declaration",
                "method_declaration",
            ][..],
        );

        // Swift
        m.insert(
            "swift",
            &[
                "function_declaration",
                "class_declaration",
                "struct_declaration",
            ][..],
        );

        // Scala
        m.insert(
            "scala",
            &["function_definition", "class_definition", "object_definition"][..],
        );

        // C#
        m.insert(
            "csharp",
            &["method_declaration", "class_declaration", "struct_declaration"][..],
        );

        m
    });

/// Returns the splittable node types for a given language.
///
/// Returns `None` if the language is not supported.
#[inline]
pub fn get_splittable_nodes(language: &str) -> Option<&'static [&'static str]> {
    SPLITTABLE_NODES.get(language).copied()
}

/// Returns true if the given node type is splittable for the specified language.
#[inline]
pub fn is_splittable(language: &str, node_type: &str) -> bool {
    get_splittable_nodes(language)
        .map(|nodes| nodes.contains(&node_type))
        .unwrap_or(false)
}

/// Returns all supported languages.
pub fn supported_languages() -> impl Iterator<Item = &'static str> {
    SPLITTABLE_NODES.keys().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_nodes() {
        let nodes = get_splittable_nodes("python").unwrap();
        assert!(nodes.contains(&"function_definition"));
        assert!(nodes.contains(&"class_definition"));
        assert!(nodes.contains(&"decorated_definition"));
        assert!(!nodes.contains(&"async_function_definition"));
    }

    #[test]
    fn test_rust_nodes() {
        let nodes = get_splittable_nodes("rust").unwrap();
        assert!(nodes.contains(&"function_item"));
        assert!(nodes.contains(&"impl_item"));
        assert!(nodes.contains(&"struct_item"));
        assert!(nodes.contains(&"enum_item"));
        assert!(nodes.contains(&"trait_item"));
        assert!(nodes.contains(&"mod_item"));
    }

    #[test]
    fn test_javascript_nodes() {
        let nodes = get_splittable_nodes("javascript").unwrap();
        assert!(nodes.contains(&"function_declaration"));
        assert!(nodes.contains(&"arrow_function"));
        assert!(nodes.contains(&"class_declaration"));
        assert!(nodes.contains(&"method_definition"));
        assert!(nodes.contains(&"export_statement"));
    }

    #[test]
    fn test_go_nodes() {
        let nodes = get_splittable_nodes("go").unwrap();
        assert!(nodes.contains(&"function_declaration"));
        assert!(nodes.contains(&"method_declaration"));
        assert!(nodes.contains(&"type_declaration"));
    }

    #[test]
    fn test_java_nodes() {
        let nodes = get_splittable_nodes("java").unwrap();
        assert!(nodes.contains(&"method_declaration"));
        assert!(nodes.contains(&"class_declaration"));
        assert!(nodes.contains(&"interface_declaration"));
        assert!(nodes.contains(&"constructor_declaration"));
    }

    #[test]
    fn test_cpp_nodes() {
        let nodes = get_splittable_nodes("cpp").unwrap();
        assert!(nodes.contains(&"function_definition"));
        assert!(nodes.contains(&"class_specifier"));
        assert!(nodes.contains(&"namespace_definition"));
    }

    #[test]
    fn test_is_splittable() {
        assert!(is_splittable("python", "function_definition"));
        assert!(is_splittable("rust", "impl_item"));
        assert!(is_splittable("ruby", "method_definition"));
        assert!(is_splittable("php", "class_declaration"));
        assert!(is_splittable("swift", "struct_declaration"));
        assert!(is_splittable("scala", "object_definition"));
        assert!(is_splittable("csharp", "struct_declaration"));
        assert!(!is_splittable("python", "expression_statement"));
        assert!(!is_splittable("unknown_lang", "function_definition"));
    }

    #[test]
    fn test_unsupported_language() {
        assert!(get_splittable_nodes("cobol").is_none());
    }

    #[test]
    fn test_supported_languages() {
        let langs: Vec<_> = supported_languages().collect();
        assert!(langs.contains(&"python"));
        assert!(langs.contains(&"rust"));
        assert!(langs.contains(&"javascript"));
        assert!(langs.contains(&"typescript"));
        assert!(langs.contains(&"go"));
        assert!(langs.contains(&"java"));
        assert!(langs.contains(&"cpp"));
        assert!(langs.contains(&"c"));
        assert!(langs.contains(&"ruby"));
        assert!(langs.contains(&"php"));
        assert!(langs.contains(&"swift"));
        assert!(langs.contains(&"scala"));
        assert!(langs.contains(&"csharp"));
    }
}
