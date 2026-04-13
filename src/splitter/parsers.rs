use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::Mutex;
use tree_sitter::Parser;

/// Manages tree-sitter parsers for multiple programming languages.
pub struct LanguageParser {
    parser: Parser,
}

impl LanguageParser {
    /// Creates a new LanguageParser with the specified tree-sitter language.
    fn new(language: tree_sitter::Language) -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .expect("Failed to set parser language");
        Self { parser }
    }

    /// Returns a reference to the underlying parser.
    pub fn parser(&self) -> &Parser {
        &self.parser
    }

    /// Returns a mutable reference to the underlying parser.
    pub fn parser_mut(&mut self) -> &mut Parser {
        &mut self.parser
    }

    /// Parses source code and returns the syntax tree.
    pub fn parse(&mut self, source: &str) -> Option<tree_sitter::Tree> {
        self.parser.parse(source, None)
    }
}

/// Global parser cache using OnceCell for thread-safe lazy initialization.
static PARSER_CACHE: OnceCell<Mutex<HashMap<&'static str, Parser>>> = OnceCell::new();

/// Returns a parser for the specified language.
///
/// Supported languages: python, javascript, typescript, tsx, rust, go, java, cpp, c
///
/// Returns None if the language is not supported.
pub fn get_parser(language: &str) -> Option<Parser> {
    let lang = match language.to_lowercase().as_str() {
        "python" | "py" => tree_sitter_python::LANGUAGE,
        "javascript" | "js" => tree_sitter_javascript::LANGUAGE,
        "typescript" | "ts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX,
        "rust" | "rs" => tree_sitter_rust::LANGUAGE,
        "go" => tree_sitter_go::LANGUAGE,
        "java" => tree_sitter_java::LANGUAGE,
        "cpp" | "c++" | "cxx" | "cc" => tree_sitter_cpp::LANGUAGE,
        "c" => tree_sitter_c::LANGUAGE,
        _ => return None,
    };

    let mut parser = Parser::new();
    parser.set_language(&lang.into()).ok()?;
    Some(parser)
}

/// Maps a file extension to a language name.
///
/// Returns None if the extension is not recognized.
pub fn extension_to_language(ext: &str) -> Option<&'static str> {
    match ext.to_lowercase().as_str() {
        "py" | "pyw" | "pyi" => Some("python"),
        "js" | "mjs" | "cjs" => Some("javascript"),
        "ts" | "mts" | "cts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "jsx" => Some("javascript"),
        "rs" => Some("rust"),
        "go" => Some("go"),
        "java" => Some("java"),
        "cpp" | "cxx" | "cc" | "c++" | "hpp" | "hxx" | "h++" => Some("cpp"),
        "c" | "h" => Some("c"),
        _ => None,
    }
}

/// Returns a parser based on file extension.
///
/// Convenience function that combines extension_to_language and get_parser.
pub fn get_parser_for_extension(ext: &str) -> Option<Parser> {
    extension_to_language(ext).and_then(get_parser)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_parser_python() {
        let parser = get_parser("python");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_javascript() {
        let parser = get_parser("javascript");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_typescript() {
        let parser = get_parser("typescript");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_rust() {
        let parser = get_parser("rust");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_go() {
        let parser = get_parser("go");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_java() {
        let parser = get_parser("java");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_cpp() {
        let parser = get_parser("cpp");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_c() {
        let parser = get_parser("c");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_unknown() {
        let parser = get_parser("unknown");
        assert!(parser.is_none());
    }

    #[test]
    fn test_extension_to_language() {
        assert_eq!(extension_to_language("py"), Some("python"));
        assert_eq!(extension_to_language("js"), Some("javascript"));
        assert_eq!(extension_to_language("ts"), Some("typescript"));
        assert_eq!(extension_to_language("tsx"), Some("tsx"));
        assert_eq!(extension_to_language("rs"), Some("rust"));
        assert_eq!(extension_to_language("go"), Some("go"));
        assert_eq!(extension_to_language("java"), Some("java"));
        assert_eq!(extension_to_language("cpp"), Some("cpp"));
        assert_eq!(extension_to_language("c"), Some("c"));
        assert_eq!(extension_to_language("unknown"), None);
    }

    #[test]
    fn test_get_parser_for_extension() {
        assert!(get_parser_for_extension("py").is_some());
        assert!(get_parser_for_extension("js").is_some());
        assert!(get_parser_for_extension("unknown").is_none());
    }

    #[test]
    fn test_language_parser_parse() {
        let lang = tree_sitter_python::LANGUAGE;
        let mut lp = LanguageParser::new(lang.into());
        let tree = lp.parse("def foo(): pass");
        assert!(tree.is_some());
    }
}
