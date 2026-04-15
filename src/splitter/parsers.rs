// Skipped grammars: Kotlin (`tree-sitter-kotlin` 0.3 uses the older tree-sitter
// `Language` API and is incompatible with this crate's tree-sitter 0.26 parser setup).
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

/// Returns a parser for the specified language.
///
/// Supported languages: python, javascript, typescript, tsx, rust, go, java, cpp,
/// c, ruby, php, swift, scala, csharp
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
        "ruby" | "rb" => tree_sitter_ruby::LANGUAGE,
        "php" => tree_sitter_php::LANGUAGE_PHP,
        "swift" => tree_sitter_swift::LANGUAGE,
        "scala" => tree_sitter_scala::LANGUAGE,
        "csharp" | "c#" | "cs" => tree_sitter_c_sharp::LANGUAGE,
        _ => return None,
    };

    let parser = LanguageParser::new(lang.into());
    Some(parser.parser)
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
        "rb" => Some("ruby"),
        "php" => Some("php"),
        "swift" => Some("swift"),
        "scala" => Some("scala"),
        "cs" => Some("csharp"),
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
    fn test_get_parser_ruby() {
        let parser = get_parser("ruby");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_php() {
        let parser = get_parser("php");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_swift() {
        let parser = get_parser("swift");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_scala() {
        let parser = get_parser("scala");
        assert!(parser.is_some());
    }

    #[test]
    fn test_get_parser_csharp() {
        let parser = get_parser("csharp");
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
        assert_eq!(extension_to_language("rb"), Some("ruby"));
        assert_eq!(extension_to_language("php"), Some("php"));
        assert_eq!(extension_to_language("swift"), Some("swift"));
        assert_eq!(extension_to_language("scala"), Some("scala"));
        assert_eq!(extension_to_language("cs"), Some("csharp"));
        assert_eq!(extension_to_language("unknown"), None);
    }

    #[test]
    fn test_get_parser_for_extension() {
        assert!(get_parser_for_extension("py").is_some());
        assert!(get_parser_for_extension("js").is_some());
        assert!(get_parser_for_extension("rb").is_some());
        assert!(get_parser_for_extension("php").is_some());
        assert!(get_parser_for_extension("swift").is_some());
        assert!(get_parser_for_extension("scala").is_some());
        assert!(get_parser_for_extension("cs").is_some());
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
