use std::fs::{self, File};
use std::io::Write;

use sindexer::cli;
use sindexer::overview;
use tempfile::TempDir;

/// Create a test file (and its parent directories).
fn create_file(dir: &TempDir, path: &str) {
    let full_path = dir.path().join(path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    File::create(&full_path)
        .unwrap()
        .write_all(b"content")
        .unwrap();
}

/// Initialize a fake git repository so .gitignore rules apply.
fn init_git_repo(dir: &TempDir) {
    fs::create_dir_all(dir.path().join(".git")).unwrap();
}

#[test]
fn overview_walk_uses_indexing_ignore_semantics() {
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(&dir);
    create_file(&dir, "src/main.rs");
    create_file(&dir, "src/lib.rs");
    create_file(&dir, "src/deep/x.rs");
    create_file(&dir, "README.md");
    create_file(&dir, "docs/a.md");
    // All three below must be excluded: DEFAULT_IGNORE_PATTERNS, hidden,
    // gitignore respectively.
    create_file(&dir, "target/junk.rs");
    create_file(&dir, ".hidden.rs");
    fs::write(dir.path().join(".gitignore"), "ignored_dir/\n").unwrap();
    create_file(&dir, "ignored_dir/z.rs");

    let overview = overview::overview(dir.path(), 3, 0, false).unwrap();

    assert_eq!(overview.total_files, 5);
    assert_eq!(overview.total_dirs, 3); // src, src/deep, docs
    assert_eq!(overview.root_files, 1);
    let src = overview.dirs.iter().find(|d| d.path == "src").unwrap();
    assert_eq!((src.files, src.subtree_files, src.subtree_dirs), (2, 3, 1));
    assert_eq!(src.ext, "rs");
    assert_eq!(overview.languages[0], ("rs".to_string(), 3));
    // Deterministic ordering: (depth, path).
    let paths: Vec<&str> = overview.dirs.iter().map(|d| d.path.as_str()).collect();
    assert_eq!(paths, ["docs", "src", "src/deep"]);
}

#[test]
fn overview_token_budget_binds_rendered_output() {
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(&dir);
    for i in 0..30 {
        create_file(&dir, &format!("pkg{i}/mod.rs"));
        create_file(&dir, &format!("pkg{i}/inner/deep/file{i}.rs"));
    }
    let overview = overview::overview(dir.path(), 4, 300, false).unwrap();
    assert!(
        overview.token_estimate <= 300 || overview.effective_depth == 1,
        "estimate {} depth {}",
        overview.token_estimate,
        overview.effective_depth
    );
}

#[tokio::test]
async fn overview_verb_cli_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(&dir);
    create_file(&dir, "src/a.rs");
    create_file(&dir, "README.md");
    let args: Vec<String> = vec![
        "overview".to_string(),
        dir.path().display().to_string(),
        "--human".to_string(),
    ];
    assert_eq!(cli::run(&args).await.unwrap(), 0);
}
