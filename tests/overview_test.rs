use std::fs::{self, File};
use std::io::Write;

use sindexer::cli;
use sindexer::overview;
use tempfile::TempDir;

/// Build a CLI argument vector for parser tests.
fn args(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| part.to_string()).collect()
}

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
    fs::create_dir(dir.path().join("empty_dir")).unwrap();

    let overview = overview::overview(dir.path(), 3, 0, false, false).unwrap();

    assert_eq!(overview.total_files, 5);
    assert_eq!(overview.total_dirs, 4); // src, src/deep, docs, empty_dir
    assert_eq!(overview.root_files, 1);
    let src = overview.dirs.iter().find(|d| d.path == "src").unwrap();
    assert_eq!((src.files, src.subtree_files, src.subtree_dirs), (2, 3, 1));
    assert_eq!(src.ext, "rs");
    assert_eq!(overview.languages[0], ("rs".to_string(), 3));
    // Deterministic ordering: (depth, path).
    let paths: Vec<&str> = overview.dirs.iter().map(|d| d.path.as_str()).collect();
    assert_eq!(paths, ["docs", "empty_dir", "src", "src/deep"]);
}

#[test]
fn overview_token_budget_binds_rendered_output() {
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(&dir);
    for i in 0..30 {
        create_file(&dir, &format!("pkg{i}/mod.rs"));
        create_file(&dir, &format!("pkg{i}/inner/deep/file{i}.rs"));
    }
    let overview = overview::overview(dir.path(), 4, 300, false, false).unwrap();
    let rendered = overview::render_json(&overview);
    assert_eq!(overview.token_estimate, rendered.len().div_ceil(4));
    assert_eq!(overview.over_budget, overview.token_estimate > 300);
    assert_eq!(
        overview.omitted_dirs,
        overview.total_dirs - overview.dirs.len()
    );
    if !overview.over_budget {
        assert!(overview.token_estimate <= 300);
    }
}

#[test]
fn overview_reports_top_level_pruning_and_largest_dirs() {
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(&dir);
    for i in 0..12 {
        for file_count in 0..=i {
            create_file(&dir, &format!("pkg{i:02}/file{file_count}.rs"));
        }
        create_file(&dir, &format!("pkg{i:02}/deep/inner/file.rs"));
    }
    let overview = overview::overview(dir.path(), 2, 180, true, false).unwrap();
    let rendered = overview::render_human(&overview);
    assert_eq!(overview.token_estimate, rendered.len().div_ceil(4));
    assert_eq!(
        overview.omitted_dirs,
        overview.total_dirs - overview.dirs.len()
    );
    assert!(overview.omitted_dirs > 0);
    let retained: Vec<usize> = overview
        .dirs
        .iter()
        .map(|entry| entry.path.trim_start_matches("pkg").parse().unwrap())
        .collect();
    assert_eq!(retained, (12 - retained.len()..12).collect::<Vec<_>>());
}

#[test]
fn overview_walk_follows_configured_symlinks() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        init_git_repo(&dir);
        create_file(&dir, "src/main.rs");
        create_file(&dir, "external/lib.rs");
        symlink(dir.path().join("external"), dir.path().join("src/link")).unwrap();

        let following = overview::overview(dir.path(), 3, 0, false, true).unwrap();
        let not_following = overview::overview(dir.path(), 3, 0, false, false).unwrap();

        // The target and the file reached through the link are both walked.
        assert_eq!(following.total_files, 3);
        assert!(following.dirs.iter().any(|entry| entry.path == "src/link"));
        // The real target is still present; only the link-expanded copy is not.
        assert_eq!(not_following.total_files, 2);
        assert!(!not_following
            .dirs
            .iter()
            .any(|entry| entry.path == "src/link"));
    }
}

#[cfg(unix)]
#[test]
fn overview_reports_unread_walk_entries() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    init_git_repo(&dir);
    create_file(&dir, "src/main.rs");
    symlink(dir.path().join("does-not-exist"), dir.path().join("broken")).unwrap();

    let overview = overview::overview(dir.path(), 2, 0, false, true).unwrap();
    assert_eq!(overview.total_files, 1);
    assert_eq!(overview.walk_errors, 1);
    assert!(overview::render_json(&overview).contains(r#""walk_errors":1"#));
    assert!(overview::render_human(&overview).contains("output is incomplete"));
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

#[tokio::test]
async fn overview_verb_supports_end_of_flags() {
    // Parsing must reach runtime path validation rather than reject a dash
    // path as an unknown flag. A nonexistent dash path is therefore Err (1),
    // not the parser's Ok(2).
    assert!(cli::run(&args(&["overview", "--", "-dash-path"]))
        .await
        .is_err());
    assert!(cli::run(&args(&["overview", "-"])).await.is_err());
    assert_eq!(
        cli::run(&args(&["overview", "-dash", "/definitely/not/here"]))
            .await
            .unwrap(),
        2
    );
}
