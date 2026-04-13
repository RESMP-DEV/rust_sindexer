# Zilliz Claude-Context-Core Walker Analysis

Analysis of file discovery and ignore pattern handling in `@zilliz/claude-context-core/dist/context.js`.

## 1. DEFAULT_SUPPORTED_EXTENSIONS

Defined at lines 44-52. These are the file extensions that will be indexed:

```javascript
const DEFAULT_SUPPORTED_EXTENSIONS = [
    // Programming languages
    '.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.m', '.mm',
    // Text and markup files
    '.md', '.markdown', '.ipynb',
    // Commented out (not indexed by default):
    // '.txt', '.json', '.yaml', '.yml', '.xml', '.html', '.htm',
    // '.css', '.scss', '.less', '.sql', '.sh', '.bash', '.env'
];
```

**Total: 23 extensions** covering:
- TypeScript/JavaScript: `.ts`, `.tsx`, `.js`, `.jsx`
- Python: `.py`
- JVM languages: `.java`, `.kt`, `.scala`
- Systems languages: `.cpp`, `.c`, `.h`, `.hpp`, `.rs`, `.go`
- .NET: `.cs`
- Scripting: `.php`, `.rb`
- Mobile: `.swift`, `.m`, `.mm`
- Documentation: `.md`, `.markdown`
- Notebooks: `.ipynb`

**Notable omissions** (commented out in source):
- Config files: `.json`, `.yaml`, `.yml`, `.xml`
- Web: `.html`, `.htm`, `.css`, `.scss`, `.less`
- Shell: `.sh`, `.bash`
- Database: `.sql`
- Environment: `.env`, `.txt`

## 2. DEFAULT_IGNORE_PATTERNS

Defined at lines 53-98. These patterns exclude files and directories from indexing:

```javascript
const DEFAULT_IGNORE_PATTERNS = [
    // Common build output and dependency directories
    'node_modules/**',
    'dist/**',
    'build/**',
    'out/**',
    'target/**',
    'coverage/**',
    '.nyc_output/**',

    // IDE and editor files
    '.vscode/**',
    '.idea/**',
    '*.swp',
    '*.swo',

    // Version control
    '.git/**',
    '.svn/**',
    '.hg/**',

    // Cache directories
    '.cache/**',
    '__pycache__/**',
    '.pytest_cache/**',

    // Logs and temporary files
    'logs/**',
    'tmp/**',
    'temp/**',
    '*.log',

    // Environment and config files
    '.env',
    '.env.*',
    '*.local',

    // Minified and bundled files
    '*.min.js',
    '*.min.css',
    '*.min.map',
    '*.bundle.js',
    '*.bundle.css',
    '*.chunk.js',
    '*.vendor.js',
    '*.polyfills.js',
    '*.runtime.js',
    '*.map',  // source map files

    // Duplicated as simple directory names (no globstar)
    'node_modules', '.git', '.svn', '.hg', 'build', 'dist', 'out',
    'target', '.vscode', '.idea', '__pycache__', '.pytest_cache',
    'coverage', '.nyc_output', 'logs', 'tmp', 'temp'
];
```

**Total: 46 patterns** organized by category.

**Note:** Patterns appear twice - once with `/**` globstar suffix and once as plain directory names. This provides redundancy for different matching scenarios.

## 3. getCodeFiles() - File Discovery

Located at lines 547-570. Recursively traverses the codebase to find indexable files.

### Algorithm

```javascript
async getCodeFiles(codebasePath) {
    const files = [];

    const traverseDirectory = async (currentPath) => {
        const entries = await fs.promises.readdir(currentPath, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(currentPath, entry.name);

            // Check if path matches ignore patterns
            if (this.matchesIgnorePattern(fullPath, codebasePath)) {
                continue;
            }

            if (entry.isDirectory()) {
                await traverseDirectory(fullPath);  // Recurse into subdirectories
            } else if (entry.isFile()) {
                const ext = path.extname(entry.name);
                if (this.supportedExtensions.includes(ext)) {
                    files.push(fullPath);
                }
            }
        }
    };

    await traverseDirectory(codebasePath);
    return files;
}
```

### Key Behaviors

1. **Depth-first traversal**: Uses recursive async function
2. **Early pruning**: Checks ignore patterns BEFORE recursing into directories
3. **Extension filtering**: Only includes files with extensions in `supportedExtensions`
4. **Returns absolute paths**: Full paths are accumulated and returned

### Pattern Matching (lines 903-953)

The `matchesIgnorePattern()` method:
1. Converts Windows paths to forward slashes for consistency
2. For each pattern, calls `isPatternMatch()`

The `isPatternMatch()` method handles three cases:
1. **Directory patterns** (ending with `/`): Match any path component
2. **Path patterns** (containing `/`): Match exact relative path
3. **Simple patterns**: Match filename only (any directory depth)

The `simpleGlobMatch()` method converts glob patterns to regex:
- Escapes regex special characters except `*`
- Converts `*` to `.*` for wildcard matching

## 4. loadIgnorePatterns() - .gitignore Integration

Located at lines 803-828. Loads ignore patterns from multiple sources.

### Algorithm

```javascript
async loadIgnorePatterns(codebasePath) {
    try {
        let fileBasedPatterns = [];

        // 1. Find all .xxxignore files in codebase root
        const ignoreFiles = await this.findIgnoreFiles(codebasePath);
        for (const ignoreFile of ignoreFiles) {
            const patterns = await this.loadIgnoreFile(ignoreFile, path.basename(ignoreFile));
            fileBasedPatterns.push(...patterns);
        }

        // 2. Load global ~/.context/.contextignore
        const globalIgnorePatterns = await this.loadGlobalIgnoreFile();
        fileBasedPatterns.push(...globalIgnorePatterns);

        // 3. Merge with existing patterns (preserves MCP custom patterns)
        if (fileBasedPatterns.length > 0) {
            this.addCustomIgnorePatterns(fileBasedPatterns);
        }
    } catch (error) {
        // Continue with existing patterns on error
    }
}
```

### Ignore File Discovery (findIgnoreFiles, lines 834-854)

Scans the codebase root for files matching pattern: `.` + anything + `ignore`

Examples of matched files:
- `.gitignore`
- `.dockerignore`
- `.eslintignore`
- `.prettierignore`
- `.contextignore`

```javascript
async findIgnoreFiles(codebasePath) {
    const entries = await fs.promises.readdir(codebasePath, { withFileTypes: true });
    const ignoreFiles = [];

    for (const entry of entries) {
        if (entry.isFile() &&
            entry.name.startsWith('.') &&
            entry.name.endsWith('ignore')) {
            ignoreFiles.push(path.join(codebasePath, entry.name));
        }
    }
    return ignoreFiles;
}
```

### Ignore File Parsing (getIgnorePatternsFromFile, lines 785-797)

Static method that reads and parses ignore files:

```javascript
static async getIgnorePatternsFromFile(filePath) {
    const content = await fs.promises.readFile(filePath, 'utf-8');
    return content
        .split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'));  // Remove empty lines and comments
}
```

**Parsing rules:**
- Splits on newlines
- Trims whitespace
- Filters out empty lines
- Filters out comment lines (starting with `#`)

### Global Ignore File (loadGlobalIgnoreFile, lines 859-869)

Looks for `~/.context/.contextignore`:

```javascript
async loadGlobalIgnoreFile() {
    const homeDir = require('os').homedir();
    const globalIgnorePath = path.join(homeDir, '.context', '.contextignore');
    return await this.loadIgnoreFile(globalIgnorePath, 'global .contextignore');
}
```

### Pattern Merging

Patterns are merged using `addCustomIgnorePatterns()` (lines 471-480):
- Combines current patterns with new patterns
- Uses `Set` to deduplicate
- Preserves order (existing patterns first)

## 5. Extension Customization

### Environment Variables

**CUSTOM_EXTENSIONS** (lines 959-976):
```javascript
getCustomExtensionsFromEnv() {
    const envExtensions = envManager.get('CUSTOM_EXTENSIONS');
    if (!envExtensions) return [];

    return envExtensions
        .split(',')
        .map(ext => ext.trim())
        .filter(ext => ext.length > 0)
        .map(ext => ext.startsWith('.') ? ext : `.${ext}`);  // Ensure dot prefix
}
```

**CUSTOM_IGNORE_PATTERNS** (lines 982-998):
```javascript
getCustomIgnorePatternsFromEnv() {
    const envIgnorePatterns = envManager.get('CUSTOM_IGNORE_PATTERNS');
    if (!envIgnorePatterns) return [];

    return envIgnorePatterns
        .split(',')
        .map(pattern => pattern.trim())
        .filter(pattern => pattern.length > 0);
}
```

### Constructor Pattern Merging (lines 100-141)

The `Context` constructor combines patterns from multiple sources:

```javascript
// Extensions: DEFAULT + config.supportedExtensions + config.customExtensions + env
const allSupportedExtensions = [
    ...DEFAULT_SUPPORTED_EXTENSIONS,
    ...(config.supportedExtensions || []),
    ...(config.customExtensions || []),
    ...envCustomExtensions
];
this.supportedExtensions = [...new Set(allSupportedExtensions)];  // Dedupe

// Ignore patterns: DEFAULT + config.ignorePatterns + config.customIgnorePatterns + env
const allIgnorePatterns = [
    ...DEFAULT_IGNORE_PATTERNS,
    ...(config.ignorePatterns || []),
    ...(config.customIgnorePatterns || []),
    ...envCustomIgnorePatterns
];
this.ignorePatterns = [...new Set(allIgnorePatterns)];  // Dedupe
```

## 6. Summary

### Pattern Priority (lowest to highest)

1. `DEFAULT_IGNORE_PATTERNS` (built-in)
2. `config.ignorePatterns` (constructor config)
3. `config.customIgnorePatterns` (constructor config)
4. `CUSTOM_IGNORE_PATTERNS` env var
5. `.gitignore` and other `.xxxignore` files (runtime loaded)
6. `~/.context/.contextignore` (global user config)
7. `addCustomIgnorePatterns()` calls (MCP or programmatic)

### Limitations

1. **Shallow ignore file scanning**: Only scans codebase root, not subdirectories
2. **Simple glob matching**: Only supports `*` wildcard, not `**` for recursive matching, `?`, `[]`, or negation patterns
3. **No .gitignore nesting**: Doesn't respect nested `.gitignore` files in subdirectories
4. **No negation patterns**: Cannot use `!pattern` to un-ignore files
5. **Path separator handling**: Converts to forward slashes internally, may have edge cases on Windows

### Key Configuration Points

| Source | Method | When Applied |
|--------|--------|--------------|
| Defaults | Constructor | Initialization |
| Config object | Constructor | Initialization |
| Environment vars | Constructor | Initialization |
| Ignore files | `loadIgnorePatterns()` | Before indexing (`indexCodebase`) |
| MCP/programmatic | `addCustomIgnorePatterns()` | Anytime |
