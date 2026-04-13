# AST Code Splitter Analysis

Analysis of `@zilliz/claude-context-core` AST-based code splitting implementation.

**Source:** `dist/splitter/ast-splitter.js`

## Overview

The `AstCodeSplitter` class uses Tree-sitter for language-aware code splitting. It parses source code into an AST, extracts logical code units (functions, classes, etc.), and splits them into chunks suitable for embedding or LLM context windows.

**Default Configuration:**
- `chunkSize`: 2500 characters
- `chunkOverlap`: 300 characters

**Fallback:** When a language is unsupported or parsing fails, the splitter falls back to `LangChainCodeSplitter` for character-based splitting.

---

## SPLITTABLE_NODE_TYPES by Language

The splitter identifies these AST node types as logical boundaries for chunking:

### JavaScript
```
function_declaration
arrow_function
class_declaration
method_definition
export_statement
```

### TypeScript
```
function_declaration
arrow_function
class_declaration
method_definition
export_statement
interface_declaration
type_alias_declaration
```

### Python
```
function_definition
class_definition
decorated_definition
async_function_definition
```

### Java
```
method_declaration
class_declaration
interface_declaration
constructor_declaration
```

### C++
```
function_definition
class_specifier
namespace_definition
declaration
```

### Go
```
function_declaration
method_declaration
type_declaration
var_declaration
const_declaration
```

### Rust
```
function_item
impl_item
struct_item
enum_item
trait_item
mod_item
```

### C# (additional)
```
method_declaration
class_declaration
interface_declaration
struct_declaration
enum_declaration
```

### Scala (additional)
```
method_declaration
class_declaration
interface_declaration
constructor_declaration
```

### Language Aliases

The splitter maps these aliases to their canonical parsers:
- `js` → JavaScript
- `ts` → TypeScript
- `py` → Python
- `c++`, `c` → C++
- `rs` → Rust
- `cs` → C#

---

## extractChunks() Algorithm

**Location:** Lines 98-138

### Purpose
Traverse the AST and extract content from nodes matching `SPLITTABLE_NODE_TYPES` for the given language.

### Algorithm

```
extractChunks(rootNode, code, splittableTypes, language, filePath):
    chunks = []

    traverse(currentNode):
        // Check if node type is splittable
        if currentNode.type IN splittableTypes:
            startLine = currentNode.startPosition.row + 1  // 1-indexed
            endLine = currentNode.endPosition.row + 1
            nodeText = code[currentNode.startIndex : currentNode.endIndex]

            // Only include non-empty content
            if nodeText.trim().length > 0:
                chunks.push({
                    content: nodeText,
                    metadata: { startLine, endLine, language, filePath }
                })

        // Recursively traverse children
        for child in currentNode.children:
            traverse(child)

    traverse(rootNode)

    // Fallback: if no chunks found, use entire file as single chunk
    if chunks.length == 0:
        chunks.push({
            content: code,
            metadata: { startLine: 1, endLine: lineCount, language, filePath }
        })

    return chunks
```

### Key Characteristics

1. **Depth-first traversal:** Visits all nodes recursively, extracting splittable types at any depth.

2. **Nested extraction:** If a class contains methods, both the class and each method may be extracted as separate chunks (since both types are in `splittableTypes`). This creates potential duplication.

3. **Position tracking:** Uses Tree-sitter's `startPosition.row` and `startIndex`/`endIndex` for accurate source location mapping.

4. **Graceful fallback:** If no splittable nodes are found (e.g., a file with only imports or comments), the entire file becomes a single chunk.

---

## refineChunks() Logic

**Location:** Lines 140-152

### Purpose
Ensure all chunks fit within `chunkSize`. Large AST nodes (e.g., a 5000-character function) are split further.

### Algorithm

```
refineChunks(chunks, originalCode):
    refinedChunks = []

    for chunk in chunks:
        if chunk.content.length <= chunkSize:
            // Chunk fits, keep as-is
            refinedChunks.push(chunk)
        else:
            // Split large chunk into smaller pieces
            subChunks = splitLargeChunk(chunk, originalCode)
            refinedChunks.push(...subChunks)

    // Apply overlap to all refined chunks
    return addOverlap(refinedChunks)
```

### splitLargeChunk() Sub-algorithm

**Location:** Lines 154-195

Splits a large chunk using line-based character accumulation:

```
splitLargeChunk(chunk, originalCode):
    lines = chunk.content.split('\n')
    subChunks = []
    currentChunk = ''
    currentStartLine = chunk.metadata.startLine
    currentLineCount = 0

    for i, line in enumerate(lines):
        lineWithNewline = line + '\n' (except last line)

        // Check if adding this line would exceed chunkSize
        if currentChunk.length + lineWithNewline.length > chunkSize AND currentChunk.length > 0:
            // Emit current sub-chunk
            subChunks.push({
                content: currentChunk.trim(),
                metadata: {
                    startLine: currentStartLine,
                    endLine: currentStartLine + currentLineCount - 1,
                    language, filePath
                }
            })
            // Start new sub-chunk with current line
            currentChunk = lineWithNewline
            currentStartLine = chunk.metadata.startLine + i
            currentLineCount = 1
        else:
            currentChunk += lineWithNewline
            currentLineCount++

    // Emit final sub-chunk
    if currentChunk.trim().length > 0:
        subChunks.push(...)

    return subChunks
```

### Characteristics

1. **Line-preserving:** Never splits mid-line. If a single line exceeds `chunkSize`, it becomes its own chunk.

2. **Greedy accumulation:** Adds lines until the next line would exceed the limit, then starts a new chunk.

3. **Metadata continuity:** Tracks line numbers relative to the original file, not the chunk.

---

## Chunk Overlap Mechanics

**Location:** Lines 197-218

### Purpose
Add redundant content between adjacent chunks to preserve context across chunk boundaries. This helps embedding models and LLMs understand code that spans multiple chunks.

### Algorithm

```
addOverlap(chunks):
    if chunks.length <= 1 OR chunkOverlap <= 0:
        return chunks  // No overlap needed

    overlappedChunks = []

    for i, chunk in enumerate(chunks):
        content = chunk.content
        metadata = copy(chunk.metadata)

        // Prepend overlap from previous chunk
        if i > 0 AND chunkOverlap > 0:
            prevChunk = chunks[i - 1]
            overlapText = prevChunk.content[-chunkOverlap:]  // Last N characters
            content = overlapText + '\n' + content
            metadata.startLine = max(1, metadata.startLine - getLineCount(overlapText))

        overlappedChunks.push({ content, metadata })

    return overlappedChunks
```

### Key Characteristics

1. **Character-based overlap:** Takes the last `chunkOverlap` characters (default 300) from the previous chunk, not a line count.

2. **Forward-only:** Each chunk receives overlap from the previous chunk. The first chunk has no overlap prepended.

3. **Newline separator:** Overlap text is joined to the chunk with a newline, ensuring the prepended context is on separate lines.

4. **Line number adjustment:** `startLine` is adjusted backwards based on how many lines the overlap text contains.

### Overlap Flow Example

Given chunks A, B, C with `chunkOverlap = 300`:

```
Original:
  Chunk A: "function foo() { ... }"  (lines 1-20)
  Chunk B: "function bar() { ... }"  (lines 21-40)
  Chunk C: "function baz() { ... }"  (lines 41-60)

After addOverlap:
  Chunk A: unchanged (no previous chunk)
  Chunk B: "[last 300 chars of A]\nfunction bar() { ... }"
           startLine adjusted from 21 to ~18 (depending on line count)
  Chunk C: "[last 300 chars of B]\nfunction baz() { ... }"
           startLine adjusted similarly
```

---

## Processing Pipeline

```
split(code, language, filePath)
    │
    ├─ getLanguageConfig(language)
    │      └─ Returns { parser, nodeTypes } or null
    │
    ├─ [If unsupported] → LangChainCodeSplitter.split()
    │
    ├─ parser.parse(code) → AST tree
    │
    ├─ extractChunks(tree.rootNode, ...)
    │      └─ Traverse AST, extract splittable nodes
    │
    └─ refineChunks(chunks, code)
           ├─ splitLargeChunk() for oversized chunks
           └─ addOverlap() for context continuity
```

---

## Potential Issues and Considerations

### 1. Nested Chunk Duplication
If a class and its methods are both splittable types, the class body (including methods) is extracted as one chunk, and each method is also extracted separately. This creates redundant content.

### 2. Overlap May Exceed Chunk Size
After adding overlap, a chunk's total size may exceed `chunkSize`. The overlap is added after refinement, so a chunk near the size limit becomes oversized.

### 3. No Backward Overlap
Only forward context (previous chunk into current) is preserved. If understanding a chunk requires context from the next chunk, that connection is lost.

### 4. Character vs Semantic Boundaries
The overlap takes exactly N characters, which may cut mid-word or mid-statement. A semantic-aware overlap (ending at statement boundaries) would be cleaner.

### 5. Single-Line Chunks
If a single line exceeds `chunkSize`, the line-based splitting cannot break it further. The chunk will exceed the configured size.

---

## Configuration Recommendations

| Use Case | chunkSize | chunkOverlap |
|----------|-----------|--------------|
| Embedding (default) | 2500 | 300 |
| LLM context window | 4000-8000 | 500 |
| Fine-grained search | 1000 | 200 |
| Minimal overlap | 2500 | 0 |
