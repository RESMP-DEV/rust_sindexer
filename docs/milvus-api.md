# Milvus v2 REST API Reference

This document covers the Milvus v2 REST API endpoints for vector database operations.

## Authentication

All endpoints require authentication via Bearer token in the Authorization header:

```
Authorization: Bearer <TOKEN>
Content-Type: application/json
```

## Base URL

```
http://localhost:19530
```

---

## POST /v2/vectordb/collections/create

Creates a new collection with the specified schema and index parameters.

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `collectionName` | string | Yes | Name of the collection to create |
| `schema` | object | Yes | Schema definition for the collection |
| `indexParams` | array | No | Index parameters for vector fields |

#### Schema Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `autoID` | boolean | No | If true, Milvus auto-generates entity IDs |
| `fields` | array | Yes | Array of field definitions |

#### Field Definition

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Field name |
| `dataType` | string | Yes | Data type: `Int64`, `VarChar`, `FloatVector`, `Float`, `Bool`, etc. |
| `isPrimaryKey` | boolean | No | Whether this field is the primary key |
| `isPartitionKey` | boolean | No | Whether this field is a partition key |
| `defaultValue` | any | No | Default value for the field |
| `elementTypeParams` | object | No | Type-specific parameters (e.g., `dim` for vectors, `max_length` for VarChar) |

#### Index Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fieldName` | string | Yes | Name of the vector field to index |
| `metricType` | string | Yes | Metric type: `L2`, `IP` (Inner Product), `COSINE` |
| `indexType` | string | Yes | Index type: `AUTOINDEX`, `HNSW`, `IVF_FLAT`, `IVF_SQ8`, `IVF_PQ` |

### Request Example

```json
{
    "collectionName": "my_collection",
    "schema": {
        "autoID": false,
        "fields": [
            {
                "name": "id",
                "dataType": "Int64",
                "isPrimaryKey": true
            },
            {
                "name": "vector",
                "dataType": "FloatVector",
                "elementTypeParams": {"dim": 128}
            },
            {
                "name": "title",
                "dataType": "VarChar",
                "elementTypeParams": {"max_length": 256}
            }
        ]
    },
    "indexParams": [
        {
            "fieldName": "vector",
            "metricType": "L2",
            "indexType": "AUTOINDEX"
        }
    ]
}
```

### Response

#### Success (200)

```json
{
    "code": 0,
    "data": {},
    "message": "success"
}
```

#### Error Response

```json
{
    "code": 65535,
    "message": "collection already exists"
}
```

### cURL Example

```bash
curl --request POST \
  --url "${CLUSTER_ENDPOINT}/v2/vectordb/collections/create" \
  --header "Authorization: Bearer ${TOKEN}" \
  --header "Content-Type: application/json" \
  -d '{
    "collectionName": "my_collection",
    "schema": {
        "autoID": false,
        "fields": [
            {"name": "id", "dataType": "Int64", "isPrimaryKey": true},
            {"name": "vector", "dataType": "FloatVector", "elementTypeParams": {"dim": 128}}
        ]
    },
    "indexParams": [
        {"fieldName": "vector", "metricType": "L2", "indexType": "AUTOINDEX"}
    ]
}'
```

---

## POST /v2/vectordb/entities/insert

Inserts entities into a collection.

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `collectionName` | string | Yes | Target collection name |
| `partitionName` | string | No | Target partition name (uses default if not specified) |
| `data` | array | Yes | Array of entity objects to insert |

Each entity object in `data` must include:
- The primary key field (unless `autoID` is enabled)
- The vector field
- Any additional scalar fields defined in the schema

### Request Example

```json
{
    "collectionName": "my_collection",
    "partitionName": "partitionA",
    "data": [
        {
            "id": 1,
            "vector": [0.358, -0.602, 0.184, -0.263, 0.903],
            "title": "Document A"
        },
        {
            "id": 2,
            "vector": [0.199, 0.060, 0.698, 0.261, 0.839],
            "title": "Document B"
        }
    ]
}
```

### Response

#### Success (200)

```json
{
    "code": 0,
    "data": {
        "insertCount": 2,
        "insertIds": [1, 2]
    }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `code` | integer | Status code (0 = success) |
| `data.insertCount` | integer | Number of entities inserted |
| `data.insertIds` | array | IDs of inserted entities |

### cURL Example

```bash
curl --request POST \
  --url "${CLUSTER_ENDPOINT}/v2/vectordb/entities/insert" \
  --header "Authorization: Bearer ${TOKEN}" \
  --header "Content-Type: application/json" \
  -d '{
    "collectionName": "my_collection",
    "data": [
        {"id": 1, "vector": [0.358, -0.602, 0.184, -0.263, 0.903], "title": "Doc A"},
        {"id": 2, "vector": [0.199, 0.060, 0.698, 0.261, 0.839], "title": "Doc B"}
    ]
}'
```

---

## POST /v2/vectordb/entities/search

Performs vector similarity search on a collection.

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `collectionName` | string | Yes | Collection to search |
| `data` | array | Yes | Query vectors (array of arrays of floats) |
| `annsField` | string | Yes | Name of the vector field to search |
| `limit` | integer | Yes | Maximum number of results per query vector |
| `filter` | string | No | Boolean expression for filtering (e.g., `"age > 18"`) |
| `outputFields` | array | No | Fields to include in results |
| `searchParams` | object | No | Index-specific search parameters |
| `partitionNames` | array | No | Partitions to search |
| `groupingField` | string | No | Field to group results by |

#### Search Parameters

| Parameter | Index Type | Description |
|-----------|------------|-------------|
| `nprobe` | IVF_FLAT, IVF_SQ8, IVF_PQ | Number of cluster units to search. Must be < `nlist` |
| `ef` | HNSW | Search scope. Range: [top_k, 32768] |
| `search_k` | ANNOY | Search scope. Must be >= top_k |

### Request Example

```json
{
    "collectionName": "my_collection",
    "data": [
        [0.358, -0.602, 0.184, -0.263, 0.903],
        [0.199, 0.060, 0.698, 0.261, 0.839]
    ],
    "annsField": "vector",
    "limit": 10,
    "filter": "age >= 18",
    "outputFields": ["id", "title", "score"],
    "searchParams": {
        "params": {
            "nprobe": 16
        }
    }
}
```

### Response

#### Success (200)

```json
{
    "code": 0,
    "data": [
        [
            {"id": 551, "distance": 0.088, "title": "Result 1"},
            {"id": 296, "distance": 0.080, "title": "Result 2"},
            {"id": 43, "distance": 0.078, "title": "Result 3"}
        ],
        [
            {"id": 102, "distance": 0.092, "title": "Result A"},
            {"id": 445, "distance": 0.085, "title": "Result B"}
        ]
    ]
}
```

The response contains an array of result arrays, one per query vector. Each result includes:

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer/string | Primary key of the matching entity |
| `distance` | float | Distance/similarity score (lower is more similar for L2) |
| Additional fields as specified in `outputFields` |

### cURL Example

```bash
curl --request POST \
  --url "${CLUSTER_ENDPOINT}/v2/vectordb/entities/search" \
  --header "Authorization: Bearer ${TOKEN}" \
  --header "Content-Type: application/json" \
  -d '{
    "collectionName": "my_collection",
    "data": [
        [0.358, -0.602, 0.184, -0.263, 0.903]
    ],
    "annsField": "vector",
    "limit": 5,
    "outputFields": ["id", "title"]
}'
```

---

## POST /v2/vectordb/entities/delete

Deletes entities from a collection by primary key IDs or filter expression.

### Request Body

You can delete entities using either `ids` or `filter`, but not both.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `collectionName` | string | Yes | Collection to delete from |
| `ids` | array | Conditional | Array of primary key values to delete |
| `filter` | string | Conditional | Boolean expression to match entities for deletion |
| `partitionName` | string | No | Partition to delete from |

### Request Example (Delete by IDs)

```json
{
    "collectionName": "my_collection",
    "ids": [1, 2, 3, 4, 5]
}
```

### Request Example (Delete by Filter)

```json
{
    "collectionName": "my_collection",
    "filter": "status == 'inactive' and age < 18"
}
```

### Response

#### Success (200)

```json
{
    "code": 0,
    "cost": 0,
    "data": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `code` | integer | Status code (0 = success) |
| `cost` | integer | Operation cost |
| `data` | object | Empty on success |

### cURL Example

```bash
# Delete by IDs
curl --request POST \
  --url "${CLUSTER_ENDPOINT}/v2/vectordb/entities/delete" \
  --header "Authorization: Bearer ${TOKEN}" \
  --header "Content-Type: application/json" \
  -d '{
    "collectionName": "my_collection",
    "ids": [18, 19, 20]
}'

# Delete by filter
curl --request POST \
  --url "${CLUSTER_ENDPOINT}/v2/vectordb/entities/delete" \
  --header "Authorization: Bearer ${TOKEN}" \
  --header "Content-Type: application/json" \
  -d '{
    "collectionName": "my_collection",
    "filter": "id in [18, 19, 20]"
}'
```

---

## POST /v2/vectordb/collections/describe

Retrieves the schema and details of a collection.

> Note: Despite the endpoint name suggesting GET semantics, this is a POST request.

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `collectionName` | string | Yes | Name of the collection to describe |

### Request Example

```json
{
    "collectionName": "my_collection"
}
```

### Response

#### Success (200)

```json
{
    "code": 0,
    "data": {
        "collectionName": "my_collection",
        "description": "",
        "autoId": false,
        "numShards": 1,
        "fields": [
            {
                "fieldId": 100,
                "name": "id",
                "description": "",
                "type": "Int64",
                "params": {},
                "isPrimary": true
            },
            {
                "fieldId": 101,
                "name": "vector",
                "description": "",
                "type": "FloatVector",
                "params": {"dim": 128}
            }
        ],
        "indexes": [
            {
                "fieldName": "vector",
                "indexName": "vector_idx",
                "metricType": "L2"
            }
        ],
        "aliases": [],
        "collectionId": "456909630285026300",
        "consistencyLevel": "Bounded",
        "numPartitions": 1,
        "enableDynamicField": true,
        "load": "LoadStateLoaded"
    }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `collectionName` | string | Name of the collection |
| `description` | string | Collection description |
| `autoId` | boolean | Whether auto-ID generation is enabled |
| `numShards` | integer | Number of shards |
| `fields` | array | Field definitions |
| `fields[].fieldId` | integer | Unique field identifier |
| `fields[].name` | string | Field name |
| `fields[].type` | string | Data type |
| `fields[].params` | object | Type-specific parameters (e.g., `dim` for vectors) |
| `fields[].isPrimary` | boolean | Whether this is the primary key |
| `indexes` | array | Index definitions |
| `indexes[].fieldName` | string | Indexed field name |
| `indexes[].indexName` | string | Index name |
| `indexes[].metricType` | string | Metric type (L2, IP, COSINE) |
| `aliases` | array | Collection aliases |
| `collectionId` | string | Unique collection identifier |
| `consistencyLevel` | string | Consistency level |
| `numPartitions` | integer | Number of partitions |
| `enableDynamicField` | boolean | Whether dynamic fields are enabled |
| `load` | string | Load state (LoadStateLoaded, LoadStateNotLoad) |

### cURL Example

```bash
curl --request POST \
  --url "${CLUSTER_ENDPOINT}/v2/vectordb/collections/describe" \
  --header "Authorization: Bearer ${TOKEN}" \
  --header "Content-Type: application/json" \
  -d '{
    "collectionName": "my_collection"
}'
```

---

## Data Types Reference

| DataType | Description |
|----------|-------------|
| `Bool` | Boolean values |
| `Int8` | 8-bit integer |
| `Int16` | 16-bit integer |
| `Int32` | 32-bit integer |
| `Int64` | 64-bit integer |
| `Float` | 32-bit floating point |
| `Double` | 64-bit floating point |
| `VarChar` | Variable-length string (requires `max_length` in `elementTypeParams`) |
| `JSON` | JSON object |
| `Array` | Array type |
| `FloatVector` | Float vector (requires `dim` in `elementTypeParams`) |
| `BinaryVector` | Binary vector |
| `Float16Vector` | 16-bit float vector |
| `BFloat16Vector` | Brain float 16 vector |
| `SparseFloatVector` | Sparse float vector |

## Metric Types Reference

| Metric | Description | Use Case |
|--------|-------------|----------|
| `L2` | Euclidean distance | General-purpose, normalized vectors |
| `IP` | Inner product | Recommendation, when magnitude matters |
| `COSINE` | Cosine similarity | Text embeddings, normalized comparison |

## Index Types Reference

| Index | Description | Best For |
|-------|-------------|----------|
| `AUTOINDEX` | Automatic index selection | Default choice |
| `FLAT` | Brute-force search | Small datasets, 100% recall |
| `IVF_FLAT` | Inverted file with flat quantization | Medium datasets |
| `IVF_SQ8` | IVF with scalar quantization | Memory efficiency |
| `IVF_PQ` | IVF with product quantization | Large datasets, memory constrained |
| `HNSW` | Hierarchical navigable small world | High performance, high recall |

## Error Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Unexpected error |
| 2 | Connect failed |
| 65535 | Collection/entity related errors |

---

## Filter Expression Syntax

Filter expressions use a SQL-like syntax:

### Comparison Operators
- `==`, `!=`: Equality
- `<`, `<=`, `>`, `>=`: Comparison
- `in`: Set membership
- `like`: Pattern matching

### Logical Operators
- `and`, `or`, `not`

### Examples

```python
# Simple comparison
"age >= 18"

# Set membership
"id in [1, 2, 3, 4, 5]"

# String matching
"title like 'AI%'"

# Compound expression
"(age >= 18 and status == 'active') or role == 'admin'"

# JSON field access
"metadata['category'] == 'technology'"
```
