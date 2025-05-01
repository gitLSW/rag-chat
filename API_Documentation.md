### Endpoint Schema Documentation

All endpoints require a Bearer Token authentication and most require an additional API key (except `/chat` and `/search`). The system automatically validates document metadata against the defined schemas and enforces access controls based on user roles.

#### 1. `POST /addDocumentSchema`
- **Purpose**: Defines a new JSON schema for a specific document type.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "docType": {
      "type": "string",
      "description": "The type identifier for documents using this schema"
    },
    "docSchema": {
      "type": "object",
      "description": "A valid JSON Schema that will validate document metadata of this type",
      "additionalProperties": true
    }
  },
  "required": ["docType", "docSchema"]
}
```

#### 2. `POST /addDocument`
- **Purpose**: Uploads a document file along with its metadata that must conform to a predefined schema.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "file": {
      "type": "string",
      "format": "binary",
      "description": "The document file to upload"
    },
    "doc_data": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the document"
        },
        "path": {
          "type": "string",
          "description": "Optional path for organizing documents",
          "nullable": true
        },
        "docType": {
          "type": "string",
          "description": "Document type that matches a predefined schema",
          "nullable": true
        },
        "accessGroups": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of groups with access to this document"
        }
      },
      "required": ["id", "accessGroups"],
      "additionalProperties": {
        "description": "Additional properties must match the schema defined for the docType"
      }
    }
  },
  "required": ["file", "doc_data"]
}
```

#### 3. `POST /updateDocument`
- **Purpose**: Updates an existing document's metadata (uses same schema as `/addDocument`'s doc_data).
- **Request Schema**: Same as `/addDocument`'s `doc_data` property.

#### 4. `POST /deleteDocument`
- **Purpose**: Deletes a document by its ID.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "The ID of the document to delete"
    }
  },
  "required": ["id"]
}
```

#### 5. `POST /search`
- **Purpose**: Performs semantic search across documents.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "The search query/question"
    },
    "searchDepth": {
      "type": "integer",
      "description": "Number of results to return. The actual returned number of found Documents may vary.",
      "default": 10
    }
  },
  "required": ["question"]
}
```

#### 6. `WebSocket /chat`
- **Purpose**: Streams LLM responses to queries in real-time.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "The question to ask the LLM"
    },
    "searchDepth": {
      "type": "integer",
      "description": "Depth for document retrieval",
      "default": 10
    }
  },
  "required": ["question"]
}
```