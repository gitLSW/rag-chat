### API Documentation

All endpoints require a Bearer Token authentication and most require an additional API key (except `/chat`). The system automatically validates document metadata against the defined schemas and enforces access controls based on user roles.

#### 1. `POST /createAccessGroup`
- **Purpose**: Creates a new access group for document permissions.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "access_group": {
      "type": "string",
      "description": "The name/identifier of the access group to create"
    }
  },
  "required": ["access_group"]
}
```
- **Response Schema**: Success/Failure Response

#### 2. `POST /addDocumentSchema`
- **Purpose**: Defines a new JSON schema for a specific document type. Cannot override existing Schemata.
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
- **Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "detail": {
      "type": "string",
      "description": "Success message"
    },
    "data": {
      "type": "object",
      "properties": {
          "id": {"type": "string"},
          "path": {"type": "string"},
          "docType": {"type": "string"},
          "accessGroups": {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 1
          }
      },
      "required": ["id", "path", "docType", "accessGroups"],
      "description": "The added JSON schema, combined with the server's required base document schema",
      "additionalProperties": true
    }
  }
}
```

#### 3. `POST /deleteDocumentSchema`
- **Purpose**: Deletes an existing document schema, if it is unused by all documents.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "doc_type": {
      "type": "string",
      "description": "The type identifier for documents using this schema"
    }
  },
  "required": ["doc_type"]
}
```
- **Response Schema**: Success/Failure Response

#### 4. `POST /createDocument`
- **Purpose**: Create or fully override a document file along with its metadata that must conform to a predefined schema.
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
          "items": {"type": "string"},
          "minItems": 1,
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
- **Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "detail": {
      "type": "string",
      "description": "Success message"
    },
    "data": {
      "type": "object",
      "properties": {
          "id": {"type": "string"},
          "path": {"type": "string"},
          "docType": {"type": "string"},
          "accessGroups": {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 1
          }
      },
      "required": ["id", "path", "docType", "accessGroups"],
      "description": "The processed document data including extracted metadata",
      "additionalProperties": true
    }
  }
}
```

#### 5. `POST /updateDocument`
- **Purpose**: Override an existing document's metadata.
- **Request Schema**: Same as `/createDocument`'s `doc_data` property.
```json
{
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
      "items": {"type": "string"},
      "minItems": 1,
      "description": "List of groups with access to this document"
    }
  },
  "required": ["id", "accessGroups"],
  "additionalProperties": {
    "description": "Additional properties must match the schema defined for the docType"
  }
}
```
- **Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "detail": {
      "type": "string",
      "description": "Success message"
    },
    "data": {
      "type": "object",
      "properties": {
          "id": {"type": "string"},
          "path": {"type": "string"},
          "docType": {"type": "string"},
          "accessGroups": {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 1
          }
      },
      "required": ["id", "path", "docType", "accessGroups"],
      "description": "The updated document data",
      "additionalProperties": true
    }
  }
}
```

#### 6. `POST /deleteDocument`
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
- **Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "detail": {
      "type": "string",
      "description": "Success message"
    },
    "data": {
      "type": "object",
      "properties": {
          "id": {"type": "string"},
          "path": {"type": "string"},
          "docType": {"type": "string"},
          "accessGroups": {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 1
          }
      },
      "required": ["id", "path", "docType", "accessGroups"],
      "description": "The deleted document data",
      "additionalProperties": true
    }
  }
}
```

#### 7. `POST /getDocumentData`
- **Purpose**: Retrieves the extracted data and text content of an existing document.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "The ID of the document to read"
    }
  },
  "required": ["id"]
}
```
- **Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "doc_id": {
      "type": "string",
      "description": "The ID of the document to read"
    },
    "text": {
      "type": "string",
      "description": "The extracted text content of the document"
    },
    "data": {
      "type": "object",
      "properties": {
          "id": {"type": "string"},
          "path": {"type": "string"},
          "docType": {"type": "string"},
          "accessGroups": {
              "type": "array",
              "items": {"type": "string"},
              "minItems": 1
          }
      },
      "required": ["id", "path", "docType", "accessGroups"],
      "description": "The processed document data including extracted metadata",
      "additionalProperties": true
    }
  },
  "required": ["doc_id", "text", "data"]
}
```

#### 8. `POST /search`
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
- **Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "detail": {
      "type": "string",
      "description": "Number of found documents"
    },
    "data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "docId": {"type": "string"},
          "pageNum": {
            "type": "integer",
            "description": "Page number where match was found",
            "nullable": true
          },
          "docType": {
            "type": "string",
            "description": "Document's JSON schema type"
          }
        }
      }
    }
  }
}
```

#### 9. `WebSocket /chat`
- **Purpose**: Opens a web socket connection, which streams the LLM response as chunks in real-time.
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
- **Response Schema**: It uses a web socket connection.