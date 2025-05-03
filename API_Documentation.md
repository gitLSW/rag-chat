### API Documentation

#### Header
All endpoints require a Bearer Token authentication and most require an additional API key (except `/chat`). The system automatically validates document metadata against the defined schemas and enforces access controls based on user roles.
- **Bearer Token**: `Bearer BEARER_TOKEN`
- **API Key**: `x-api-key API_KEY`


#### 1. `POST /createAccessGroup`
- **Purpose**: Creates a new access group for document permissions.
- **Access Control**: Requires admin role
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "accessGroup": {
      "type": "string",
      "description": "The name/identifier of the access group to create"
    }
  },
  "required": ["accessGroup"]
}
```
- **Response Schema**: Success/Failure Response

#### 2. `POST /addDocumentSchema`
- **Purpose**: Defines a new JSON schema for a specific document type. Cannot override existing Schemata.
- **Access Control**: Requires admin role
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
- **Response Schema**: Success/Failure Response with the added schema

#### 3. `POST /deleteDocumentSchema`
- **Purpose**: Deletes an existing document schema, if it is unused by all documents.
- **Access Control**: Requires admin role
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "docType": {
      "type": "string",
      "description": "The type identifier of the schema to delete"
    }
  },
  "required": ["docType"]
}
```
- **Response Schema**: Success/Failure Response

#### 4. `POST /createDocument`
- **Purpose**: Uploads and processes a document file along with its metadata that must conform to a predefined schema.
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
    "allowOverride": {
      "type": "boolean",
      "description": "Whether to allow overriding an existing document",
      "default": true
    },
    "docData": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the document"
        },
        "path": {
          "type": "string",
          "description": "Optional path for organizing documents"
        },
        "docType": {
          "type": "string",
          "description": "Document type that matches a predefined schema"
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
  "required": ["file", "docData"]
}
```
- **Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "detail": {"type": "string"},
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
      "additionalProperties": true
    }
  }
}
```

#### 5. `POST /updateDocument`
- **Purpose**: Updates an existing document's metadata.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "mergeExisting": {
      "type": "boolean",
      "description": "Whether to merge with existing data",
      "default": false
    },
    "docData": {
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
      "required": ["id"],
      "additionalProperties": true
    }
  },
  "required": ["docData"]
}
```
- **Response Schema**: Same as createDocument response

#### 6. `POST /getDocument`
- **Purpose**: Retrieves a document's metadata and text content.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "The ID of the document to retrieve"
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
    "detail": {"type": "string"},
    "data": {
      "type": "object",
      "properties": {
        "text": {"type": "string"},
        "id": {"type": "string"},
        "path": {"type": "string"},
        "docType": {"type": "string"},
        "accessGroups": {
          "type": "array",
          "items": {"type": "string"},
          "minItems": 1
        }
      },
      "additionalProperties": true
    }
  }
}
```

#### 7. `POST /deleteDocument`
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
- **Response Schema**: Success/Failure Response with deleted document data

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
      "description": "Number of results to return",
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
    "detail": {"type": "string"},
    "data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "docId": {"type": "string"},
          "pageNum": {"type": "integer"},
          "docType": {"type": "string"}
        }
      }
    }
  }
}
```

#### 9. `WebSocket /chat`
- **Purpose**: Opens a web socket connection for streaming LLM responses.
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
- **Response**: Streams text responses through the WebSocket connection