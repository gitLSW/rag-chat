# API Documentation

## Authentication
- **API Key**: `x-api-key API_KEY`\
The `/users` and `/documents` endpoints require an API key in each request header and the request source must prevously be whitelisted.
- **Bearer Token**: `Bearer BEARER_TOKEN`\
All endpoints require a RSA256 encrypted Bearer Token (in .pem file format) authentication in every request header.
The Bearer Token must be generated with `jsonwebtoken` and its payload must encode:
```json
{
  "companyId": USER_COMPANY_IDENTIFIER,
  "userId": USER_IDENTIFIER
}
```
A superuser access is provided by the system when the token's user_id is "superuser".
The public key will be retreived automatically from the Auth server's URL, which needs to be specified in the .env file.



## Endpoints

### Users
Access groups for each company must be used by at least one user previous to using them.
Every document has a list of accessGroups associated with it.
Administrator roles are always added automatically by the system.
They are named `admin` in the System.
A superuser access is provided by the system when the token's user_id is "superuser".
This superuser can be used to create the first admin user for a new company.

#### `POST /users`
- **Purpose**: Creates or overwrites a user with access groups for document permissions.
- **Access Control**: Requires `admin` role
- **Request Schema**:
```json
{
    "type": "object",
    "properties": {
        "id": { "type": "string" },
        "accessRoles": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1,
            "description": "The user's access groups permitting him to view or modify documents within his groups.",
        }
    },
    "required": ["id", "accessRoles"],
    "additionalProperties": false
}
```
- **Response Schema**: Success/Failure Response

#### `DELETE /users/{user_id}`
- **Purpose**: Deletes an existing user.
- **Access Control**: Requires `admin` role
- **Request Schema**: `None`
- **Response Schema**: Success/Failure Response


### Document Schemata
The system automatically validates extracted and uploded document metadata against the previously defined JSON schemata.
If too many schemata are registered (= too much text for the LLM), no new ones will be able to be created.

#### `POST /documentSchemata`
- **Purpose**: Defines a new JSON schema for a specific document type. Cannot override existing Schemata.
- **Access Control**: Requires `admin` role
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

#### `GET /documentSchemata`
- **Purpose**: Retrieves all document types and their corresponding JSON schema.
- **Request Schema**: `None`
- **Response Schema**: Success/Failure response with a JSON dictionary containg the document types as keys and their JSON schema as values.
```json
{
    DOC_TYPE_1: {
      "type": "object",
      "description": "A valid JSON Schema that will validate document metadata of this type",
      "additionalProperties": true
    }
}
```

#### `DELETE /documentSchemata/{doc_type}`
- **Purpose**: Deletes an existing document schema, if it is unused by all documents.
- **Access Control**: Requires `admin` role
- **Request Schema**: `None`
- **Response Schema**: Success/Failure Response


### Document Processing & Storage
The system aims to provide a semantic search and metadata extraction from natural text documents as well as a LLM chat infused with relevant context data.
After uploading a file, the system extracts its text as a string via different extraction libraries or an OCR, if the file is a scanned PDF (page contains a image).
The extracted text gets persisted and split into paragraphs and the paragraphs get vectorized by the `BERT Sentence Transformer`.
These paragraph vector embeddings get saved alongside the exact page and documentId in a VectorDB (`ChromaDB`).
If no document metadata or type was provided the system will automatically try to find a document type by comparing the alignment of the document's emebdding vector to every previously defined JSON schema's embedding vector via the dot-product.
Only if they align above a certain threshold, the document will get assigned a type.
If a document type was provided or automatically determined, a document metadata extraction will be performed.
The LLM (provided locally by `vLLM`) will be queried to fill the associated JSON schema of the document's type.
The extracted JSON metadata will be merged with and overwritten by the uploaded document metadata.
The merged data gets validated against the document type's schema.
If it passed, it will be added to the document database (`MongoDB`).
Finally, the document's text content gets saved as a txt file.

#### `POST /documents`
- **Purpose**: Uploads and processes a document file along with its metadata that must conform to a predefined schema. If accessGroups is null or empty, everybody has document access. If the automatic JSON extract failed, the provides metadata, plain document text and paragraph embeddings will still be saved. Supported MIME file types: ```pdf, docx, odt, rtf, txt, csv, xml, json.```
- **Request Schema**:

`Form Table`:

| Field Name     | Type          | Description                                               | Required | Notes                         |
|----------------|---------------|-----------------------------------------------------------|----------|-------------------------------|
| file           | File (binary) | The document file to upload                               | ✅       | Must be a valid file          |
| forceOCR       | Boolean       | Use OCR to extract text only                              | ❌       | Only works with PDFs          |
| allowOverride  | Boolean       | Allow overriding an existing document                     | ❌       | Defaults to true              |
| docData        | JSON string   | Metadata object for the document                          | ✅       | Must be a stringified JSON    |

`docData`:
```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier for the document"
    },
    "path": {
      "type": ["string", "null"],
      "description": "Optional path for organizing documents"
    },
    "docType": {
      "type": ["string", "null"],
      "description": "Document type that matches a predefined schema"
    },
    "accessGroups": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "minItems": 1,
      "description": "List of user access role groups with access to this document"
    }
  },
  "required": ["id"],
  "additionalProperties": {
    "description": "Additional properties must match the schema defined for the docType"
  }
}
```
- **Response Schema**: Success/Failure response with the added document's metadata and extracted text
```json
{
  "type": "object",
  "properties": {
    "detail": {"type": "string"},
    "data": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "text": {"type": "string"},
        "path": {"type": "string"},
        "docType": {"type": ["string", "null"]},
        "accessGroups": {
          "type": ["array", null],
          "items": {"type": "string"},
          "minItems": 1
        }
      },
      "additionalProperties": true
    }
  }
}
```

#### `PUT /documents/{doc_id}`
- **Purpose**: Updates an existing document's metadata. The updated document data must conform to the JSON schema of its document type.
- **Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "mergeExisting": {
      "type": "boolean",
      "description": "Whether to merge with existing data. Fields will still overwrite fields of the same name on the old documents. Arrays of same named fields will not be merged.",
      "default": false
    },
    "docData": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Optional unique identifier for the document. If existant, it must match the doc_id in the URL."
        },
        "path": {
          "type": ["string", "null"],
          "description": "Optional path for organizing documents"
        },
        "docType": {
          "type": ["string", "null"],
          "description": "Document type that matches a predefined schema"
        },
        "accessGroups": {
          "type": ["array", "null"],
          "items": {"type": "string"},
          "minItems": 1,
          "description": "List of user access role groups with access to this document"
        }
      },
      "additionalProperties": {
        "description": "Additional properties must match the schema defined for the docType"
      }
    }
  },
  "required": ["docData"]
}
```
- **Response Schema**: Same as `POST /document` response

#### `GET /documents/{doc_id}`
- **Purpose**: Retrieves a document's metadata and text content.
- **Request Schema**: `None`
- **Response Schema**: Success/Failure response with the retrieved document's metadata
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
        "docType": {"type": ["string", "null"]},
        "text": {"type": "string"},
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

#### `DELETE /documents/{doc_id}`
- **Purpose**: Deletes a document by its ID.
- **Request Schema**: `None`
- **Response Schema**: Success/Failure Response with deleted document's metadata

### Semantic Search

After a user question was received, it gets vectorized into a sentence embedding and the VectorDB gets queried to find the nearest neighbour paragraph vector matches to the question vector in the embedding vector space.
The `searchDepth` parameter determines how many nearest neighbours shall be retrieved.
Now the system finds all the unique documents which the neighbours point

#### `GET /search`
- **Purpose**: Performs semantic search across documents.
- **Query Parameters**:
  - `question` (string, required): The search query/question.
  - `searchDepth` (integer, optional, default=10): Number of similar paragraphs to find.
- **Example Request URL**: `/search?question=your+search+query&searchDepth=30`
- **Response Schema**: Success/Failure response with array of found paragraps' document IDs, page numbers and schema types (can differ in length to the searchDepth)
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
          "docType": {"type": ["string", "null"]},
        }
      }
    }
  }
}
```

### LLM Chat

After a user started a chat websocket and asks a question, a semantic search is performed on it and the system reads the found documents concurrently.
Once read, the LLM will be asked to concurrently summarize the texts with respect to the users question.
As soon as all summaries have been provided, they get shown to the LLM alongside the question.
If the user enabled Database queries the LLM will also be shown all document schemata of the retrieved the documents' types.
Additionally, it is informed of its abaility to perform database searches by writing them in tags.
If the command was successful, the system will show the output to the LLM, which will now finally answer the question.

#### `WebSocket /chat`
- **Purpose**: Opens a WebSocket connection for managing and interacting with LLM chat sessions. Supports starting, sending messages, pausing, resuming, and deleting chats.
- **Request Schema**: It uses a predefinded actions, which are sent as JSON messages over the WebSocket.
```json
{
  "oneOf": [
    {
      "type": "object",
      "properties": {
        "action": { "const": "start" },
        "chat_id": { "type": "string" }
      },
      "required": ["action", "chat_id"]
    },
    {
      "type": "object",
      "properties": {
        "action": { "const": "message" },
        "chat_id": { "type": "string" },
        "message": { "type": "string" },
        "use_db": { "type": "boolean" },
        "rag_search_depth": { "type": "integer" },
        "show_chat_history": { "type": "boolean" },
        "resuming": { "type": "boolean" }
      },
      "required": ["action", "chat_id", "message"]
    },
    {
      "type": "object",
      "properties": {
        "action": { "enum": ["pause", "resume", "delete"] },
        "chat_id": { "type": "string" }
      },
      "required": ["action", "chat_id"]
    }
  ]
}
```
- **Response Schema**: Text messages streamed back to the client, depending on the action.