# RAGdoll User Guide: Agents, API Keys, and External Access

This guide explains how to use RAGdoll as an application user:

- Add LLM provider API keys.
- Create and configure an agent.
- Create agent access keys.
- Use an agent externally from another application without logging in.

The external access-key workflow is the most important integration path. It lets another system talk to an agent using an agent access key and a role name.

## Concepts

RAGdoll uses two different kinds of keys:

| Key type | Purpose | Where it is used |
| --- | --- | --- |
| LLM provider API key | Lets RAGdoll call a model provider such as OpenAI, Gemini, or Idun. | Stored in the config app under API keys, then selected when creating or editing an agent. |
| Agent access key | Lets an external system talk to one specific agent. | Sent to the backend when resolving agent info and asking questions. |

An LLM provider API key is for paying/calling the model provider. An agent access key is for using a configured RAGdoll agent externally.

## Add an LLM Provider API Key

1. Log in to the RAGdoll config application.
2. Open the API keys page.
3. Create a new API key entry.
4. Select the provider, for example `openai`.
5. Paste the provider API key.
6. Choose whether the key is for LLM, embeddings, or both.
7. Save it.

The provider key is stored by RAGdoll and can be selected when creating or editing agents. It is not the same as an agent access key.

## Create an Agent

1. Open the Agents page.
2. Create a new agent.
3. Fill in the agent name, description, prompt, model settings, and retrieval settings.
4. Select the LLM provider API key for the agent.
5. Select the embedding API key if the agent uses embeddings.
6. Add one or more roles.
7. Save the agent.

Roles are important for external use. An external request must specify the role it wants to talk to. The role controls which documents/context the conversation can use.

## Upload Documents

1. Open the agent.
2. Go to the documents area.
3. Upload the documents the agent should use.
4. Wait for processing to complete.
5. Make sure each role has access to the intended documents.

If the role does not have access to the relevant documents, the external chat request can still work, but the answer may not include the expected RAG context.

## Create an Agent Access Key

1. Open the agent in the config application.
2. Go to the Access Keys tab.
3. Click Generate Access Key.
4. Give the key a name.
5. Optionally set an expiry date.
6. Choose the key visibility:

| Visibility | Behavior |
| --- | --- |
| View once | The key is shown only when it is created. After that, the UI hides it. |
| View anytime | The key can be revealed and copied again from the access-key list. |

7. Create the key.
8. Copy the raw key value.

Important: for view-once keys, copy the raw key immediately. The key name and key id are not enough for external use.

## External Use

External applications do not need a logged-in RAGdoll user. The agent access key is the authorization credential.

The external application needs:

- The backend base URL.
- An agent access key.
- A role name.
- A chat history.
- A session id if the application wants progress/task state to be remembered during a run.

For the production server, the backend base URL is:

```text
https://iplvr.it.ntnu.no/backend
```

For local development, the backend base URL is usually:

```text
http://localhost:8000
```

### Step 1: Resolve Agent Info from an Access Key

Use this endpoint to find the agent connected to an access key and to list available roles.

```http
GET /agent-info-by-accesskey
```

Header:

```text
access-key: YOUR_AGENT_ACCESS_KEY
```

Example:

```bash
curl https://iplvr.it.ntnu.no/backend/agent-info-by-accesskey \
  -H "access-key: YOUR_AGENT_ACCESS_KEY"
```

Example response:

```json
{
  "agent_id": "6a1d614955f55909e1272f02",
  "name": "Example Agent",
  "roles": [
    {
      "name": "student",
      "description": "Answers questions for students",
      "document_access": ["document-id-1", "document-id-2"]
    }
  ]
}
```

Use `agent_id` and one of the role `name` values in the chat request.

If the request returns `401 Unauthorized`, check:

- You are using the full raw access key, not the key name or key id.
- The key has not expired.
- The key belongs to the environment you are calling. A key from the server database will not work against a local database.
- The request includes the `access-key` header exactly.

### Step 2: Ask the Agent a Question

Use this endpoint to send a message to the agent.

```http
POST /api/chat/ask
```

Body fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `agent_id` | string | Yes | The agent id returned by `/agent-info-by-accesskey`. |
| `active_role_id` | string | Yes | The role name to speak as, for example `student`. |
| `access_key` | string | Yes | The raw agent access key. |
| `chat_log` | array | Yes | The conversation history. Include the latest user message. |
| `user_information` | array of strings | No | Extra user or game-state facts to include in the prompt. |
| `user_actions` | array of strings | No | Recent actions from the external application or game. |
| `progress` | array | No | Structured task progress entries. |
| `session_id` | string | No | External session identifier. If supplied, the backend automatically adds recent stored progress for this session. |
| `include_progress` | boolean | No | Defaults to `true`. Set to `false` to prevent automatic session progress from being added. |
| `progress_limit` | number | No | Defaults to `5`. Maximum number of recent stored progress tasks to add. Maximum accepted value is `20`. |

Example:

```bash
curl https://iplvr.it.ntnu.no/backend/api/chat/ask \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "6a1d614955f55909e1272f02",
    "active_role_id": "student",
    "access_key": "YOUR_AGENT_ACCESS_KEY",
    "chat_log": [
      {
        "role": "user",
        "content": "What is this agent able to help me with?"
      }
    ]
  }'
```

Example response shape:

```json
{
  "response": {
    "response": "The agent response text is here.",
    "context_used": [
      {
        "document_name": "example.pdf",
        "category": "default",
        "chunk_index": 3,
        "content": "The retrieved context text..."
      }
    ]
  }
}
```

The returned `response.response` value is the assistant answer. `response.context_used` shows which document chunks were used, when context is available.

### Chat Log Format

The `chat_log` array should include previous turns plus the newest user message.

Example multi-turn request:

```json
{
  "agent_id": "6a1d614955f55909e1272f02",
  "active_role_id": "student",
  "access_key": "YOUR_AGENT_ACCESS_KEY",
  "chat_log": [
    {
      "role": "user",
      "content": "What is this document about?"
    },
    {
      "role": "agent",
      "content": "It explains the project setup."
    },
    {
      "role": "user",
      "content": "Summarize the most important setup steps."
    }
  ]
}
```

The final item should normally be the newest user question.

### Optional Live Context

External clients can send extra context with each chat request. This is useful for games, simulators, or training applications where the LLM should know what the user has done.

Example:

```json
{
  "agent_id": "6a1d614955f55909e1272f02",
  "active_role_id": "instructor",
  "access_key": "YOUR_AGENT_ACCESS_KEY",
  "session_id": "unity-session-001",
  "chat_log": [
    {
      "role": "user",
      "content": "What should I do next?"
    }
  ],
  "user_information": [
    "The player is in the engine room.",
    "The player has low health."
  ],
  "user_actions": [
    "Opened the toolbox",
    "Inspected the broken cable"
  ],
  "progress": [
    {
      "task_name": "Repair engine",
      "description": "Repair the ship engine",
      "status": "started",
      "subtask_progress": [
        {
          "subtask_name": "Find tool",
          "description": "Find the wrench",
          "completed": true,
          "step_progress": [
            {
              "step_name": "Open toolbox",
              "repetition_number": 0,
              "completed": true
            }
          ]
        }
      ]
    }
  ]
}
```

These fields are optional. Existing plain chat integrations can keep sending only `agent_id`, `active_role_id`, `access_key`, and `chat_log`.

If `session_id` is included, the backend loads the most recent stored progress tasks for that agent/session and adds them to the prompt automatically. The prompt tells the LLM to use this progress only when it is relevant to the user's question, current task, or next action.

## Voice Endpoints

RAGdoll includes Whisper-based transcription endpoints for external applications that send recorded audio.

### Transcribe Audio Only

```http
POST /api/chat/transcribe
```

Form fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `audio` | file | Yes | Audio file to transcribe. WAV is recommended. |
| `language` | string | No | Optional language code, for example `en` or `no`. |

Example:

```bash
curl https://iplvr.it.ntnu.no/backend/api/chat/transcribe \
  -F "audio=@question.wav" \
  -F "language=en"
```

Example response:

```json
{
  "success": true,
  "transcription": "What should I do next?",
  "server_processed": true,
  "processing_time_seconds": 1.2,
  "processor": "Server-based Whisper"
}
```

This endpoint does not talk to an agent. It only returns text.

### Transcribe Audio and Ask Agent

```http
POST /api/chat/askTranscribe
```

Form fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `audio` | file | Yes | Audio file containing the user's question. |
| `data` | JSON string | Yes | A serialized chat command with `agent_id`, `active_role_id`, `access_key`, and optional context fields. |

Example:

```bash
curl https://iplvr.it.ntnu.no/backend/api/chat/askTranscribe \
  -F "audio=@question.wav" \
  -F 'data={
    "agent_id": "6a1d614955f55909e1272f02",
    "active_role_id": "student",
    "access_key": "YOUR_AGENT_ACCESS_KEY",
    "chat_log": []
  }'
```

The backend transcribes the audio, appends the transcription as the latest user message, validates the access key, and sends the request through the same RAG pipeline as `/api/chat/ask`.

Example response shape:

```json
{
  "transcription": "What should I do next?",
  "response": {
    "response": "The agent response text is here.",
    "context_used": []
  }
}
```

## Progress Endpoints

Progress endpoints are intended for external training applications, games, or simulators that want to store task state and reuse it in later chat requests.

Progress is stored in memory in the backend process and is scoped by:

- `agent_id`
- `session_id`

Progress entries are pruned after roughly 24 hours and also disappear if the backend restarts. For durable game state, the external application should still keep its own source of truth.

Recommended external workflow:

1. Generate a `session_id` when the external run starts.
2. Store that `session_id` in the game/client memory.
3. Send the same `session_id` with progress updates.
4. Send the same `session_id` with `/api/chat/ask` or `/api/chat/askTranscribe`.
5. The backend automatically includes the most recent progress tasks in the LLM prompt.

Example session id:

```text
unity-session-2026-06-02-player-123
```

The session id does not need to be secret. The access key is the authorization credential.

### Initialize Tasks

```http
POST /api/progress/initializeTasks
```

Body fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `agent_id` | string | Yes | Agent id connected to the access key. |
| `access_key` | string | Yes | Raw agent access key. |
| `session_id` | string | Recommended | External session identifier. Defaults to `default` if omitted. |
| `items` | array | Yes | List of progress task objects. |

Example:

```json
{
  "agent_id": "6a1d614955f55909e1272f02",
  "access_key": "YOUR_AGENT_ACCESS_KEY",
  "session_id": "unity-session-001",
  "items": [
    {
      "task_name": "Repair engine",
      "description": "Repair the ship engine",
      "status": "pending",
      "subtask_progress": []
    }
  ]
}
```

### Update One Task

```http
POST /api/progress/updateTask
```

Body fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `agent_id` | string | Yes | Agent id connected to the access key. |
| `access_key` | string | Yes | Raw agent access key. |
| `session_id` | string | Recommended | External session identifier. Defaults to `default` if omitted. |
| `task_name` | string | Yes | Task identifier. |
| `description` | string | Yes | Human-readable task description. |
| `status` | string | Yes | `pending`, `started`, or `complete`. |
| `subtask_progress` | array | No | Subtasks and step progress. |

Example:

```json
{
  "agent_id": "6a1d614955f55909e1272f02",
  "access_key": "YOUR_AGENT_ACCESS_KEY",
  "session_id": "unity-session-001",
  "task_name": "Repair engine",
  "description": "Repair the ship engine",
  "status": "started",
  "subtask_progress": [
    {
      "subtask_name": "Find tool",
      "description": "Find the wrench",
      "completed": true,
      "step_progress": [
        {
          "step_name": "Open toolbox",
          "repetition_number": 0,
          "completed": true
        }
      ]
    }
  ]
}
```

### Fetch Progress

```http
GET /api/progress?agent_id=AGENT_ID&session_id=SESSION_ID
```

Header:

```text
access-key: YOUR_AGENT_ACCESS_KEY
```

Example:

```bash
curl "https://iplvr.it.ntnu.no/backend/api/progress?agent_id=6a1d614955f55909e1272f02&session_id=unity-session-001" \
  -H "access-key: YOUR_AGENT_ACCESS_KEY"
```

Optional query fields:

| Field | Type | Description |
| --- | --- | --- |
| `session_id` | string | Return progress for this session. Defaults to `default` if omitted. |
| `limit` | number | Return only the newest N progress tasks. |

Fetched progress can be passed back into `/api/chat/ask` or `/api/chat/askTranscribe` in the optional `progress` field, but this is no longer required when the chat request includes the same `session_id`. The backend automatically loads recent progress for that session.

## External Chat UI for Testing

RAGdollChat includes a test page for external access-key use:

```text
/external
```

Production:

```text
https://iplvr.it.ntnu.no/chat/external
```

Local:

```text
http://localhost:3001/external
```

The page lets you choose:

- Local backend: `http://localhost:8000`
- Server backend: `https://iplvr.it.ntnu.no/backend`

Then enter:

- Agent access key
- Role name

Use this page to verify that a key and role work before integrating an external application. After connecting, the page also includes endpoint test tools for:

- `POST /api/chat/ask`
- `POST /api/chat/transcribe`
- `POST /api/chat/askTranscribe`
- `POST /api/progress/initializeTasks`
- `POST /api/progress/updateTask`
- `GET /api/progress`

## Minimal JavaScript Example

```js
const backendUrl = "https://iplvr.it.ntnu.no/backend";
const accessKey = "YOUR_AGENT_ACCESS_KEY";
const role = "student";

async function resolveAgent() {
  const response = await fetch(`${backendUrl}/agent-info-by-accesskey`, {
    headers: {
      "access-key": accessKey,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to resolve agent: ${response.status}`);
  }

  return response.json();
}

async function askAgent(question) {
  const agent = await resolveAgent();

  const response = await fetch(`${backendUrl}/api/chat/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      agent_id: agent.agent_id,
      active_role_id: role,
      access_key: accessKey,
      chat_log: [{ role: "user", content: question }],
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to ask agent: ${response.status}`);
  }

  const data = await response.json();
  return data.response.response;
}

askAgent("Hello, what can you help me with?")
  .then(console.log)
  .catch(console.error);
```

## Troubleshooting

### `401 Unauthorized` from `/agent-info-by-accesskey`

Most common causes:

- The value is not the full raw access key.
- The key has expired.
- The key was revoked.
- The key belongs to another environment.
- The browser/app is calling local backend while the key exists only on the server, or the reverse.

### `400 Role not found`

The `active_role_id` must exactly match a role name configured on the agent.

Use `/agent-info-by-accesskey` to list valid role names.

### External browser app blocked by CORS

If a browser app on localhost calls the production backend directly, the production backend must allow that origin.

Example server setting:

```env
CORS_ALLOWED_ORIGINS=http://localhost:3001,http://localhost:4001
```

For a stricter production setup, remove localhost origins when they are no longer needed.

### Key is hidden in the UI

If the key was created as `View once`, it cannot be revealed again from the list. Create a new key and copy it immediately, or create a `View anytime` key when repeated access is needed.
