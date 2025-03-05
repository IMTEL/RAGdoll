## **Chat service pipeline**

**Query Embedding:** Convert the user’s query (or a combination of query and context) into an embedding using your embeddings model.

**Context Retrieval:** Perform a similarity search in your database to pull out relevant pieces of text (guidance about tasks, environment details, etc.).

**Prompt Generation:** Combine the user’s query, the retrieved context, and additional information (like user actions or environmental variables) into a well-structured prompt.

**LLM Generation:** Send the prompt to your LLM to generate a helpful, natural language response.

**Return Answer:** Deliver the response back to the VR environment, where the NPC can relay the information to the user.