## **Chat service pipeline**

**Query Embedding:** Convert the user’s query (or a combination of query and context) into an embedding using your embeddings model.

**Context Retrieval:** Perform a similarity search in your database to pull out relevant pieces of text (guidance about tasks, environment details, etc.).

**Prompt Generation:** Combine the user’s query, the retrieved context, and additional information (like user actions or environmental variables) into a well-structured prompt.

**LLM Generation:** Send the prompt to your LLM to generate a helpful, natural language response.

**Return Answer:** Deliver the response back to the VR environment, where the NPC can relay the information to the user.

###  RAG Pipeline Implementation

**a. Query Processing & Embedding:**

    When a user submits a query, first clean and preprocess the text.
    Use your embeddings model to generate an embedding vector for the query.

**b. Similarity Search:***

    Use your cosine similarity function to compare the query embedding with stored document embeddings.
    Retrieve the top N documents (e.g., top 3) that are most relevant to the query.

**c. Prompt Generation:**

    Combine:
        The base prompt (e.g., "You are a helpful assistant.")
        The user query
        The retrieved context (excerpts or summaries from the documents)
        Any additional user action or VR environment data
    This composite prompt should provide sufficient context for the LLM to generate a useful response.

**d. LLM Response:**

    Pass the prompt to your LLM.
    Post-process the generated response if needed (e.g., trimming, formatting).