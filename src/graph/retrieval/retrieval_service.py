import logging
from typing import Dict, List
from uuid import UUID

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

from src.core.dependencies import get_llm
from src.db.content.dao import PostgresDatabase
from src.db.graph.dao import GraphDatabase
from src.models.chunks import Chunk
from src.services.embedding.base import EmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CYPHER_GENERATION_TEMPLATE = """
You are a Neo4j expert. Generate ONLY a Cypher query without any explanatory text.

Instructions:
1. Analyze the user's question to determine if they want:
   - Specific node types (e.g., "societies", "philosophers", "books") - use label filtering
   - Relationships between entities - include neighbor queries
   - Simple listing - return only matching nodes without relationships
2. For listing queries (e.g., "Which societies are there?", "List all philosophers"), return only the nodes without relationships.
3. For relationship queries (e.g., "How is X related to Y?"), include neighbor exploration.
4. Use label filtering when the question mentions specific types (e.g., SOCIETY, PHILOSOPHER, BOOK).
5. Combine entity matching and full-text search with UNION.
6. Return distinct results to avoid duplicates.
7. CRITICAL: Always filter nodes by document_ids using: WHERE ANY(doc_id IN $document_ids WHERE doc_id IN n.document_ids)
8. NEVER use: n.document_ids IN $document_ids (this will not work for multiple document IDs)
9. NEVER use: $document_ids CONTAINS n.document_ids (this will not work for multiple document IDs)
10. IMPORTANT: When using type(r) in RETURN clause, ensure the relationship variable [r] is defined in the MATCH pattern
11. Use -[r]-> or -[r:TYPE]-> patterns when you need to return relationship information
12. For "Who is X the mentor of?" queries, look for outgoing MENTOR_OF relationships from X
13. For "Who is the mentor of X?" queries, look for incoming MENTOR_OF relationships to X

Schema:
{schema}

Example 1 (Listing query):
Question: "Which societies are there in philosophy?"
Entities: ["societies", "philosophy"]
Document IDs: $document_ids
MATCH (n:SOCIETY)
WHERE ANY(doc_id IN $document_ids WHERE doc_id IN n.document_ids)
RETURN DISTINCT n.title AS node_title, labels(n) AS node_labels, properties(n) AS node_properties, null AS relationship_type, null AS neighbor_title, null AS neighbor_labels, null AS neighbor_properties
UNION
CALL db.index.fulltext.queryNodes('node_titles_and_text', $query + '~') YIELD node, score
WHERE 'SOCIETY' IN labels(node) AND ANY(doc_id IN $document_ids WHERE doc_id IN node.document_ids)
RETURN DISTINCT node.title AS node_title, labels(node) AS node_labels, properties(node) AS node_properties, null AS relationship_type, null AS neighbor_title, null AS neighbor_labels, null AS neighbor_properties

Example 2 (Relationship query):
Question: "How is Isaac Newton related to gravity?"
Entities: ["Isaac Newton", "gravity"]
Document IDs: $document_ids
MATCH (n) WHERE n.title IN $entities AND ANY(doc_id IN $document_ids WHERE doc_id IN n.document_ids)
WITH n
OPTIONAL MATCH (n)-[r]-(neighbor)
WHERE ANY(doc_id IN $document_ids WHERE doc_id IN neighbor.document_ids)
RETURN DISTINCT n.title AS node_title, labels(n) AS node_labels, properties(n) AS node_properties, type(r) AS relationship_type, neighbor.title AS neighbor_title, labels(neighbor) AS neighbor_labels, properties(neighbor) AS neighbor_properties
UNION
CALL db.index.fulltext.queryNodes('node_titles_and_text', $query + '~') YIELD node, score
WHERE ANY(doc_id IN $document_ids WHERE doc_id IN node.document_ids)
WITH node
OPTIONAL MATCH (node)-[r]-(neighbor)
WHERE ANY(doc_id IN $document_ids WHERE doc_id IN neighbor.document_ids)
RETURN DISTINCT node.title AS node_title, labels(node) AS node_labels, properties(node) AS node_properties, type(r) AS relationship_type, neighbor.title AS neighbor_title, labels(neighbor) AS neighbor_labels, properties(neighbor) AS neighbor_properties

Example 3 (Mentor query):
Question: "Who is Parmenides the mentor of?"
Entities: ["Parmenides"]
Document IDs: $document_ids
MATCH (n:PHILOSOPHER {{title: 'Parmenides'}})-[r:MENTOR_OF]->(student)
WHERE ANY(doc_id IN $document_ids WHERE doc_id IN n.document_ids) AND ANY(doc_id IN $document_ids WHERE doc_id IN student.document_ids)
RETURN DISTINCT n.title AS node_title, labels(n) AS node_labels, properties(n) AS node_properties, type(r) AS relationship_type, student.title AS neighbor_title, labels(student) AS neighbor_labels, properties(student) AS neighbor_properties
UNION
CALL db.index.fulltext.queryNodes('node_titles_and_text', 'Parmenides~') YIELD node, score
WITH node
OPTIONAL MATCH (node)-[r:MENTOR_OF]->(neighbor)
WHERE ANY(doc_id IN $document_ids WHERE doc_id IN node.document_ids) AND ANY(doc_id IN $document_ids WHERE doc_id IN neighbor.document_ids)
RETURN DISTINCT node.title AS node_title, labels(node) AS node_labels, properties(node) AS node_properties, type(r) AS relationship_type, neighbor.title AS neighbor_title, labels(neighbor) AS neighbor_labels, properties(neighbor) AS neighbor_properties

Question: {query}
Entities: {entities}
Document IDs: $document_ids
"""

ENTITY_EXTRACTION_TEMPLATE = """
From the following text, extract up to 5 key entities (people, places, concepts, topics) that could be nodes in a knowledge graph.
Focus on proper nouns and key technical terms.
Return the entities as a comma-separated list.

Text: {text}
Entities:
"""

GRAPH_SCHEMA = """
Node labels: ['PERSON', 'PHILOSOPHER', 'SCIENTIST', 'IDEA', 'CONCEPT', 'BOOK', 'AREA', 'DISCIPLINE', 'TOPIC', 'EVENT', 'LOCATION', 'INSTITUTION', 'SOCIETY']
Relationship types: ['AUTHORED', 'INTRODUCED', 'INFLUENCED', 'MENTOR_OF', 'BELONGS_TO', 'PART_OF', 'LIVED_IN', 'WORKED_AT', 'RELATED_TO', 'DERIVED_FROM', 'DESCRIBES']
Node properties: Each node has a 'title' (string), 'document_ids' (list of UUIDs), and may have additional properties like 'description', 'date', 'summary', etc. Use properties(n) to return all available properties.
Full-Text Index: A full-text index named 'node_titles_and_text' exists for searching node titles.
Document Filtering: ALWAYS filter nodes by document_ids using: WHERE ANY(doc_id IN $document_ids WHERE doc_id IN n.document_ids)
CRITICAL: Do NOT use n.document_ids IN $document_ids or $document_ids CONTAINS n.document_ids - these will fail for multiple document IDs!
Query Intent Recognition: 
- For listing queries (e.g., "Which X are there?", "List all Y"), filter by node labels and return only nodes without relationships
- For relationship queries (e.g., "How is X related to Y?"), include neighbor exploration
- Use label filtering based on question keywords (societies → SOCIETY, philosophers → PHILOSOPHER, etc.)
"""


class RetrievalService:
    """
    Implements a multi-step retrieval chain combining vector search, entity extraction,
    and a hybrid graph query to fetch rich, relevant context.
    """

    def __init__(
        self,
        pg_db: PostgresDatabase,
        graph_db: GraphDatabase,
        em: EmbeddingModel,
    ):
        self.pg_db = pg_db
        self.graph_db = graph_db
        self.em = em
        self.llm = get_llm()
        
        entity_prompt = PromptTemplate.from_template(ENTITY_EXTRACTION_TEMPLATE)
        self.entity_extraction_chain = entity_prompt | self.llm | StrOutputParser()
        
        self.retrieval_chain = self._create_retrieval_chain()

    def _clean_cypher_query(self, text: str) -> str:
        """A helper function to clean markdown fences from the LLM's output."""
        logger.info(f"Raw LLM output: {text}")
        
        # Look for code blocks with ```cypher or ```
        import re
        
        # Pattern to match code blocks
        cypher_pattern = r'```(?:cypher)?\s*\n(.*?)```'
        matches = re.findall(cypher_pattern, text, re.DOTALL)
        
        if matches:
            # Take the first match (should be the Cypher query)
            cypher_query = matches[0].strip()
            logger.info(f"Extracted from code block: {cypher_query}")
        else:
            # Fallback: look for lines starting with Cypher keywords
            lines = text.split('\n')
            cypher_lines = []
            in_cypher = False
            
            for line in lines:
                stripped = line.strip()
                # Start collecting when we see a Cypher keyword
                if not in_cypher and stripped.upper().startswith(('MATCH', 'RETURN', 'WITH', 'CALL', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'UNION')):
                    in_cypher = True
                    cypher_lines.append(line)
                elif in_cypher:
                    # Stop if we hit explanatory text or empty lines after the query
                    if stripped.startswith(('This query', 'The query', 'Based on')):
                        break
                    cypher_lines.append(line)
            
            cypher_query = '\n'.join(cypher_lines).strip()
            logger.info(f"Extracted from keyword detection: {cypher_query}")
        
        # Final cleanup
        if cypher_query.startswith('```cypher'):
            cypher_query = cypher_query[9:].strip()
        if cypher_query.startswith('```'):
            cypher_query = cypher_query[3:].strip()
        if cypher_query.endswith('```'):
            cypher_query = cypher_query[:-3].strip()
        
        # Fix common document filtering mistakes
        cypher_query = self._fix_document_filtering_syntax(cypher_query)
        
        # Fix common relationship variable mistakes
        cypher_query = self._fix_relationship_variable_syntax(cypher_query)
        
        logger.info(f"Final cleaned Cypher query: {cypher_query}")
        return cypher_query
    
    def _fix_document_filtering_syntax(self, cypher_query: str) -> str:
        """Fix common document filtering syntax mistakes in Cypher queries."""
        
        import re
        
        # Common mistake: n.document_ids IN $document_ids
        # This checks if the entire array is in the parameter list, which never matches
        # Fix: Replace with ANY(doc_id IN $document_ids WHERE doc_id IN n.document_ids)
        
        incorrect_pattern = r'(\w+)\.document_ids\s+IN\s+\$document_ids'
        correct_replacement = r'ANY(doc_id IN $document_ids WHERE doc_id IN \1.document_ids)'
        
        if re.search(incorrect_pattern, cypher_query):
            logger.warning(f"Found incorrect document filtering syntax, fixing...")
            cypher_query = re.sub(incorrect_pattern, correct_replacement, cypher_query)
            logger.info(f"Fixed query: {cypher_query}")
        
        # Another common mistake: WHERE $document_ids CONTAINS n.document_ids
        incorrect_pattern_2 = r'\$document_ids\s+CONTAINS\s+(\w+)\.document_ids'
        if re.search(incorrect_pattern_2, cypher_query):
            logger.warning(f"Found incorrect CONTAINS syntax, fixing...")
            cypher_query = re.sub(incorrect_pattern_2, r'ANY(doc_id IN $document_ids WHERE doc_id IN \1.document_ids)', cypher_query)
            logger.info(f"Fixed query: {cypher_query}")
        
        # Validate that the query has proper document filtering
        if '$document_ids' in cypher_query and 'ANY(doc_id IN $document_ids WHERE doc_id IN' not in cypher_query:
            logger.warning(f"Query uses $document_ids but doesn't have proper ANY() syntax")
            logger.warning(f"Query might not filter correctly for multiple document IDs")
        
        return cypher_query

    def _fix_relationship_variable_syntax(self, cypher_query: str) -> str:
        """Fix common relationship variable syntax mistakes in Cypher queries."""
        
        import re
        
        # Common mistake: Using type(r) without defining the r variable
        if 'type(r)' in cypher_query:
            logger.warning(f"Query uses type(r), checking if all relationship patterns have r variable...")
            
            # Find all relationship patterns and check if they have the r variable
            # Look for patterns like: -[:TYPE]-> or -[]-> but not -[r:TYPE]-> or -[r]->
            
            # Pattern 1: -[:TYPE]-> (missing r variable)
            missing_r_patterns = re.findall(r'-\[:([\w_]+)\]->', cypher_query)
            if missing_r_patterns:
                logger.warning(f"Found relationship patterns without r variable: {missing_r_patterns}")
                cypher_query = re.sub(r'-\[:([\w_]+)\]->', r'-[r:\1]->', cypher_query)
                logger.info(f"Fixed -[:TYPE]-> patterns")
            
            # Pattern 2: -[:TYPE]- (bidirectional, missing r variable)
            missing_r_patterns_bidir = re.findall(r'-\[:([\w_]+)\]-(?!>)', cypher_query)
            if missing_r_patterns_bidir:
                logger.warning(f"Found bidirectional relationship patterns without r variable: {missing_r_patterns_bidir}")
                cypher_query = re.sub(r'-\[:([\w_]+)\]-(?!>)', r'-[r:\1]-', cypher_query)
                logger.info(f"Fixed -[:TYPE]- patterns")
            
            # Pattern 3: -[]-> (any relationship, missing r variable)
            if re.search(r'-\[\]->', cypher_query):
                logger.warning(f"Found -[]-> pattern without r variable")
                cypher_query = re.sub(r'-\[\]->', r'-[r]->', cypher_query)
                logger.info(f"Fixed -[]-> patterns")
            
            # Pattern 4: -[]- (bidirectional any relationship, missing r variable)
            if re.search(r'-\[\]-(?!>)', cypher_query):
                logger.warning(f"Found -[]- pattern without r variable")
                cypher_query = re.sub(r'-\[\]-(?!>)', r'-[r]-', cypher_query)
                logger.info(f"Fixed -[]- patterns")
            
            # Final check: if we still have type(r) but no r variable anywhere, replace with null
            if 'type(r)' in cypher_query and not re.search(r'-\[r[^\]]*\]', cypher_query):
                logger.warning(f"Still no relationship variable r found, replacing type(r) with null...")
                cypher_query = cypher_query.replace('type(r)', 'null')
                logger.info(f"Replaced type(r) with null")
            
            logger.info(f"Final relationship variable fixed query: {cypher_query}")
        
        return cypher_query

    def _create_retrieval_chain(self):
        """Builds the custom LCEL chain for context retrieval."""

        cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
        cypher_generation_chain = (
            cypher_prompt 
            | self.llm 
            | StrOutputParser()
            | RunnableLambda(self._clean_cypher_query) # Add the cleaning step
        )

        vector_search_fn = RunnableLambda(self._vector_search).with_config(run_name="VectorSearch")
        extract_entities_fn = RunnableLambda(self._extract_entities).with_config(run_name="ExtractEntities")
        run_graph_query_fn = RunnableLambda(self._run_graph_query).with_config(run_name="RunGraphQuery")

        return (
            RunnablePassthrough.assign(vector_context=vector_search_fn)
            | RunnablePassthrough.assign(
                entities=(lambda x: extract_entities_fn.invoke(x["vector_context"]))
            )
            | RunnablePassthrough.assign(
                cypher_query=(
                    lambda x: cypher_generation_chain.invoke(
                        {
                            "schema": GRAPH_SCHEMA,
                            "query": x["query"],
                            "entities": x["entities"],
                        }
                    )
                )
            )
            | RunnablePassthrough.assign(graph_context=run_graph_query_fn)
            | (
                lambda x: {
                    "chunk_context": x["vector_context"],
                    "graph_context": x["graph_context"],
                }
            )
        )

    async def _vector_search(self, inputs: Dict) -> List[Dict]:
        """Performs vector search and returns serialized chunk data."""
        q_embedding = await self.em.get_embedding(inputs["query"])
        chunks = self.pg_db.get_chunks_by_similarity(
            document_ids=inputs["document_ids"],
            embedding=q_embedding,
            limit=inputs.get("vector_limit", 5),
            threshold=1.0
        )
        logger.info(f"Vector search found {len(chunks)} chunks.")
        return [Chunk.model_validate(c).model_dump() for c in chunks]

    def _extract_entities(self, chunks: List[Dict]) -> List[str]:
        """Extracts entities from chunk text using the pre-built chain."""
        text_blob = " ".join([chunk.get("text", "") for chunk in chunks])
        if not text_blob:
            return []

        entity_string = self.entity_extraction_chain.invoke({"text": text_blob})
        entities = [e.strip() for e in entity_string.split(",") if e.strip()]
        logger.info(f"Extracted entities: {entities}")
        return entities

    def _run_graph_query(self, inputs: Dict) -> List[Dict]:
        """Executes the generated Cypher query against the graph database."""
        cypher_query = inputs["cypher_query"]
        logger.info(f"Executing Cypher query: {cypher_query}")
        try:
            # Convert UUIDs to strings for Neo4j compatibility
            document_ids_str = [str(doc_id) for doc_id in inputs["document_ids"]]
            params = {
                "entities": inputs["entities"], 
                "query": inputs["query"],
                "document_ids": document_ids_str
            }
            results = self.graph_db.query(cypher_query, params=params)
            
            # Remove duplicates based on node_title and relationship_type
            seen = set()
            unique_results = []
            for result in results:
                # Create a unique key for deduplication
                key = (
                    result.get("node_title"),
                    result.get("relationship_type"),
                    result.get("neighbor_title")
                )
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            logger.info(f"Graph query returned {len(unique_results)} unique results (from {len(results)} total).")
            return unique_results
        except Exception as e:
            logger.error(f"Graph query failed: {e}", exc_info=True)
            return []

    async def retrieve_contexts(
        self,
        query: str,
        document_ids: List[UUID],
        vector_limit: int = 5,
        graph_limit: int = 15,
    ) -> Dict[str, List[Dict]]:
        """
        Invokes the custom retrieval chain to fetch combined vector and graph context.
        """
        logger.info(f"Initiating custom retrieval chain for query: '{query}'")

        chain_input = {
            "query": query,
            "document_ids": document_ids,
            "vector_limit": vector_limit,
            "graph_limit": graph_limit,
        }

        return await self.retrieval_chain.ainvoke(chain_input)