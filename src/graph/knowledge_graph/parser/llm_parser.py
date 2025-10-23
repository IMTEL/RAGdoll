"""
Graph Parser that parses natural language text chunks into a graph structure using langchain

Pydantic-validated schema ➜ guarantees well-formed output

Structured-output prompt with real examples ➜ better accuracy

OutputFixingParser ➜ automatic retry if the LLM drifts

Canonical ID helper (lower_snake_case with underscores) used
everywhere ➜ no more edge-endpoint mismatches

Sentence tokeniser (NLTK) ➜ splits long paragraphs for higher recall

Ontology growth ➜ unseen labels or relation types are added on the fly and logged

ID	keep canonical slug (plato, virtue_ethics)—stable across runs; DB uniqueness on id.
Node label property	store type as ALLCAPS (PERSON). Cypher can use : dynamic label if you wish.
Display name	always include properties.name (for people) or properties.title (for works).
Unknown node/edge label	accept as NEW_*, add to whitelist, log warning.
LLM stack	PromptTemplate → ChatOpenAI → OutputFixingParser(Pydantic).
Batch size	handled later in the DB ingest layer; parser stays stateless.
"""

#pip install langchain-openai langchain-core python-slugify nltk python-dotenv
#python -m nltk.downloader punkt 

from __future__ import annotations
import logging

from dotenv import load_dotenv
load_dotenv()
from src.core.config import Config

# from nltk.tokenize import sent_tokenize

from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.prompts import PromptTemplate
from src.services.knowledge_graph.parser.parser import Parser
from src.db.graph.models import Node, Edge, NodeType, EdgeType, GraphPayload

config = Config()
sent_tokenize = lambda text: [text]
logger = logging.getLogger(__name__)

base_parser   = PydanticOutputParser(pydantic_object=GraphPayload)
repair_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model=config.GPT_MODEL, temperature=0),
)

VALID_EDGE_TYPES: frozenset[str] = frozenset({
    "AUTHORED",
    "INTRODUCED",
    "FORMULATED",
    "PROPOSED",
    "PUPIL_OF",
    "MENTOR_OF",
    "INFLUENCED",
    "DERIVED_FROM",
    "BELONGS_TO",
    "STUDIED_IN",
    "PART_OF",
    "LIVED_IN",
    "BORN_IN",
    "WORKED_AT",
    "CHILD_OF",
    "SIBLING_OF",
    "SPOUSE_OF",
    "DESCRIBES",
    "REFERENCES",
    "RELATED_TO",
    "KNOWS",
})


SYSTEM_PROMPT = """
You are an information-extraction agent.

Return ONLY valid JSON matching this schema:
{format_instructions}

Rules
-----
• Node `id` must be lower_snake_case (underscores, no spaces/hyphens).  
• `type` must be one of the allowed node labels (or NEW_*).  
• Edge `type` must be one of the allowed edge labels (or NEW_EDGE_TYPE_*).  
• Provide at least a `"name"` or `"title"` in node.properties.
• If appliccable, include description in node.properties.

Allowed node labels: {topic_labels}  
Allowed edge labels: {edge_labels}

Example
-------
Input: "Leonhard Euler introduced Euler's Identity in 1748."
Output:
{{
  "nodes":[
    {{"id":"leonhard_euler","type":"PERSON","title":"Leonhard Euler",
     "properties":{{"born":"1707","died":"1783"}}}},
    {{"id":"eulers_identity","type":"IDENTITY","title":"Euler's Identity",
     "properties":{{"year_introduced":"1748"}}}}
  ],
  "edges":[
    {{"from":"leonhard_euler","to":"eulers_identity","type":"INTRODUCED"}}
  ]
}}

Now extract entities and relations from this text:
"{text}"
"""
class LLM_Parser(Parser):
    
    def __init__(self, topic_labels: set[str]) -> None:
        """Initialize the LLM parser with topic and edge labels."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self.edge_labels = VALID_EDGE_TYPES
        self.topic_labels = topic_labels
        
        self.prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["text"],
            partial_variables={
                "format_instructions": base_parser.get_format_instructions(),
                "topic_labels": ", ".join(sorted(self.topic_labels)),
                "edge_labels":  ", ".join(sorted(self.edge_labels)),
            },
        )
        self.chain = self.prompt | ChatOpenAI(model=config.GPT_MODEL, temperature=0) | repair_parser


    # ==========================================================
    # Graph Parser
    # ==========================================================

    @staticmethod
    def ensure(label: str, store: set[str], kind: str) -> None:
        """Add unseen labels to ontology and log."""
        if label not in store:
            store.add(label)
            logging.warning("[Ontology] added new %s label: %s", kind, label)

    def parse_text(self, text: str, graph_id: str, split_sentences: bool = False) -> dict:
        """
        Convert raw text into { "nodes":[…], "edges":[…] } ready for Neo4j ingest.
        """
        sentences = sent_tokenize(text) if split_sentences else [text]
        node_map: dict[str, Node] = {}
        edge_seen: set[tuple[str, str, str]] = set()
        edges: list[Edge] = []

        for sentence in sentences:
            payload: GraphPayload = self.chain.invoke({"text": sentence})

            # merge nodes
            for n in payload.nodes:
                node_map.setdefault(n.id, n)

            # merge & dedup edges
            for e in payload.edges:
                key = (e.from_, e.to, e.type)
                if key not in edge_seen:
                    edge_seen.add(key)
                    edges.append(e)

        return {
            "nodes": [n.model_dump(exclude_none=True) for n in node_map.values()],
            "edges": [e.model_dump(exclude_none=True, by_alias=True) for e in edges],
        }
        
    def parse_multiple_texts(self, texts: list[str], graph_id: str, chunk_id: str, document_id: str, split_sentences: bool = True) -> dict:
        """
        Aggregate nodes/edges from several independent text snippets.
        Returns a dict compatible with GraphPayload instantiation.
        """
        merged_nodes: dict[str, Node] = {}
        edge_seen: set[tuple[str, str, str]] = set()
        edges: list[Edge] = []

        for chunk in texts:
            g = self.parse_text(chunk, graph_id, split_sentences=split_sentences)
            for n in g["nodes"]:
                merged_nodes.setdefault(n["id"], Node(**n))   # dedupe by id
            for e in g["edges"]:
                key = (e["from"], e["to"], e["type"])
                if key not in edge_seen:
                    edge_seen.add(key)
                    edges.append(Edge(**e))

        return {
            "graph_id": graph_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "nodes": [n.model_dump(exclude_none=True) for n in merged_nodes.values()],
            "edges": [e.model_dump(exclude_none=True, by_alias=True) for e in edges],
        }
    
    def parse_multiple_texts_optimized(self, texts: list[str], graph_id: str, chunk_id: str, document_id: str, split_sentences: bool = True) -> dict:
        """
        Optimized batch processing that combines multiple texts into a single LLM call.
        This is more efficient than processing each text individually.
        """
        if not texts:
            return {"graph_id": graph_id, "chunk_id": chunk_id, "document_id": document_id, "nodes": [], "edges": []}
        
        if len(texts) == 1:
            # Single text - use regular processing
            result = self.parse_text(texts[0], graph_id, split_sentences)
            # Add the missing fields to make it compatible with GraphPayload
            return {
                "graph_id": graph_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "nodes": result["nodes"],
                "edges": result["edges"],
            }
        
        # Combine all texts into a single batch for LLM processing
        if split_sentences:
            # If splitting sentences, combine all sentences from all texts
            all_sentences = []
            for text in texts:
                all_sentences.extend(sent_tokenize(text))
            
            # Process all sentences as a single batch
            combined_text = " ".join(all_sentences)
        else:
            # Combine all texts with clear separators
            combined_text = "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(texts)
        
        # Make a single LLM call for the combined text
        try:
            payload: GraphPayload = self.chain.invoke({"text": combined_text})
            
            # Return the combined result with required fields
            return {
                "graph_id": graph_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "nodes": [n.model_dump(exclude_none=True) for n in payload.nodes],
                "edges": [e.model_dump(exclude_none=True, by_alias=True) for e in payload.edges],
            }
        except Exception as e:
            # If batch processing fails, fall back to individual processing
            return self.parse_multiple_texts(texts, graph_id, chunk_id, document_id, split_sentences)
    
    def parse_multiple_texts_quality_optimized(self, text_chunk_pairs: list[tuple[str, str]], graph_id: str, split_sentences: bool = True) -> dict:
        """
        Quality-optimized batch processing that balances speed and graph quality.
        
        This approach:
        1. Processes each text individually to maintain context quality
        2. But batches the API calls in smaller groups to reduce overhead
        3. Merges results intelligently to avoid information loss
        
        Args:
            text_chunk_pairs: List of (text, chunk_id) tuples
            graph_id: The graph ID
            split_sentences: Whether to split sentences during parsing
        """
        if not text_chunk_pairs:
            return {"nodes": [], "edges": []}
        
        if len(text_chunk_pairs) == 1:
            text, chunk_id = text_chunk_pairs[0]
            result = self.parse_text(text, graph_id, split_sentences)
            # Add chunk_id to all nodes and edges
            for node in result["nodes"]:
                node["chunk_ids"] = [chunk_id]
            for edge in result["edges"]:
                edge["chunk_ids"] = [chunk_id]
            return result
        
        # Smart batching: process in small groups to balance quality and speed
        batch_size = 3  # Process 3 texts at a time - good balance
        merged_nodes: dict[str, Node] = {}
        edge_seen: set[tuple[str, str, str]] = set()
        edges: list[Edge] = []
        
        # Process text-chunk pairs in small batches
        for i in range(0, len(text_chunk_pairs), batch_size):
            batch_pairs = text_chunk_pairs[i:i + batch_size]
            
            if len(batch_pairs) == 1:
                # Single text - process individually for best quality
                text, chunk_id = batch_pairs[0]
                g = self.parse_text(text, graph_id, split_sentences)
                
                # Merge nodes with chunk_id
                for n in g["nodes"]:
                    node = Node(**n)
                    if node.id in merged_nodes:
                        # Merge chunk_ids for existing node
                        if chunk_id not in merged_nodes[node.id].chunk_ids:
                            merged_nodes[node.id].chunk_ids.append(chunk_id)
                    else:
                        # New node
                        node.chunk_ids = [chunk_id]
                        merged_nodes[node.id] = node
                
                # Merge edges with chunk_id
                for e in g["edges"]:
                    edge = Edge(**e)
                    key = (edge.from_, edge.to, edge.type)
                    if key not in edge_seen:
                        edge.chunk_ids = [chunk_id]
                        edge_seen.add(key)
                        edges.append(edge)
                    else:
                        # Find existing edge and add chunk_id
                        for existing_edge in edges:
                            if (existing_edge.from_, existing_edge.to, existing_edge.type) == key:
                                if chunk_id not in existing_edge.chunk_ids:
                                    existing_edge.chunk_ids.append(chunk_id)
                                break
            else:
                # Small batch - combine with separators for efficiency
                batch_texts = [text for text, _ in batch_pairs]
                batch_chunk_ids = [chunk_id for _, chunk_id in batch_pairs]
                combined_text = "\n\n--- CONTEXT SEPARATOR ---\n\n".join(batch_texts)
                
                try:
                    # Single LLM call for the small batch
                    payload: GraphPayload = self.chain.invoke({"text": combined_text})
                    
                    # Merge nodes with all chunk_ids from this batch
                    for n in payload.nodes:
                        if n.id in merged_nodes:
                            # Merge chunk_ids for existing node
                            for chunk_id in batch_chunk_ids:
                                if chunk_id not in merged_nodes[n.id].chunk_ids:
                                    merged_nodes[n.id].chunk_ids.append(chunk_id)
                        else:
                            # New node
                            n.chunk_ids = batch_chunk_ids.copy()
                            merged_nodes[n.id] = n
                    
                    # Merge edges with all chunk_ids from this batch
                    for e in payload.edges:
                        key = (e.from_, e.to, e.type)
                        if key not in edge_seen:
                            e.chunk_ids = batch_chunk_ids.copy()
                            edge_seen.add(key)
                            edges.append(e)
                        else:
                            # Find existing edge and add chunk_ids
                            for existing_edge in edges:
                                if (existing_edge.from_, existing_edge.to, existing_edge.type) == key:
                                    for chunk_id in batch_chunk_ids:
                                        if chunk_id not in existing_edge.chunk_ids:
                                            existing_edge.chunk_ids.append(chunk_id)
                                    break
                            
                except Exception as e:
                    # If batch fails, fall back to individual processing
                    logger.warning(f"Batch processing failed, falling back to individual: {e}")
                    for text, chunk_id in batch_pairs:
                        g = self.parse_text(text, graph_id, split_sentences)
                        
                        # Merge nodes with chunk_id
                        for n in g["nodes"]:
                            node = Node(**n)
                            if node.id in merged_nodes:
                                # Merge chunk_ids for existing node
                                if chunk_id not in merged_nodes[node.id].chunk_ids:
                                    merged_nodes[node.id].chunk_ids.append(chunk_id)
                            else:
                                # New node
                                node.chunk_ids = [chunk_id]
                                merged_nodes[node.id] = node
                        
                        # Merge edges with chunk_id
                        for e in g["edges"]:
                            edge = Edge(**e)
                            key = (edge.from_, edge.to, edge.type)
                            if key not in edge_seen:
                                edge.chunk_ids = [chunk_id]
                                edge_seen.add(key)
                                edges.append(edge)
                            else:
                                # Find existing edge and add chunk_id
                                for existing_edge in edges:
                                    if (existing_edge.from_, existing_edge.to, existing_edge.type) == key:
                                        if chunk_id not in existing_edge.chunk_ids:
                                            existing_edge.chunk_ids.append(chunk_id)
                                        break
        
        return {
            "nodes": [n.model_dump(exclude_none=True) for n in merged_nodes.values()],
            "edges": [e.model_dump(exclude_none=True, by_alias=True) for e in edges],
        }











