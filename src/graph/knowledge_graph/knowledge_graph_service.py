import json
from typing import TextIO

from fastapi import HTTPException

from src.db.graph.enums import GraphDatabaseType
from src.db.models import Status
from src.db.graph.dao import create_graph_database
from src.db.graph.models import GraphPayload
from src.services.knowledge_graph.parser.llm_parser import LLM_Parser
from src.services.knowledge_graph.parser.parser import Parser
from src.services.knowledge_graph.parser.parser_factory import get_parser, ParserType


BASE_LABELS: frozenset[str] = frozenset({
        "PERSON", "PHILOSOPHER", "SCIENTIST", "IDEA", "CONCEPT",
        "FORMULA", "THEOREM", "IDENTITY", "METHOD", "PRINCIPLE",
        "EQUATION", "BOOK", "PUBLICATION", "PAPER", "TREATISE",
        "LOCATION", "CITY", "COUNTRY", "INSTITUTION", "UNIVERSITY",
        "SOCIETY", "AREA", "DISCIPLINE", "TOPIC", "EVENT",
        "DISCOVERY", "APPLICATION",
    })

class KnowledgeGraphService:
    # TODO: should this be singleton?
        
    def init_with_graph_id(self, graph_id: str):
        """
        Initialize the knowledge graph service with the specified database type.
        """
        self.graph_id = graph_id
        self.db = create_graph_database(GraphDatabaseType.NEO4J)
        topic_labels = self.get_all_labels()
        self.parser: Parser = get_parser(ParserType.LLM_PARSER, topic_labels)
        
    def get_all_labels(self) -> set[str]:
        # graph_labels = set(self.db.get_topic_labels(self.graph_id))
        # return BASE_LABELS | graph_labels
        return BASE_LABELS

    def populate_graph_from_text(self, text: str, chunk_id: str, document_id: str, split_sentences: bool = False) -> GraphPayload:
        """
        Ingest a single text snippet into the knowledge graph.
        """
        graph_data = self.parser.parse_text(text, self.graph_id, split_sentences)
        graph_data["graph_id"] = str(self.graph_id)
        graph_data["chunk_id"] = str(chunk_id)
        graph_data["document_id"] = str(document_id)
        payload = GraphPayload(**graph_data)
        status = self.db.post_graph(payload)
        if status.status != "success":
            raise Exception(f"Failed to ingest text: {status.detail}")
        return payload

    def populate_graph_from_multiple_texts(self, texts: list[str], chunk_id: str, document_id: str, split_sentences: bool = True) -> GraphPayload:
        """
        Ingest multiple text snippets into the knowledge graph.
        """
        graph_data = self.parser.parse_multiple_texts(texts, self.graph_id, chunk_id, document_id, split_sentences)
        payload = GraphPayload(**graph_data)
        status = self.db.post_graph(payload)
        if status.status != "success":
            raise Exception(f"Failed to ingest texts: {status.detail}")
        return payload
    
    def load_payload_from_file(self, f: TextIO) -> GraphPayload:
        """
        Given an open file handle (JSON), parse into GraphPayload.
        """
        try:
            data = json.load(f)
            return GraphPayload(**data)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON payload: {e}"
            )

    def populate_graph_from_text_batch(self, text_chunk_pairs: list[tuple[str, str]], split_sentences: bool = False) -> list[GraphPayload]:
        """
        Ingest multiple text snippets with their chunk IDs into the knowledge graph in batch.
        
        This method uses the existing batch parsing capability to process multiple texts
        more efficiently than individual processing.
        
        Args:
            text_chunk_pairs: List of (text, chunk_id) tuples
            split_sentences: Whether to split sentences during parsing
            
        Returns:
            List of GraphPayload objects for each processed chunk
        """
        if not text_chunk_pairs:
            return []
        
        if len(text_chunk_pairs) == 1:
            # Single text - use regular processing
            text, chunk_id = text_chunk_pairs[0]
            return [self.populate_graph_from_text(text, chunk_id, split_sentences)]
        
        # Multiple texts - use batch processing
        # Use the quality-optimized batch parsing method for better balance
        batch_graph_data = self.parser.parse_multiple_texts_quality_optimized(text_chunk_pairs, self.graph_id, split_sentences)
        
        # Extract all unique chunk IDs
        all_chunk_ids = [chunk_id for _, chunk_id in text_chunk_pairs]
        
        # Create a single batch payload with combined graph data
        batch_payload_data = {
            "nodes": batch_graph_data["nodes"],
            "edges": batch_graph_data["edges"],
            "graph_id": str(self.graph_id),
            "chunk_id": "batch_" + "_".join(all_chunk_ids)
        }
        
        # Post the batch as a single operation
        batch_payload = GraphPayload(**batch_payload_data)
        status = self.db.post_graph(batch_payload)
        if status.status != "success":
            raise Exception(f"Failed to ingest batch: {status.detail}")
        
        # Return individual payloads for each chunk for API compatibility
        # Each chunk gets the same combined graph data but individual chunk_id
        payloads = []
        for text, chunk_id in text_chunk_pairs:
            individual_payload = GraphPayload(**{
                "nodes": batch_graph_data["nodes"],
                "edges": batch_graph_data["edges"],
                "graph_id": str(self.graph_id),
                "chunk_id": str(chunk_id)
            })
            payloads.append(individual_payload)
        
        return payloads


