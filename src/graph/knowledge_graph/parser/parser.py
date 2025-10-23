from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Parser(ABC):
    
    @abstractmethod
    def parse_text(self, text: str, graph_id: str, split_sentences: bool = False) -> dict:
        """
        Convert raw text into { "nodes":[…], "edges":[…] } ready for Neo4j ingest.
        """
        pass
    
    @abstractmethod
    def parse_multiple_texts(self, texts: list[str], graph_id: str, chunk_id: str, document_id: str, split_sentences: bool = True) -> dict:
        """
        Aggregate nodes/edges from several independent text snippets.
        Returns a dict compatible with GraphPayload instantiation.
        """
        pass