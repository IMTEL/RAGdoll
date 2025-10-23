
from enum import Enum, auto

from src.services.knowledge_graph.parser.parser import Parser
from src.services.knowledge_graph.parser.llm_parser import LLM_Parser
from src.services.knowledge_graph.parser.nlp_parser import NLP_Parser

class ParserType(Enum):
    """
    Enum for chunk database types.
    """
    LLM_PARSER = auto()
    HYBRID_NLP_PARSER = auto()

def get_parser(parser_type: ParserType, topic_labels: list[str]) -> Parser:
    """
    Factory function to get the appropriate parser based on the parser type.

    Args:
        parser_type (str): The type of parser to retrieve.

    Returns:
        KnowledgeGraphParser: An instance of the specified parser.
    """
    match parser_type:
        case ParserType.LLM_PARSER:
            return LLM_Parser(topic_labels)
        case ParserType.HYBRID_NLP_PARSER:
            return NLP_Parser(topic_labels)
        case _:
            raise ValueError(f"Unknown parser type: {parser_type}")
