import pprint # Import for pretty printing

# Keep fast imports at the top
from src.services.knowledge_graph.parser.parser import Parser
from src.db.graph.models import Node, Edge, NodeType, EdgeType, GraphPayload
class NLP_Parser(Parser):

    def __init__(self, topic_labels: list[str]):
        """
        Initialize the NLP parser with Flair's NER model.
        The slow imports are deferred to this method.
        """
        # --- SLOW IMPORTS MOVED HERE ---
        from flair.data import Sentence
        from flair.models import SequenceTagger
        
        print("  -> Loading Flair SequenceTagger model... (this may take a moment)")
        # Load the pre-trained NER model. This is a one-time operation.
        self.tagger = SequenceTagger.load("ner-large")
        self.Sentence = Sentence # Store the class for later use
        
        self.topic_labels = topic_labels

        # Mapping of common verbs to your ontology's edge types
        self.relation_mapping = {
            "introduce": "INTRODUCED",
            "introduced": "INTRODUCED",
            "author": "AUTHORED",
            "authored": "AUTHORED",
            "teach": "MENTOR_OF",
            "taught": "MENTOR_OF",
            "live": "LIVED_IN",
            "lived": "LIVED_IN",
            "propose": "PROPOSED",
            "proposed": "PROPOSED",
            "influence": "INFLUENCED",
            "influenced": "INFLUENCED",
        }

    def parse_text(self, text: str, graph_id: str, split_sentences: bool = False) -> dict:
        """
        Convert raw text into a graph structure using Flair for NER.
        """
        flair_sentence = self.Sentence(text)
        self.tagger.predict(flair_sentence)

        # 1. Extract nodes from the identified entities, letting Pydantic create the ID
        nodes = self._extract_nodes(flair_sentence)
        # Create a map lookup for title -> Node object for easy access
        node_map_by_title = {node.title: node for node in nodes}

        # 2. Extract edges by analyzing the text between entities
        edges = self._extract_edges(flair_sentence, node_map_by_title)

        # Use a dictionary to ensure final nodes are unique by their canonical ID
        final_nodes = {node.id: node for node in nodes}

        return {
            "nodes": [n.model_dump() for n in final_nodes.values()],
            "edges": [e.model_dump(by_alias=True) for e in edges],
        }

    def _extract_nodes(self, flair_sentence) -> list['Node']:
        """Extract nodes from Flair's entity spans."""
        nodes = []
        for entity in flair_sentence.get_spans('ner'):
            node_type = entity.get_label("ner").value
            if node_type == "PER":
                node_type = "PERSON"
            elif node_type == "LOC":
                node_type = "LOCATION"
            
            # CORRECTED: Provide id=None so the Pydantic validator can process it.
            nodes.append(Node(id=None, title=entity.text, type=node_type))
        return nodes

    def _extract_edges(self, flair_sentence, node_map_by_title: dict[str, 'Node']) -> list['Edge']:
        """
        A more robust rule-based method to find edges.
        This tokenizes the text between entities and checks each word.
        """
        edges = []
        entities = flair_sentence.get_spans('ner')
        
        for i in range(len(entities) - 1):
            source_entity = entities[i]
            target_entity = entities[i+1]

            start_pos = source_entity.end_pos
            end_pos = target_entity.start_pos
            text_between = flair_sentence.text[start_pos:end_pos].lower()
            
            words_between = text_between.split()

            for word in words_between:
                cleaned_word = ''.join(filter(str.isalnum, word))
                
                if cleaned_word in self.relation_mapping:
                    rel_type = self.relation_mapping[cleaned_word]
                    
                    # Look up the full Node object from the map
                    source_node = node_map_by_title.get(source_entity.text)
                    target_node = node_map_by_title.get(target_entity.text)

                    if source_node and target_node:
                        # Pass the raw titles to the Edge constructor
                        # Pydantic's validator will handle canonicalization
                        edges.append(
                            Edge(
                                from_=source_node.title,
                                to=target_node.title,
                                type=rel_type
                            )
                        )
                        break 
        return edges

    def parse_multiple_texts(self, texts: list[str], graph_id: str, chunk_id: str, document_id: str, split_sentences: bool = True) -> dict:
        """Aggregates nodes and edges from several texts."""
        merged_nodes: dict[str, 'Node'] = {}
        edge_seen: set[tuple[str, str, str]] = set()
        merged_edges: list['Edge'] = []
        
        for text in texts:
            graph = self.parse_text(text, graph_id=graph_id)
            for node_dict in graph["nodes"]:
                if node_dict["id"] not in merged_nodes:
                    merged_nodes[node_dict["id"]] = Node(**node_dict)
            
            for edge_dict in graph["edges"]:
                key = (edge_dict["from"], edge_dict["to"], edge_dict["type"])
                if key not in edge_seen:
                    # Use by_alias=True if the dict keys are 'from' instead of 'from_'
                    merged_edges.append(Edge.model_validate(edge_dict))
                    edge_seen.add(key)
        
        return {
            "graph_id": graph_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "nodes": [n.model_dump() for n in merged_nodes.values()],
            "edges": [e.model_dump(by_alias=True) for e in merged_edges],
        }

# ==========================================================
# Main execution block to test the parser
# ==========================================================
if __name__ == "__main__":
    # --- For standalone testing, define the necessary Pydantic models and helpers ---
    from pydantic import BaseModel, Field, ConfigDict, field_validator
    import re

    def canon(text: str) -> str:
        """A simple placeholder for your canonicalization/slugify utility."""
        text = text.lower()
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'[^a-z0-9_]', '', text)
        return text


    # This print statement will now run instantly
    print("Script started. Initializing NLP_Parser...")
    
    # The slow part happens here, when the class is initialized
    parser = NLP_Parser(topic_labels=["BIBLICAL_TEXT"])
    print("Parser initialized successfully.")

    # Define the sample text
    test_text = """
    Now Lot, who was moving about with Abram, also had flocks and herds and tents. 
    But the land could not support them while they stayed together, for their possessions 
    were so great that they were not able to stay together. And quarreling arose 
    between Abram's herders and Lot's herders. So Abram said to Lot, 'Let’s not 
    have any quarreling between you and me.' Abram lived in the land of Canaan, 
    while Lot lived among the cities of the plain and pitched his tents near Sodom.
    """

    # Parse the text
    print("\n--- Parsing Text ---")
    print(f"Input Text: \"{test_text.strip()}\"")
    
    extracted_graph = parser.parse_text(test_text, graph_id="genesis_13")

    # Print the final results
    print("\n--- Final Extracted Graph ---")
    pprint.pprint(extracted_graph)
    print("-------------------------------------\n")
