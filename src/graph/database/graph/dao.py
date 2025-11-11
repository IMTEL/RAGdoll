from abc import ABC, abstractmethod
import os
from neo4j import GraphDatabase as Neo4jDriver

from src.db.graph.enums import GraphDatabaseType
from src.db.graph.models import GraphPayload, GraphStructure
from src.db.models import Status
from src.models.graphs import Node, Edge, Graph

class GraphDatabase(ABC):
    """
    Abstract class for Connecting to a Database
    """
    @classmethod
    def __instancecheck__(cls, instance: any) -> bool:
        return cls.__subclasscheck__(type(instance))

    @classmethod
    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, GraphDatabase)
    
    @abstractmethod
    def close(self):
        """
        Close the database connection.
        """
        pass
    
    @abstractmethod
    def is_reachable(self) -> bool:
        """
        Check if the database is reachable.

        Returns:
            bool: True if the database is reachable, False otherwise.
        """
        pass

    @abstractmethod
    def post_node(self, node: Node, graph_id: str, chunk_id: str, document_id: str) -> Status:
        """
        Posts a node to the graph database.

        Args:
            node (Node): The node to be posted.

        Returns:
            Status: The status of the operation.
        """
        pass

    @abstractmethod
    def post_edge(self, edge: Edge, graph_id: str, chunk_id: str) -> Status:
        """
        Posts an edge to the graph database.

        Args:
            edge (Edge): The edge to be posted.

        Returns:
            Status: The status of the operation.
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Node | None:
        """
        Retrieves a node from the graph database by its ID.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            Node: A Node object representing the retrieved node.
        """
        pass

    @abstractmethod
    def get_edge(self, from_id: str, to_id: str, rel_type: str) -> Edge | None:
        """
        Retrieves an edge from the graph database by its endpoints and type.

        Args:
            from_id (str): ID of the source node.
            to_id (str): ID of the target node.
            rel_type (str): The relationship type.

        Returns:
            Edge: An Edge object representing the retrieved edge.
        """
        pass

    @abstractmethod
    def post_graph(self, payload: GraphPayload) -> Status:
        """
        Posts a graph to the database.

        Args:
            payload (GraphPayload): The payload containing nodes and edges to be posted.

        Returns:
            Status: The status of the operation.
        """
        pass

    @abstractmethod
    def get_graph(self, graph_id: str) -> GraphStructure:
        """
        Retrieves the full subgraph (all nodes & edges) for the given graph_id.
        """
        pass
    
    @abstractmethod
    def get_topic_labels(self, graph_id: str) -> list[str]:
        """
        Retrieves the list of existing topic labels for nodes.
        """
        pass
    
    @abstractmethod
    def get_edge_labels(self, graph_id: str) -> list[str]:
        """
        Retrieves the list of existing edge labels for relationships.
        """
        pass
    
    @staticmethod
    def get_node_titles(self, graph_id: str) -> list[dict]:
        """Retrieves all existing node titles in the graph.
        Args:
            graph_id (str)
        Returns:
            list[dict]: _description_
        """

    @abstractmethod
    def query(self, cypher: str, params: dict[str, any]) -> list[dict[str, any]]:
        """
        Run an arbitrary Cypher query and return a list of dicts.
        """
        pass


class Neo4jGraphDatabase(GraphDatabase):
    """
    Neo4j-backed Database that uses Node, Edge, and GraphPayload models.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver = Neo4jDriver.driver(uri, auth=(user, password))

    def close(self):
        if self._driver:
            self._driver.close()

    def is_reachable(self) -> bool:
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    def post_node(self, node: Node, graph_id: str, chunk_id: str, document_id: str) -> Status:
        props = {"id": node.id, "graph_id": graph_id}
        if node.title:
            props["title"] = node.title
        props.update(node.properties or {})
        with self._driver.session() as session:
            session.execute_write(self._create_node_tx, node.type, props, chunk_id, document_id)
        return Status(status="success", detail=f"Node '{node.id}' upserted in graph {graph_id}, chunk {chunk_id}")

    @staticmethod
    def _create_node_tx(tx, label: str, props: dict, chunk_id: str, document_id: str):
        cypher = f"""
            MERGE (n {{id: $id, graph_id: $graph_id}})
            ON CREATE
            SET n:{label},
                n += $props,
                n.chunk_ids = [$chunk_id],
                n.document_ids = [$document_id]
            ON MATCH
            SET n += $props,
                n.chunk_ids =
                    CASE
                    WHEN NOT $chunk_id IN coalesce(n.chunk_ids, []) THEN coalesce(n.chunk_ids, []) + $chunk_id
                    ELSE n.chunk_ids
                    END,
                n.document_ids =
                    CASE
                    WHEN NOT $document_id IN coalesce(n.document_ids, []) THEN coalesce(n.document_ids, []) + $document_id
                    ELSE n.document_ids
                    END
        """
        tx.run(
            cypher,
            id=props["id"],
            graph_id=props["graph_id"],
            props=props,
            chunk_id=chunk_id,
            document_id=document_id
        )

    def post_edge(self, edge: Edge, graph_id: str, chunk_id: str) -> Status:
        with self._driver.session() as session:
            session.execute_write(
                self._create_edge_tx,
                edge.from_, edge.to, edge.type, graph_id, chunk_id
            )
        return Status(status="success", detail=f"Edge '{edge.type}' in graph {graph_id}, chunk {chunk_id}")

    @staticmethod
    def _create_edge_tx(tx, from_id: str, to_id: str, rel_type: str, graph_id: str, chunk_id: str):
        cypher = f"""
        MATCH (a {{id: $from_id, graph_id: $graph_id}}),
            (b {{id: $to_id,   graph_id: $graph_id}})
        MERGE (a)-[r:{rel_type} {{graph_id: $graph_id}}]->(b)
        ON CREATE
        SET r.chunk_ids = [$chunk_id]
        ON MATCH
        SET r.chunk_ids =
            CASE
            WHEN NOT $chunk_id IN coalesce(r.chunk_ids, []) THEN coalesce(r.chunk_ids, []) + $chunk_id
            ELSE r.chunk_ids
            END
        """
        tx.run(
            cypher,
            from_id=from_id,
            to_id=to_id,
            graph_id=graph_id,
            chunk_id=chunk_id
        )

    def get_node(self, node_id: str) -> Node | None:
        with self._driver.session() as session:
            result = session.execute_read(self._get_node_tx, node_id)
            if result:
                return Node(**result)
            return None

    @staticmethod
    def _get_node_tx(tx, node_id: str):
        cypher = "MATCH (n {id: $node_id}) RETURN n"
        rec = tx.run(cypher, node_id=node_id).single()
        if not rec:
            return None
        n = rec["n"]
        label = next(iter(n.labels)) if n.labels else ""
        return {
            "id": n["id"],
            "title": n.get("title", ""),
            "type": label,
            "properties": {
                k: v
                for k, v in n.items()
                if k not in ("id", "title", "graph_id")
            },
            "chunk_ids": n.get("chunk_ids", [])
        }

    def get_edge(self, from_id: str, to_id: str, rel_type: str) -> Edge | None:
        with self._driver.session() as session:
            result = session.execute_read(
                self._get_relationship_tx, from_id, to_id, rel_type
            )
            if result:
                return Edge(**result)
            return None

    @staticmethod
    def _get_relationship_tx(tx, from_id: str, to_id: str, rel_type: str):
        cypher = (
            "MATCH (a {id: $from_id})-[r]->(b {id: $to_id})\n"
            "WHERE type(r) = $rel_type\n"
            "RETURN a.id AS from, b.id AS to, type(r) AS type"
        )
        rec = tx.run(cypher, from_id=from_id, to_id=to_id, rel_type=rel_type).single()
        if not rec:
            return None
        return {
            "from": rec["from"],
            "to": rec["to"],
            "type": rec["type"]
        }

    def post_graph(self, payload: GraphPayload) -> Status:
        for n in payload.nodes:
            self.post_node(n, payload.graph_id, payload.chunk_id, payload.document_id)
        for e in payload.edges:
            self.post_edge(e, payload.graph_id, payload.chunk_id)
        return Status(status="success", detail=f"Graph {payload.graph_id} populated")

    def get_graph(self, graph_id: str) -> GraphStructure:
        with self._driver.session() as session:
            nodes = session.execute_read(self._fetch_nodes_tx, graph_id)
            edges = session.execute_read(self._fetch_edges_tx, graph_id)
        return GraphStructure(graph_id=graph_id, nodes=nodes, edges=edges)

    def get_topic_labels(self, graph_id: str) -> list[str]:
        """
        Return every distinct label attached to ANY node that belongs to `graph_id`.
        """
        with self._driver.session() as session:
            return session.execute_read(self._topic_labels_tx, graph_id)

    def get_edge_labels(self, graph_id: str) -> list[str]:
        """
        Return every distinct relationship-type used inside `graph_id`.
        """
        with self._driver.session() as session:
            return session.execute_read(self._edge_labels_tx, graph_id)
        
    @staticmethod
    def _topic_labels_tx(tx, graph_id: str) -> list[str]:
        q = (
            "MATCH (n) "
            "WHERE n.graph_id = $graph_id "
            "UNWIND labels(n) AS label "
            "RETURN DISTINCT label"
        )
        return [record["label"] for record in tx.run(q, graph_id=graph_id)]

    @staticmethod
    def _edge_labels_tx(tx, graph_id: str) -> list[str]:
        q = (
            "MATCH ()-[r]->() "
            "WHERE r.graph_id = $graph_id "
            "RETURN DISTINCT type(r) AS rel"
        )
        return [record["rel"] for record in tx.run(q, graph_id=graph_id)]
        
    # TODO: Performance note on indexing
    """
    Add an index on graph_id for both nodes and relationships once:

    CREATE INDEX node_graph_id IF NOT EXISTS FOR (n) ON (n.graph_id);
    CREATE INDEX rel_graph_id  IF NOT EXISTS FOR ()-[r]-() ON (r.graph_id);

    (Neo4j 5 syntax shown; in Neo4j 4 use CREATE INDEX ... FOR (n:YourLabel).)

    Each query touches only the index + a tiny projection (labels or
    type(r)), so it remains millisecond-fast even with millions of nodes.
    
    for node titles, we can also create an index on the title property
    CREATE INDEX IF NOT EXISTS node_title_by_graph
    FOR (n) ON (n.graph_id, n.title);
    
    """
    
    def get_node_titles(self, graph_id: str) -> list[dict]:
        """
        Return every existing node title inside `graph_id`.

        Output schema:
            [
              {"title": "Euler",           "label": "PERSON", "id": "node-123"},
              {"title": "Leonhard Euler",  "label": "PERSON", "id": "node-123"},
              {"title": "Calculus",        "label": "TOPIC",  "id": "node-456"},
              ...
            ]

        - Multiple titles can point to the same `id` (aliases).
        - Nodes without a `title` property are ignored.
        """
        with self._driver.session() as sess:
            return sess.execute_read(self._titles_tx, graph_id)
        
    @staticmethod
    def _titles_tx(tx, graph_id: str) -> list[dict]:
        q = (
            "MATCH (n) "
            "WHERE n.graph_id = $graph_id AND n.title IS NOT NULL "
            "RETURN n.title AS title, labels(n)[0] AS label, n.id AS id"
        )
        return [
            {"title": r["title"], "label": r["label"], "id": r["id"]}
            for r in tx.run(q, graph_id=graph_id)
        ]
    
    @staticmethod
    def _fetch_nodes_tx(tx, graph_id: str):
        q = "MATCH (n) WHERE n.graph_id = $graph_id RETURN n"
        result = tx.run(q, graph_id=graph_id)
        out = []
        for rec in result:
            n = rec["n"]
            label = next(iter(n.labels)) if n.labels else ""
            props = {k: v for k, v in n.items() if k not in ("id", "title", "graph_id", "chunk_ids")}
            out.append({"id": n["id"], "type": label, "title": n.get("title",""), "properties": props, "chunk_ids": n.get("chunk_ids", [])})
        return out

    @staticmethod
    def _fetch_edges_tx(tx, graph_id: str):
        q = """
        MATCH (a)-[r]->(b)
        WHERE r.graph_id = $graph_id
        RETURN
          a.id AS from,
          b.id AS to,
          type(r) AS type,
          r.chunk_ids AS chunk_ids
        """
        result = tx.run(q, graph_id=graph_id)
        return [{"from": r["from"], "to": r["to"], "type": r["type"], "chunk_ids": r["chunk_ids"] or []} for r in result]
    
    def query(self, cypher: str, params: dict[str, any] = None) -> list[dict[str, any]]:
        """
        Execute a read-only Cypher query and return each row as a dict.
        """
        params = params or {}
        with self._driver.session() as session:
            return session.execute_read(self._run_query_tx, cypher, params)
        
    def execute_write(self, cypher: str, params: dict[str, any] = None) -> list[dict[str, any]]:
        """
        Execute a write Cypher query and return each row as a dict.
        """
        params = params or {}
        with self._driver.session() as session:
            return session.execute_write(self._run_query_tx, cypher, params)

    @staticmethod
    def _run_query_tx(tx, cypher: str, params: dict[str, any]) -> list[dict[str, any]]:
        result = tx.run(cypher, params)
        return [record.data() for record in result]


def create_graph_database(
    db_system: GraphDatabaseType,
    uri=None, user=None, password=None
) -> GraphDatabase:
    match db_system:
        case GraphDatabaseType.NEO4J:
            return Neo4jGraphDatabase(uri=uri, user=user, password=password)
        case _:
            raise ValueError(f"Unsupported database system: {db_system}")
