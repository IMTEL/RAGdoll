## How to visualize the graph

Go to http://localhost:7474, login, and run the query:

```bash
MATCH p = (n)-[r]->(m) RETURN p;

MATCH p = (n)-[r]->(m) RETURN p LIMIT 100;

# or, to see a specific graph (replace 'ccf00233-88a7-41cd-8877-e4324fc317c2' with your graph_id):

MATCH (n {graph_id: 'ccf00233-88a7-41cd-8877-e4324fc317c2'})
WITH n
MATCH p = (n)-[r]->(m)
WHERE r.graph_id = n.graph_id AND m.graph_id = n.graph_id
RETURN p;
```

Adjust "LIMIT 100" for larger graphs.