
from __future__ import annotations

import json, logging
from typing import Iterable, Tuple, List, Dict, Set

import nltk
import spacy
from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
from slugify import slugify

from src.models import Node, Edge, GraphPayload, NodeType, EdgeType

import re

try:
    from allennlp.predictors.predictor import Predictor
    _OPENIE_PREDICTOR = Predictor.from_path(
        # 50 MB, downloaded/cached automatically in ~/.allennlp/
        "https://storage.googleapis.com/allennlp-public-models/"
        "openie-model.2020.03.26.tar.gz",
        cuda_device=-1,          # CPU
        quiet=True,
    )
    logging.info("[Setup] AllenNLP OpenIE ✓ loaded")
except Exception as e:
    _OPENIE_PREDICTOR = None
    logging.info("[Setup] AllenNLP OpenIE unavailable – %s", e)


# Simple regex that pulls   ARG0   V   ARG1   (subject-verb-object) triples
_TRIPLE_RE = re.compile(
    r"\[ARG0: (?P<subj>[^\]]+?)\]\s+\[V: (?P<verb>[^\]]+?)\]\s+\[ARG1: (?P<obj>[^\]]+?)\]"
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")

# Ensure the required NLTK resources are available (they are tiny < 1 MB)
for _res in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_res}")
    except LookupError:
        logging.info("[Setup] downloading NLTK resource '%s'…", _res)
        nltk.download(_res, quiet=True)

# Load spaCy English pipeline (might be possible to swap for domain‑specific model if needed)
_nlp = spacy.load("en_core_web_sm", disable=["ner"])
_ner = spacy.load("en_core_web_sm", enable=["ner"])   # a tiny cost per doc


def canonical_id(text: str) -> str:
    """Return lower_snake_case slug stable across runs."""
    return slugify(text, separator="_")

# sentence splitting 

def sent_tokenize(text: str) -> List[str]:
    """Split *text* into sentence‑like chunks.

    Falls back to spaCy’s sentence segmentation if NLTK fails for any reason.
    """
    try:
        return _nltk_sent_tokenize(text)
    except (LookupError, OSError):
        return [s.text for s in _nlp(text).sents]

# node extraction 
_ALLOWED_NODE_LABELS: Set[str] = {e.name for e in NodeType}
_ALLOWED_EDGE_LABELS: Set[str] = {e.name for e in EdgeType}

# Map spaCy entity labels → your NodeType enum names
_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",               
    "ORGANIZATION": "ORG",      
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK",
}


def build_nodes(doc: spacy.tokens.Doc) -> dict[str, Node]:
    nodes: dict[str, Node] = {}
    # Use dedicated NER pipeline (faster if we disable other components)
    ner_doc = _ner(doc.text)
    for ent in ner_doc.ents:
        raw_label = ent.label_
        mapped = _LABEL_MAP.get(raw_label, raw_label)
        if mapped not in _ALLOWED_NODE_LABELS:
            logging.debug("[Ontology] skipping unrecognised node label '%s'", mapped)
            continue  # → **skip** instead of throwing ValidationError
        nid = canonical_id(ent.text)
        nodes.setdefault(
            nid,
            Node(
                id=nid,
                type=mapped,  # the enum validator accepts the string name
                title=ent.text,
                properties={},
            ),
        )
    return nodes

# relation extraction (pattern‑based demo) 

_RELATION_TEMPLATES = {
    "LED":        {"lemmas": {"lead", "led"}},
    "FATHER_OF":  {"lemmas": {"father", "beget", "begot"}},
    "DEFEATED":   {"lemmas": {"defeat", "slay", "kill"}},
    "BUILT":      {"lemmas": {"build", "construct"}},
    "ESTABLISHED": {"lemmas": {"establish", "found", "set"}},
    "UNIFIED":    {"lemmas": {"unite", "unify"}},
    "PARTED":     {"lemmas": {"part", "divide", "split"}},
    "ASKED":      {"lemmas": {"ask", "request"}},
    "BECAME":     {"lemmas": {"become", "became"}},
    # keep the originals too
    "INTRODUCED": {"lemmas": {"introduce", "propose", "present"}},
    "AUTHORED":   {"lemmas": {"author", "write"}},
}


def _fallback_edge(doc: spacy.tokens.Doc,
                   nodes: dict[str, Node]) -> Iterable[Edge]:
    """
    If no template fires, take the first verb that has a subject + object and
    create an edge named after the verb lemma (slug-uppercased).
    """
    for t in doc:
        if t.pos_ != "VERB":
            continue
        sbj = next((c for c in t.children
                    if c.dep_ in {"nsubj", "nsubjpass"}), None)
        obj = next((c for c in t.children
                    if c.dep_ in {"dobj", "attr", "oprd", "pobj"}), None)
        if sbj and obj:
            label = slugify(t.lemma_, separator="_").upper()  # e.g. LED, BUILT
            s_id, o_id = canonical_id(sbj.text), canonical_id(obj.text)
            if s_id in nodes and o_id in nodes:
                # allow any label by prefixing NEW_ if it’s not whitelisted
                if label not in _ALLOWED_EDGE_LABELS:
                    label = f"NEW_{label}"
                yield Edge(from_=s_id, to=o_id, type=label)
            break   # one per sentence keeps noise down


def pattern_relations(doc: spacy.tokens.Doc, nodes: dict[str, Node]) -> Iterable[Edge]:
    """Yield edges recognised by simple dependency‑pattern rules."""
    for token in doc:
        if token.pos_ != "VERB":
            continue
        for edge_type, tpl in _RELATION_TEMPLATES.items():
            if token.lemma_ not in tpl["lemmas"]:
                continue
            subj = next((c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
            obj  = next((c for c in token.children if c.dep_ in {"dobj", "attr", "oprd"}), None)
            if subj and obj:
                s_id, o_id = canonical_id(subj.text), canonical_id(obj.text)
                if s_id in nodes and o_id in nodes:
                    yield Edge(from_=s_id, to=o_id, type=edge_type)


def openie_relations(text: str) -> Iterable[tuple[str, str, str]]:
    """
    Yields (subject, relation_phrase, object) triples extracted by AllenNLP OpenIE.
    Falls back to the empty iterator if the model isn’t available.
    """
    if not _OPENIE_PREDICTOR:
        return []                 # pattern & verb-fallback will still run

    try:
        pred = _OPENIE_PREDICTOR.predict(sentence=text)
    except Exception as err:
        logging.debug("[OpenIE] predictor failed on sentence – %s", err)
        return []

    for verb_info in pred.get("verbs", []):
        for m in _TRIPLE_RE.finditer(verb_info["description"]):
            yield m.group("subj"), m.group("verb"), m.group("obj")


def build_edges(doc: spacy.tokens.Doc, nodes: dict[str, Node]) -> list[Edge]:
    edges: list[Edge] = list(pattern_relations(doc, nodes))
    # ––– OpenIE fallback –––
    for arg1, rel, arg2 in openie_relations(doc.text):
        edge_type = canonical_id(rel).upper()
        if edge_type not in _ALLOWED_EDGE_LABELS:
            logging.debug("[Ontology] skipping unrecognised edge label '%s'", edge_type)
            continue
        a1, a2 = canonical_id(arg1), canonical_id(arg2)
        if a1 in nodes and a2 in nodes:
            edges.append(Edge(from_=a1, to=a2, type=edge_type))
    if not edges:                       # only if templates gave nothing
        edges.extend(_fallback_edge(doc, nodes))
    return edges


def parse_text(text: str, split_sentences: bool = False) -> dict:
    """Convert *text* into the {nodes:[…], edges:[…]} structure."""
    sentences = sent_tokenize(text) if split_sentences else [text]

    node_map: dict[str, Node] = {}
    edges: list[Edge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for sent in sentences:
        doc = _nlp(sent)
        nodes = build_nodes(doc)
        node_map.update(nodes)
        for e in build_edges(doc, node_map):
            sig = (e.from_, e.to, e.type)
            if sig not in seen_edges:
                seen_edges.add(sig)
                edges.append(e)

    payload = GraphPayload(nodes=list(node_map.values()), edges=edges)
    return payload.model_dump(exclude_none=True, by_alias=True)


def parse_multiple_texts(texts: list[str], split_sentences: bool = True) -> dict:
    merged_nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    seen_edges: Set[tuple[str, str, str]] = set()

    for chunk in texts:
        g = parse_text(chunk, split_sentences=split_sentences)
        for n in g["nodes"]:
            merged_nodes.setdefault(n["id"], Node(**n))
        for e in g["edges"]:
            sig = (e["from"], e["to"], e["type"])
            if sig not in seen_edges:
                seen_edges.add(sig)
                edges.append(Edge(**e))

    payload = GraphPayload(nodes=list(merged_nodes.values()), edges=edges)
    return payload.model_dump(exclude_none=True, by_alias=True)

# Self‑test (run python -m src.nlp_parser)
if __name__ == "__main__":
    biblical_texts = [
        "Abraham left the city of Ur, believing Yahweh’s promise that his descendants would become a great nation. According to Genesis, he showed unshakable faith when he almost sacrificed his son Isaac on Mount Moriah, whereupon a ram was provided as a substitute. For instance in Genesis 22:12, God says, 'Now I know that you fear God, because you have not withheld from me your son, your only son.' This event is foundational in Judaism, symbolizing faith and obedience to God’s will.",
        "Moses, raised in Pharaoh’s palace, confronted Ramses II with ten plagues and then led the Exodus. At the shore of the Red Sea he stretched out his staff; the waters parted, allowing the Israelites to cross on dry ground before crashing back upon the pursuing Egyptian chariots. This event is central to Jewish identity, symbolizing liberation and divine intervention, and is commemorated in the Passover festival.",
        "Joshua succeeded Moses and, after circling Jericho’s walls for seven days with priests blowing ram’s horns, gave the shout that caused the walls to collapse. The conquest opened Canaan to the twelve tribes. Before his death, he famously declared, 'As for me and my house, we will serve the Lord.' This established the Israelites in the Promised Land, and Joshua’s leadership was pivotal in fulfilling the covenant promise to Abraham.",
        "David, a shepherd from Bethlehem, refused conventional armor and met the Philistine giant Goliath with only a sling and five smooth stones. His first shot struck Goliath’s forehead, winning a decisive victory that later paved his path to kingship. Importantly for the Jewish faith, David united the tribes and established Jerusalem as the capital, bringing the Ark of the Covenant there.",
        "King Solomon asked God for wisdom rather than riches. His famous judgment between two women claiming the same infant—threatening to divide the baby—revealed the true mother and solidified his reputation for discernment. Additionally, he built the First Temple in Jerusalem, a symbol of Israel’s covenant with Yahweh.",
        "The prophet Elijah challenged 450 prophets of Baal on Mount Carmel. After they failed to call down fire, Elijah soaked his own altar with water and prayed; fire descended, consuming the sacrifice, stones, and water, proving Yahweh’s supremacy. Even the rain followed, ending a three-year drought.",
        "Saul of Tarsus, once a persecutor of early Christians, encountered a blinding light on the road to Damascus. Hearing Jesus’ voice, he converted, became Paul the Apostle, and carried the gospel through Asia Minor and to Rome, writing epistles that form much of the New Testament. This transformation from persecutor to apostle exemplifies the power of faith and redemption.",
    ]        # see list below
    #out = parse_text(sample, split_sentences=True)
    out = parse_multiple_texts(biblical_texts, split_sentences=True)
    from pprint import pprint
    pprint(out, indent=2)
