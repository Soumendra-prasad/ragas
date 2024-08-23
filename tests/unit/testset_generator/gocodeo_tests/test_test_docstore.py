import os
import pickle
import typing as t

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from langchain.text_splitter import TokenTextSplitter
from langchain_core.embeddings import Embeddings

from ragas.testset.docstore import InMemoryDocumentStore, Node

@pytest.fixture
def test_setup():
    with patch('os.path.join', return_value='mocked_path'), \
         patch('builtins.open', new_callable=MagicMock), \
         patch('pickle.load', return_value={
             "cat": [0.1] * 768,
             "mouse": [0.2] * 768,
             "solar_system": [0.3] * 768
         }), \
         patch('langchain.text_splitter.TokenTextSplitter', autospec=True), \
         patch('ragas.testset.docstore.InMemoryDocumentStore', autospec=True):
        
        class FakeEmbeddings(Embeddings):
            def __init__(self):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_embs.pkl")
                with open(path, "rb") as f:
                    self.embeddings: dict[str, t.Any] = pickle.load(f)

            def _get_embedding(self, text: str) -> t.List[float]:
                if text in self.embeddings:
                    return self.embeddings[text]
                else:
                    return [0] * 768

            def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
                return [self._get_embedding(text) for text in texts]

            def embed_query(self, text: str) -> t.List[float]:
                return self._get_embedding(text)

            async def aembed_query(self, text: str) -> t.List[float]:
                return self._get_embedding(text)

        yield FakeEmbeddings, TokenTextSplitter, InMemoryDocumentStore, Node

# happy_path - test_adjacent_nodes - Test adjacent nodes are set correctly in the store
def test_adjacent_nodes(test_setup):
    FakeEmbeddings, TokenTextSplitter, InMemoryDocumentStore, Node = test_setup
    a1 = Node(doc_id="a1", page_content="a1", metadata={"filename": "a"})
    a2 = Node(doc_id="a2", page_content="a2", metadata={"filename": "a"})
    b = Node(doc_id="b", page_content="b", metadata={"filename": "a"})

    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    store.nodes = [a1, a2, b]
    store.set_node_relataionships()

    assert store.nodes[0].next == a2
    assert store.nodes[1].prev == a1
    assert store.nodes[2].next is None

# edge_case - test_adjacent_nodes - Test adjacent nodes with no nodes in the store
def test_adjacent_nodes_empty(test_setup):
    FakeEmbeddings, TokenTextSplitter, InMemoryDocumentStore, Node = test_setup

    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    store.nodes = []
    store.set_node_relataionships()

    assert len(store.nodes) == 0

