import pytest
from unittest.mock import MagicMock, patch
from ragas.executor import Executor
from ragas.run_config import RunConfig
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.exceptions import ExceptionInRunner
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document as LCDocument
from ragas.testset.extractor import Extractor
from numpy.typing import NDArray
import numpy as np
import numpy.typing as npt

@pytest.fixture
def setup_mocks():
    with patch('ragas.executor.Executor') as mock_executor, \
         patch('ragas.run_config.RunConfig') as mock_run_config, \
         patch('ragas.embeddings.base.BaseRagasEmbeddings') as mock_embeddings, \
         patch('ragas.exceptions.ExceptionInRunner') as mock_exception_in_runner, \
         patch('langchain.text_splitter.TextSplitter') as mock_text_splitter, \
         patch('langchain_core.documents.Document') as mock_lcdocument, \
         patch('ragas.testset.extractor.Extractor') as mock_extractor, \
         patch('numpy.typing.NDArray') as mock_ndarray, \
         patch('numpy') as mock_numpy, \
         patch('uuid.uuid4') as mock_uuid4, \
         patch('logging.getLogger') as mock_get_logger:

        mock_executor_instance = mock_executor.return_value
        mock_run_config_instance = mock_run_config.return_value
        mock_embeddings_instance = mock_embeddings.return_value
        mock_exception_in_runner_instance = mock_exception_in_runner.return_value
        mock_text_splitter_instance = mock_text_splitter.return_value
        mock_lcdocument_instance = mock_lcdocument.return_value
        mock_extractor_instance = mock_extractor.return_value
        mock_ndarray_instance = mock_ndarray.return_value
        mock_numpy_instance = mock_numpy.return_value
        mock_uuid4_instance = mock_uuid4.return_value
        mock_logger_instance = mock_get_logger.return_value

        yield {
            "mock_executor": mock_executor_instance,
            "mock_run_config": mock_run_config_instance,
            "mock_embeddings": mock_embeddings_instance,
            "mock_exception_in_runner": mock_exception_in_runner_instance,
            "mock_text_splitter": mock_text_splitter_instance,
            "mock_lcdocument": mock_lcdocument_instance,
            "mock_extractor": mock_extractor_instance,
            "mock_ndarray": mock_ndarray_instance,
            "mock_numpy": mock_numpy_instance,
            "mock_uuid4": mock_uuid4_instance,
            "mock_logger": mock_logger_instance,
        }

# happy_path - filename - Test that filename property retrieves the correct filename from metadata
def test_filename_retrieval(setup_mocks):
    mock_logger = setup_mocks['mock_logger']
    document = Document(metadata={'filename': 'test_file.txt'})
    filename = document.filename
    assert filename == 'test_file.txt'
    mock_logger.info.assert_not_called()

# happy_path - filename - Test that filename property defaults to doc_id when filename is not in metadata
def test_filename_defaults_to_doc_id(setup_mocks):
    mock_logger = setup_mocks['mock_logger']
    document = Document(metadata={}, doc_id='1234')
    filename = document.filename
    assert filename == '1234'
    mock_logger.info.assert_called_once()

# happy_path - from_langchain_document - Test from_langchain_document creates Document with correct metadata and page_content
def test_from_langchain_document(setup_mocks):
    mock_uuid4 = setup_mocks['mock_uuid4']
    mock_uuid4.return_value = 'unique-id'
    doc = LCDocument(page_content='Test content', metadata={'author': 'John Doe'})
    document = Document.from_langchain_document(doc)
    assert document.page_content == 'Test content'
    assert document.metadata == {'author': 'John Doe'}
    assert document.doc_id == 'unique-id'

# happy_path - from_llamaindex_document - Test from_llamaindex_document creates Document with correct metadata and text
def test_from_llamaindex_document(setup_mocks):
    mock_uuid4 = setup_mocks['mock_uuid4']
    mock_uuid4.return_value = 'unique-id'
    doc = LlamaindexDocument(text='Llama content', metadata={'source': 'Llama'})
    document = Document.from_llamaindex_document(doc)
    assert document.page_content == 'Llama content'
    assert document.metadata == {'source': 'Llama'}
    assert document.doc_id == 'unique-id'

# happy_path - __eq__ - Test that __eq__ returns True for documents with same doc_id
def test_eq_same_doc_id(setup_mocks):
    doc1 = Document(doc_id='abcd')
    doc2 = Document(doc_id='abcd')
    assert doc1 == doc2

# happy_path - next - Test that next property retrieves the correct next node relationship
def test_next_relationship(setup_mocks):
    node = Node(relationships={Direction.NEXT: 'node_2'})
    assert node.next == 'node_2'

# happy_path - prev - Test that prev property retrieves the correct previous node relationship
def test_prev_relationship(setup_mocks):
    node = Node(relationships={Direction.PREV: 'node_1'})
    assert node.prev == 'node_1'

# happy_path - add_documents - Test add_documents correctly splits and adds documents as nodes
def test_add_documents(setup_mocks):
    mock_embeddings = setup_mocks['mock_embeddings']
    mock_splitter = setup_mocks['mock_text_splitter']
    mock_splitter.transform_documents.return_value = [LCDocument(page_content='Node1 content', metadata={})]
    store = InMemoryDocumentStore(splitter=mock_splitter, embeddings=mock_embeddings)
    store.add_documents([LCDocument(page_content='Doc1 content', metadata={})])
    assert len(store.nodes) == 1
    assert store.nodes[0].page_content == 'Node1 content'

# happy_path - add_nodes - Test add_nodes correctly handles nodes without embeddings and keyphrases
def test_add_nodes_without_embeddings(setup_mocks):
    mock_embeddings = setup_mocks['mock_embeddings']
    mock_extractor = setup_mocks['mock_extractor']
    mock_embeddings.embed_text.return_value = 'mock_embedding'
    mock_extractor.extract.return_value = ['keyphrase1']
    store = InMemoryDocumentStore(embeddings=mock_embeddings, extractor=mock_extractor)
    node = Node(page_content='Node1 content', embedding=None)
    store.add_nodes([node])
    assert node.embedding == 'mock_embedding'
    assert node.keyphrases == ['keyphrase1']

# happy_path - set_run_config - Test set_run_config updates the run configuration for embeddings
def test_set_run_config(setup_mocks):
    mock_embeddings = setup_mocks['mock_embeddings']
    run_config = RunConfig()
    store = InMemoryDocumentStore(embeddings=mock_embeddings)
    store.set_run_config(run_config)
    mock_embeddings.set_run_config.assert_called_once_with(run_config)

# edge_case - filename - Test that filename property handles empty metadata gracefully
def test_filename_empty_metadata(setup_mocks):
    mock_logger = setup_mocks['mock_logger']
    document = Document(metadata={}, doc_id='5678')
    filename = document.filename
    assert filename == '5678'
    mock_logger.info.assert_called_once()

# edge_case - from_langchain_document - Test from_langchain_document with None input raises an error
def test_from_langchain_document_none(setup_mocks):
    with pytest.raises(TypeError, match='document cannot be None'):
        Document.from_langchain_document(None)

# edge_case - from_llamaindex_document - Test from_llamaindex_document with None input raises an error
def test_from_llamaindex_document_none(setup_mocks):
    with pytest.raises(TypeError, match='document cannot be None'):
        Document.from_llamaindex_document(None)

# edge_case - __eq__ - Test that __eq__ returns False for documents with different doc_id
def test_eq_different_doc_id(setup_mocks):
    doc1 = Document(doc_id='efgh')
    doc2 = Document(doc_id='ijkl')
    assert doc1 != doc2

# edge_case - add_documents - Test add_documents with no embeddings set raises an assertion error
def test_add_documents_no_embeddings(setup_mocks):
    mock_splitter = setup_mocks['mock_text_splitter']
    mock_splitter.transform_documents.return_value = [LCDocument(page_content='Node2 content', metadata={})]
    store = InMemoryDocumentStore(splitter=mock_splitter, embeddings=None)
    with pytest.raises(AssertionError, match='Embeddings must be set'):
        store.add_documents([LCDocument(page_content='Doc2 content', metadata={})])

# edge_case - add_nodes - Test add_nodes with no extractor set raises an assertion error
def test_add_nodes_no_extractor(setup_mocks):
    mock_embeddings = setup_mocks['mock_embeddings']
    store = InMemoryDocumentStore(embeddings=mock_embeddings, extractor=None)
    node = Node(page_content='Node3 content', embedding=[0.1, 0.2])
    with pytest.raises(AssertionError, match='Extractor must be set'):
        store.add_nodes([node])

