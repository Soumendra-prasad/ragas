import pytest
from unittest.mock import patch, MagicMock
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.executor import Executor
from ragas.run_config import RunConfig
from langchain.text_splitter import TextSplitter
from ragas.exceptions import ExceptionInRunner
from ragas.testset.extractor import Extractor
from ragas_core.documents import Document as LCDocument

@pytest.fixture
def setup_mocks():
    with patch('uuid.uuid4', return_value='12345'), \
         patch('logging.getLogger', return_value=MagicMock()), \
         patch('langchain.text_splitter.TextSplitter', autospec=True), \
         patch('ragas.embeddings.base.BaseRagasEmbeddings', autospec=True), \
         patch('ragas.executor.Executor', autospec=True), \
         patch('ragas.run_config.RunConfig', autospec=True), \
         patch('ragas.testset.extractor.Extractor', autospec=True), \
         patch('ragas.exceptions.ExceptionInRunner', autospec=True), \
         patch('ragas_core.documents.Document', autospec=True):
        yield

# happy_path - filename - Test that the filename is returned from metadata if available.
def test_filename_from_metadata(setup_mocks):
    doc = Document(metadata={'filename': 'example.txt'})
    assert doc.filename == 'example.txt'

# happy_path - filename - Test that the filename defaults to doc_id if not available in metadata.
def test_filename_defaults_to_doc_id(setup_mocks):
    doc = Document(metadata={}, doc_id='12345')
    assert doc.filename == '12345'

# happy_path - from_langchain_document - Test that a new Document instance is created from an LCDocument.
def test_from_langchain_document_creates_new_instance(setup_mocks):
    lc_doc = LCDocument(page_content='Content', metadata={'author': 'John Doe'})
    doc = Document.from_langchain_document(lc_doc)
    assert isinstance(doc, Document)
    assert doc.metadata == {'author': 'John Doe'}

# happy_path - __eq__ - Test that two Document instances with the same doc_id are equal.
def test_documents_with_same_doc_id_are_equal(setup_mocks):
    doc1 = Document(doc_id='123')
    doc2 = Document(doc_id='123')
    assert doc1 == doc2

# happy_path - next - Test that the next node is returned from relationships if available.
def test_next_node_from_relationships(setup_mocks):
    node = Node(relationships={Direction.NEXT: 'NodeB'})
    assert node.next == 'NodeB'

# edge_case - filename - Test that filename returns doc_id when metadata filename is None.
def test_filename_when_metadata_filename_is_none(setup_mocks):
    doc = Document(metadata={'filename': None}, doc_id='12345')
    assert doc.filename == '12345'

# edge_case - from_langchain_document - Test that Document creation fails when LCDocument is missing page_content.
def test_document_creation_fails_without_page_content(setup_mocks):
    with pytest.raises(TypeError):
        LCDocument(metadata={'author': 'John Doe'})

# edge_case - __eq__ - Test that __eq__ returns False for two Document instances with different doc_ids.
def test_documents_with_different_doc_ids_are_not_equal(setup_mocks):
    doc1 = Document(doc_id='123')
    doc2 = Document(doc_id='456')
    assert doc1 != doc2

# edge_case - next - Test that next returns None when relationships do not contain NEXT direction.
def test_next_returns_none_when_no_next(setup_mocks):
    node = Node(relationships={})
    assert node.next is None

# edge_case - add_documents - Test that add_documents raises an assertion error when embeddings are not set.
def test_add_documents_raises_assertion_error_without_embeddings(setup_mocks):
    store = InMemoryDocumentStore(splitter=MagicMock())
    with pytest.raises(AssertionError):
        store.add_documents([Document(page_content='Content')])


