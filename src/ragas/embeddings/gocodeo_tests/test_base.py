import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ragas.run_config import RunConfig
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from llama_index.core.base.embeddings.base import BaseEmbedding

@pytest.fixture
def mock_run_config():
    return MagicMock(spec=RunConfig)

@pytest.fixture
def mock_openai_embeddings():
    return MagicMock(spec=OpenAIEmbeddings)

@pytest.fixture
def mock_base_embedding():
    return MagicMock(spec=BaseEmbedding)

@pytest.fixture
def mock_sentence_transformers():
    with patch('sentence_transformers.SentenceTransformer', autospec=True) as mock_st:
        yield mock_st

@pytest.fixture
def mock_cross_encoder():
    with patch('sentence_transformers.CrossEncoder', autospec=True) as mock_ce:
        yield mock_ce

@pytest.fixture
def mock_auto_config():
    with patch('transformers.AutoConfig.from_pretrained', autospec=True) as mock_ac:
        yield mock_ac

@pytest.fixture
def mock_model_for_sequence_classification_mapping_names():
    with patch('transformers.models.auto.modeling_auto.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES', autospec=True) as mock_mapping:
        yield mock_mapping

@pytest.fixture
def mock_numpy_intersect1d():
    with patch('numpy.intersect1d', autospec=True) as mock_np:
        yield mock_np

@pytest.fixture
def mock_sentence_transformer_encode():
    with patch('sentence_transformers.SentenceTransformer.encode', autospec=True) as mock_encode:
        yield mock_encode

@pytest.fixture
def mock_cross_encoder_predict():
    with patch('sentence_transformers.CrossEncoder.predict', autospec=True) as mock_predict:
        yield mock_predict

@pytest.fixture
def mock_loop():
    with patch('asyncio.get_event_loop', autospec=True) as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock()
        yield mock_loop

@pytest.fixture
def mock_openai_rate_limit_error():
    with patch('openai.RateLimitError', autospec=True) as mock_error:
        yield mock_error

# happy_path - test_embed_text_happy_path - Test embed_text with a valid string input
def test_embed_text_happy_path(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings, mock_run_config)
    mock_openai_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    result = wrapper.embed_query("test")
    assert result == [0.1, 0.2, 0.3]

# happy_path - test_embed_texts_happy_path - Test embed_texts with a valid list of strings
def test_embed_texts_happy_path(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings, mock_run_config)
    mock_openai_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    result = wrapper.embed_documents(["text1", "text2"])
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

# happy_path - test_set_run_config_happy_path - Test set_run_config with a valid RunConfig
def test_set_run_config_happy_path(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings)
    wrapper.set_run_config(mock_run_config)
    assert wrapper.run_config == mock_run_config

# happy_path - test_aembed_query_happy_path - Test aembed_query with a valid string input
async def test_aembed_query_happy_path(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings, mock_run_config)
    mock_openai_embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
    result = await wrapper.aembed_query("async test")
    assert result == [0.1, 0.2, 0.3]

# happy_path - test_aembed_documents_happy_path - Test aembed_documents with a valid list of strings
async def test_aembed_documents_happy_path(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings, mock_run_config)
    mock_openai_embeddings.aembed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    result = await wrapper.aembed_documents(["async text1", "async text2"])
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

# edge_case - test_embed_text_edge_case - Test embed_text with an empty string
def test_embed_text_edge_case(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings, mock_run_config)
    mock_openai_embeddings.embed_query.return_value = []
    result = wrapper.embed_query("")
    assert result == []

# edge_case - test_embed_texts_edge_case - Test embed_texts with an empty list
def test_embed_texts_edge_case(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings, mock_run_config)
    mock_openai_embeddings.embed_documents.return_value = []
    result = wrapper.embed_documents([])
    assert result == []

# edge_case - test_set_run_config_edge_case - Test set_run_config with None
def test_set_run_config_edge_case(mock_openai_embeddings):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings)
    wrapper.set_run_config(None)
    assert wrapper.run_config is not None

# edge_case - test_aembed_query_edge_case - Test aembed_query with an empty string
async def test_aembed_query_edge_case(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings, mock_run_config)
    mock_openai_embeddings.aembed_query.return_value = []
    result = await wrapper.aembed_query("")
    assert result == []

# edge_case - test_aembed_documents_edge_case - Test aembed_documents with an empty list
async def test_aembed_documents_edge_case(mock_openai_embeddings, mock_run_config):
    wrapper = LangchainEmbeddingsWrapper(mock_openai_embeddings, mock_run_config)
    mock_openai_embeddings.aembed_documents.return_value = []
    result = await wrapper.aembed_documents([])
    assert result == []

