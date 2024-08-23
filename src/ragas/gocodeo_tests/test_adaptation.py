import pytest
from unittest.mock import patch, MagicMock
from ragas.llms import llm_factory
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics.base import MetricWithLLM
from langchain_core.language_models import BaseLanguageModel

@pytest.fixture
def setup_mocks():
    with patch('ragas.llms.llm_factory', return_value=MagicMock(spec=LangchainLLMWrapper)) as mock_llm_factory, \
         patch('ragas.llms.base.BaseRagasLLM', spec=True) as mock_base_ragas_llm, \
         patch('ragas.llms.base.LangchainLLMWrapper', spec=True) as mock_langchain_llm_wrapper, \
         patch('ragas.metrics.base.MetricWithLLM', spec=True) as mock_metric_with_llm, \
         patch('langchain_core.language_models.BaseLanguageModel', spec=True) as mock_base_language_model:
        
        yield {
            'mock_llm_factory': mock_llm_factory,
            'mock_base_ragas_llm': mock_base_ragas_llm,
            'mock_langchain_llm_wrapper': mock_langchain_llm_wrapper,
            'mock_metric_with_llm': mock_metric_with_llm,
            'mock_base_language_model': mock_base_language_model,
        }

# happy_path - test_adapt_with_provided_llm - Test that the 'adapt' function correctly adapts metrics with a provided LLM and language.
def test_adapt_with_provided_llm(setup_mocks):
    mocks = setup_mocks
    metrics = [mocks['mock_metric_with_llm'](llm=None)]
    language = 'en'
    llm = mocks['mock_base_ragas_llm']()

    adapt(metrics, language, llm)

    assert isinstance(metrics[0].llm, mocks['mock_langchain_llm_wrapper'])
    assert metrics[0].llm is not None

# happy_path - test_adapt_with_default_llm_factory - Test that the 'adapt' function uses the default LLM factory when no LLM is provided.
def test_adapt_with_default_llm_factory(setup_mocks):
    mocks = setup_mocks
    metrics = [mocks['mock_metric_with_llm'](llm=None)]
    language = 'en'

    adapt(metrics, language)

    assert isinstance(metrics[0].llm, mocks['mock_langchain_llm_wrapper'])
    assert metrics[0].llm is not None

# happy_path - test_adapt_with_language_and_cache_dir - Test that the 'adapt' function adapts metrics with the specified language and cache directory.
def test_adapt_with_language_and_cache_dir(setup_mocks):
    mocks = setup_mocks
    metrics = [mocks['mock_metric_with_llm'](llm=None)]
    language = 'fr'
    llm = mocks['mock_base_ragas_llm']()
    cache_dir = '/tmp/cache'

    adapt(metrics, language, llm, cache_dir)

    assert isinstance(metrics[0].llm, mocks['mock_langchain_llm_wrapper'])
    assert metrics[0].llm is not None

# happy_path - test_adapt_retain_original_llm - Test that the 'adapt' function retains the original LLM when adapting metrics without providing a new LLM.
def test_adapt_retain_original_llm(setup_mocks):
    mocks = setup_mocks
    original_llm = mocks['mock_base_ragas_llm']()
    metrics = [mocks['mock_metric_with_llm'](llm=original_llm)]
    language = 'es'

    adapt(metrics, language)

    assert metrics[0].llm == original_llm

# happy_path - test_adapt_invokes_metric_adapt - Test that the 'adapt' function invokes the 'adapt' method on metrics that support it.
def test_adapt_invokes_metric_adapt(setup_mocks):
    mocks = setup_mocks
    metric = mocks['mock_metric_with_llm'](llm=None)
    metric.adapt = MagicMock()
    metrics = [metric]
    language = 'de'
    llm = mocks['mock_base_ragas_llm']()

    adapt(metrics, language, llm)

    metric.adapt.assert_called_once_with(language, cache_dir=None)

# edge_case - test_adapt_invalid_llm_type_raises_value_error - Test that the 'adapt' function raises a ValueError when an invalid LLM type is provided.
def test_adapt_invalid_llm_type_raises_value_error(setup_mocks):
    mocks = setup_mocks
    metrics = [mocks['mock_metric_with_llm'](llm=None)]
    language = 'en'
    invalid_llm = object()

    with pytest.raises(ValueError, match="llm must be either None or a BaseLanguageModel"):
        adapt(metrics, language, invalid_llm)

# edge_case - test_adapt_with_empty_metrics_list - Test that the 'adapt' function handles an empty metrics list without errors.
def test_adapt_with_empty_metrics_list(setup_mocks):
    metrics = []
    language = 'en'

    adapt(metrics, language)

    assert metrics == []

# edge_case - test_adapt_with_none_language - Test that the 'adapt' function correctly handles a None language input.
def test_adapt_with_none_language(setup_mocks):
    mocks = setup_mocks
    metrics = [mocks['mock_metric_with_llm'](llm=None)]
    language = None
    llm = mocks['mock_base_ragas_llm']()

    adapt(metrics, language, llm)

    assert isinstance(metrics[0].llm, mocks['mock_langchain_llm_wrapper'])
    assert metrics[0].llm is not None

# edge_case - test_adapt_with_empty_cache_dir - Test that the 'adapt' function processes metrics when cache_dir is provided as an empty string.
def test_adapt_with_empty_cache_dir(setup_mocks):
    mocks = setup_mocks
    metrics = [mocks['mock_metric_with_llm'](llm=None)]
    language = 'en'
    llm = mocks['mock_base_ragas_llm']()
    cache_dir = ''

    adapt(metrics, language, llm, cache_dir)

    assert isinstance(metrics[0].llm, mocks['mock_langchain_llm_wrapper'])
    assert metrics[0].llm is not None

# edge_case - test_adapt_with_existing_llm - Test that the 'adapt' function handles metrics with pre-existing LLMs correctly when a new LLM is provided.
def test_adapt_with_existing_llm(setup_mocks):
    mocks = setup_mocks
    existing_llm = mocks['mock_base_ragas_llm']()
    metrics = [mocks['mock_metric_with_llm'](llm=existing_llm)]
    language = 'en'
    new_llm = mocks['mock_base_ragas_llm']()

    adapt(metrics, language, new_llm)

    assert isinstance(metrics[0].llm, mocks['mock_langchain_llm_wrapper'])
    assert metrics[0].llm is not None

