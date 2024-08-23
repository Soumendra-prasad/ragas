import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np
from ragas.exceptions import MaxRetriesExceeded
from ragas.llms import BaseRagasLLM
from ragas.testset.docstore import DocumentStore, Node
from ragas.testset.filters import NodeFilter, QuestionFilter, EvolutionFilter
from ragas.llms.prompt import Prompt
from ragas.run_config import RunConfig
from ragas.llms.json_load import json_loader
from ragas.testset.evolution import Evolution, CurrentNodes, DataRow, SimpleEvolution, MultiContextEvolution, ReasoningEvolution, ConditionalEvolution

@pytest.fixture
def setup_mocks():
    with patch('ragas.llms.BaseRagasLLM', new_callable=MagicMock) as mock_llm, \
         patch('ragas.testset.docstore.DocumentStore', new_callable=MagicMock) as mock_docstore, \
         patch('ragas.testset.filters.NodeFilter', new_callable=MagicMock) as mock_node_filter, \
         patch('ragas.testset.filters.QuestionFilter', new_callable=MagicMock) as mock_question_filter, \
         patch('ragas.testset.filters.EvolutionFilter', new_callable=MagicMock) as mock_evolution_filter, \
         patch('ragas.llms.prompt.Prompt', new_callable=MagicMock) as mock_prompt, \
         patch('ragas.llms.json_load.json_loader', new_callable=MagicMock) as mock_json_loader:
        
        mock_llm_instance = mock_llm.return_value
        mock_docstore_instance = mock_docstore.return_value
        mock_node_filter_instance = mock_node_filter.return_value
        mock_question_filter_instance = mock_question_filter.return_value
        mock_evolution_filter_instance = mock_evolution_filter.return_value
        mock_prompt_instance = mock_prompt.return_value
        mock_json_loader_instance = mock_json_loader.return_value

        mock_llm_instance.generate = AsyncMock()
        mock_docstore_instance.get_random_nodes = MagicMock()
        mock_docstore_instance.get_similar = MagicMock()
        mock_node_filter_instance.filter = AsyncMock()
        mock_question_filter_instance.filter = AsyncMock()
        mock_evolution_filter_instance.filter = AsyncMock()
        mock_json_loader_instance.safe_load = AsyncMock()

        yield {
            'mock_llm': mock_llm_instance,
            'mock_docstore': mock_docstore_instance,
            'mock_node_filter': mock_node_filter_instance,
            'mock_question_filter': mock_question_filter_instance,
            'mock_evolution_filter': mock_evolution_filter_instance,
            'mock_prompt': mock_prompt_instance,
            'mock_json_loader': mock_json_loader_instance
        }

# happy_path - merge_nodes - Test that merging nodes combines their content and averages embeddings correctly.
@pytest.mark.asyncio
def test_merge_nodes_correctly_merges_content_and_embeddings(setup_mocks):
    mock_node = setup_mocks['mock_docstore'].get_random_nodes.return_value[0]
    mock_node.page_content = 'Node content'
    mock_node.keyphrases = ['node']
    mock_node.embedding = [0.4, 0.5, 0.6]

    nodes = CurrentNodes(
        root_node=Node(doc_id='doc1', page_content='Root content', keyphrases=['root'], embedding=[0.1, 0.2, 0.3]),
        nodes=[mock_node]
    )

    evolution = Evolution(generator_llm=setup_mocks['mock_llm'], docstore=setup_mocks['mock_docstore'])
    merged_node = evolution.merge_nodes(nodes)

    assert merged_node.doc_id == 'merged'
    assert merged_node.page_content == 'Node content'
    assert merged_node.keyphrases == ['root', 'node']
    assert np.allclose(merged_node.embedding, [0.25, 0.35, 0.45])

# happy_path - init - Test that init method sets is_async and run_config correctly.
def test_init_sets_is_async_and_run_config(setup_mocks):
    run_config = RunConfig(config_name='test_config')
    evolution = Evolution(generator_llm=setup_mocks['mock_llm'], docstore=setup_mocks['mock_docstore'])
    evolution.init(is_async=False, run_config=run_config)

    assert evolution.is_async is False
    assert evolution.run_config.config_name == 'test_config'

# happy_path - set_run_config - Test that set_run_config applies the configuration to all components.
def test_set_run_config_applies_to_all_components(setup_mocks):
    run_config = RunConfig(config_name='test_config')
    evolution = Evolution(
        generator_llm=setup_mocks['mock_llm'],
        docstore=setup_mocks['mock_docstore'],
        node_filter=setup_mocks['mock_node_filter'],
        question_filter=setup_mocks['mock_question_filter']
    )
    evolution.set_run_config(run_config)

    setup_mocks['mock_docstore'].set_run_config.assert_called_with(run_config)
    setup_mocks['mock_llm'].set_run_config.assert_called_with(run_config)
    setup_mocks['mock_node_filter'].set_run_config.assert_called_with(run_config)
    setup_mocks['mock_question_filter'].set_run_config.assert_called_with(run_config)

# happy_path - aretry_evolve - Test that aretry_evolve increments tries and retries evolution process.
@pytest.mark.asyncio
def test_aretry_evolve_increments_and_retries(setup_mocks):
    evolution = Evolution(generator_llm=setup_mocks['mock_llm'], docstore=setup_mocks['mock_docstore'])
    current_nodes = CurrentNodes(root_node=Node(doc_id='doc1', page_content='Root content'), nodes=[])

    setup_mocks['mock_llm'].generate.return_value.generations = [[{'text': 'Evolved question'}]]
    setup_mocks['mock_llm'].generate.return_value.generations[0][0].text = 'Evolved question'

    result = await evolution.aretry_evolve(current_tries=2, current_nodes=current_nodes)

    assert result[0] == 'Evolved question'
    assert result[1] == current_nodes
    assert result[2] == 'simple'

# happy_path - _transform_question - Test that _transform_question transforms the question using LLM.
@pytest.mark.asyncio
def test_transform_question_with_llm(setup_mocks):
    prompt = Prompt(format='Prompt text')
    setup_mocks['mock_llm'].generate.return_value.generations = [[{'text': 'Transformed question'}]]

    evolution = Evolution(generator_llm=setup_mocks['mock_llm'], docstore=setup_mocks['mock_docstore'])
    transformed_question = await evolution._transform_question(prompt, 'Original question')

    assert transformed_question == 'Transformed question'

# edge_case - merge_nodes - Test that merge_nodes handles empty nodes list gracefully.
def test_merge_nodes_with_empty_nodes_list(setup_mocks):
    nodes = CurrentNodes(
        root_node=Node(doc_id='doc1', page_content='Root content', keyphrases=[], embedding=None),
        nodes=[]
    )

    evolution = Evolution(generator_llm=setup_mocks['mock_llm'], docstore=setup_mocks['mock_docstore'])
    merged_node = evolution.merge_nodes(nodes)

    assert merged_node.doc_id == 'merged'
    assert merged_node.page_content == ''
    assert merged_node.keyphrases == []
    assert merged_node.embedding is None

# edge_case - init - Test that init method defaults run_config if none provided.
def test_init_defaults_run_config(setup_mocks):
    evolution = Evolution(generator_llm=setup_mocks['mock_llm'], docstore=setup_mocks['mock_docstore'])
    evolution.init(is_async=True, run_config=None)

    assert evolution.is_async is True
    assert evolution.run_config.default is True

# edge_case - set_run_config - Test that set_run_config does nothing if no components are set.
def test_set_run_config_no_components(setup_mocks):
    run_config = RunConfig(config_name='test_config')
    evolution = Evolution()
    evolution.set_run_config(run_config)

    # No components should be set, hence no calls to set_run_config
    assert not setup_mocks['mock_docstore'].set_run_config.called
    assert not setup_mocks['mock_llm'].set_run_config.called
    assert not setup_mocks['mock_node_filter'].set_run_config.called
    assert not setup_mocks['mock_question_filter'].set_run_config.called

# edge_case - aretry_evolve - Test that aretry_evolve raises MaxRetriesExceeded after max tries.
@pytest.mark.asyncio
def test_aretry_evolve_raises_after_max_tries(setup_mocks):
    evolution = Evolution(generator_llm=setup_mocks['mock_llm'], docstore=setup_mocks['mock_docstore'])
    current_nodes = CurrentNodes(root_node=Node(doc_id='doc1', page_content='Root content'), nodes=[])

    with pytest.raises(MaxRetriesExceeded):
        await evolution.aretry_evolve(current_tries=5, current_nodes=current_nodes)

# edge_case - _transform_question - Test that _transform_question raises assertion if LLM is None.
@pytest.mark.asyncio
def test_transform_question_raises_if_llm_none(setup_mocks):
    prompt = Prompt(format='Prompt text')
    evolution = Evolution(generator_llm=None, docstore=setup_mocks['mock_docstore'])

    with pytest.raises(AssertionError, match='generator_llm cannot be None'):
        await evolution._transform_question(prompt, 'Original question')

