

# happy_path - test_set_run_config_with_valid_run_config - Test that the run configuration is set correctly in the LLM.
async def test_set_run_config_with_valid_run_config(node_filter, mock_llm):
    run_config = RunConfig(max_tokens=100, temperature=0.5)
    node_filter.set_run_config(run_config)
    mock_llm.set_run_config.assert_called_once_with(run_config)

# happy_path - test_adapt_with_valid_language - Test that the filter adapts to a different language successfully.
def test_adapt_with_valid_language(node_filter, mock_prompt):
    node_filter.adapt(language='fr', cache_dir='/path/to/cache')
    mock_prompt.adapt.assert_called_once_with('fr', node_filter.llm, '/path/to/cache')

# happy_path - test_save_with_valid_cache_dir - Test that the filter prompts are saved correctly to the specified directory.
def test_save_with_valid_cache_dir(node_filter, mock_prompt):
    node_filter.save(cache_dir='/path/to/save')
    mock_prompt.save.assert_called_once_with('/path/to/save')

# happy_path - test_nodefilter_filter_above_threshold - Test that the NodeFilter correctly filters a node with a score above the threshold.
async def test_nodefilter_filter_above_threshold(node_filter, mock_llm, mock_context_scoring_parser):
    node = Node(page_content='valid content')
    mock_llm.generate.return_value = AsyncMock(generations=[[{'text': 'parsed text'}]])
    mock_context_scoring_parser.aparse.return_value = AsyncMock(dict=lambda: {'score': 1.6})
    result = await node_filter.filter(node)
    assert result['score'] is True

# happy_path - test_questionfilter_filter_valid_question - Test that the QuestionFilter returns the correct verdict and feedback for a valid question.
async def test_questionfilter_filter_valid_question(question_filter, mock_llm, mock_question_filter_parser):
    question = 'Is this a valid question?'
    mock_llm.generate.return_value = AsyncMock(generations=[[{'text': 'parsed text'}]])
    mock_question_filter_parser.aparse.return_value = AsyncMock(dict=lambda: {'verdict': 1, 'feedback': 'Valid question'})
    verdict, feedback = await question_filter.filter(question)
    assert verdict is True
    assert feedback == 'Valid question'

# edge_case - test_set_run_config_with_none - Test that the run configuration handles None as input gracefully.
async def test_set_run_config_with_none(node_filter, mock_llm):
    node_filter.set_run_config(None)
    mock_llm.set_run_config.assert_called_once_with(None)

# edge_case - test_adapt_with_unsupported_language - Test that the adapt function handles an unsupported language gracefully.
def test_adapt_with_unsupported_language(node_filter, mock_prompt):
    with pytest.raises(Exception) as excinfo:
        node_filter.adapt(language='xx', cache_dir='/path/to/cache')
    assert 'Unsupported language' in str(excinfo.value)

# edge_case - test_save_with_non_existent_cache_dir - Test that the save function handles a non-existent cache directory gracefully.
def test_save_with_non_existent_cache_dir(node_filter, mock_prompt):
    with pytest.raises(Exception) as excinfo:
        node_filter.save(cache_dir='/invalid/path')
    assert 'Directory not found' in str(excinfo.value)

# edge_case - test_nodefilter_filter_empty_node - Test that the NodeFilter returns an empty result when the node content is empty.
async def test_nodefilter_filter_empty_node(node_filter, mock_llm, mock_context_scoring_parser):
    node = Node(page_content='')
    mock_llm.generate.return_value = AsyncMock(generations=[[{'text': 'parsed text'}]])
    mock_context_scoring_parser.aparse.return_value = AsyncMock(dict=lambda: {})
    result = await node_filter.filter(node)
    assert result['score'] is False

# edge_case - test_questionfilter_filter_incomprehensible_question - Test that the QuestionFilter returns false verdict for an incomprehensible question.
async def test_questionfilter_filter_incomprehensible_question(question_filter, mock_llm, mock_question_filter_parser):
    question = '???'
    mock_llm.generate.return_value = AsyncMock(generations=[[{'text': 'parsed text'}]])
    mock_question_filter_parser.aparse.return_value = AsyncMock(dict=lambda: {'verdict': 0, 'feedback': 'Incomprehensible question'})
    verdict, feedback = await question_filter.filter(question)
    assert verdict is False
    assert feedback == 'Incomprehensible question'

