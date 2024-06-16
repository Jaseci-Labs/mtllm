'''test_llm.py'''

from mtllm.llms.base import BaseLLM
from unittest import TestCase
from unittest.mock import patch


class Model(BaseLLM):
    def __init__(self, verbose: bool = False, max_tries: int = 10, **kwargs: dict) -> None:
        self.verbose = verbose
        self.max_tries = max_tries
        self.kwargs = kwargs
  

class TestBaseLLM(TestCase):

    @patch('mtllm.llms.base.BaseLLM.__infer__')
    def test_resolve_output(self, mock_infer):
        mock_infer.return_value = "[Output] 5"
        base_llm = Model()
        output = base_llm.resolve_output("Meaning Out\n[Output] 5", "5", "int", "integer")
        self.assertEqual(output, "5")
        
    @patch('mtllm.llms.base.BaseLLM.__infer__')
    def test_check_output(self, mock_infer):
        mock_infer.return_value = "Yes"
        base_llm = Model()
        is_in_desired_format = base_llm._check_output("5", "int", "integer")
        self.assertTrue(is_in_desired_format)

    # @patch('mtllm.llms.base.BaseLLM.__infer__')
    # def test_extract_output(self, mock_infer):
    #     mock_infer.return_value = "[output] 5"
    #     base_llm = Model(max_tries=1)
    #     output = base_llm._extract_output("Meaning Out\n[Output] 5", "5", "int", "integer", 1)
    #     self.assertEqual(output, "5")

    @patch('mtllm.llms.base.BaseLLM.__infer__')
    def test_extract_output_max_tries_zero(self, mock_infer):
        mock_infer.return_value = "5"
        base_llm = Model()
        with self.assertRaises(ValueError):
            base_llm._extract_output("Meaning Out\n[Output] 5", "5", "int", "integer", 0)

    # to test the case where the output is not in the desired format
    # @patch('mtllm.llms.base.BaseLLM.__infer__')
    # def test_extract_output_not_in_desired_format(self, mock_infer):
    #     mock_infer.side_effect = ["5", "No"]
    #     base_llm = Model()
    #     output = base_llm._extract_output("Meaning Out\n[Output] 5", "5", "int", "integer", 1)
    #     self.assertEqual(output, "5")