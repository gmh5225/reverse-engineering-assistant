from typing import Optional, Dict, List, Any
from llama_index.prompts.base import PromptTemplate
from llama_index.selectors.prompts import SingleSelectPrompt
from llama_index.output_parsers.selection import SelectionOutputParser, _escape_curly_braces, FORMAT_STR
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.react.types import BaseReasoningStep
from llama_index.agent.react.formatter import ReActChatFormatter

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload

from llama_index.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ResponseReasoningStep,
)

from rich.json import JSON
import json

import logging

logger = logging.getLogger('reverse_engineering_assistant.llama_index_overrides')

REVA_SELECTION_OUTPUT_PARSER = """Some choices are given below. It is provided in a numbered list
(1 to {num_choices}),
where each item in the list corresponds to a summary.
---------------------
{context_list}
---------------------
Using only the choices above and not prior knowledge, return 
the choice that is most relevant to the question: '{query_str}'

{schema}

Select *only* one option. Return *only* JSON.
"""

class RevaSelectionOutputParser(SelectionOutputParser):
    def format(self, prompt_template: str) -> str:
        # We are running before the template is formatted, so we need to do a partial
        # then return a string that still contains some format fields
        # {query_str}, {num_choices}, {context_list}

        # Here we are working around the following bug: https://github.com/jerryjliu/llama_index/issues/7706
        # The multiple curly braces make life hard here, so we just find/replace :(
        template = prompt_template.replace('{schema}', _escape_curly_braces(FORMAT_STR))
        return template
    
class RevaReActChatFormatter(ReActChatFormatter):
    system_header = """
You are designed to help with a variety of reverse engineering tasks, from answering questions \
to providing summaries to other types of analyses.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please output a JSON object with the following keys:
- `thought`: a string representing your thought process
- `action`: a string containing only the tool name (one of {tool_names}) if using a tool.
- `action_input`: a valid JSON object representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
with a JSON object like so:

- `thought`: "I can answer without using any more tools."
- `answer`: [your answer here as a string]

- `thought`: "I cannot answer the question with the provided tools."
- `answer`: "Sorry, I cannot answer your query."

ONLY OUTPUT VALID JSON. If you do not, the system will not be able to parse your output.
"""

class RevaReActOutputParser(ReActOutputParser):
    logger: logging.Logger = logging.Logger('reverse_engineering_assistant.RevaReActOutputParser')

    def parse(self, output: str, is_streaming: bool = False) -> BaseReasoningStep:
        """ We need to fix the output from the LLM to match the expected outout from the parser.
        The parser is very strict.
        """
        import re

        # First examine the output for the action the LLM wants to take
        # and make sure *only* the function name is in the line. We will do this with
        # a regular expression.
        original_output = output
        
        # The output should be valid JSON
        try:
            if '```json' in output:
                self.logger.debug(f"Found ```json` in output, cleaning it up")
                cleaned_output = ''
                found = False
                for line in output.splitlines():
                    if line.strip() == '```':
                        break
                    if line.strip() == '```json':
                        found = True
                    elif found:
                        cleaned_output += line + '\n'
                output = cleaned_output

            if output.startswith('```json') and output.endswith('```'):
                output = "\n".join(output.splitlines()[1:-1])
            json_response = json.loads(output)
            self.logger.debug(f"JSON response: {json_response}")
            action = json_response.get('action')
            action_input = json_response.get('action_input')
            thought = json_response.get('thought')
            answer = json_response.get('answer')
            # Now format it as a string again
            if action and action_input and thought:
                return ActionReasoningStep(
                    thought=thought, action=action, action_input=action_input
                )
                
            if thought and answer:
                return ResponseReasoningStep(
                    thought=thought, response=answer, is_streaming=is_streaming
                )
            
            raise ValueError(f"Output from LLM is missing keys: {output}")

        except json.JSONDecodeError:
            self.logger.exception(f"Output from LLM is not valid JSON: {output}")
            raise ValueError(f"Output from LLM is not valid JSON: {output}")

class RevaLLMLog(BaseCallbackHandler):
    """Callback handler for printing llms inputs/outputs."""
    logger: logging.Logger
    def __init__(self) -> None:
        self.logger = logging.Logger('reverse_engineering_assistant.RevaLLMLog')
        self.logger.setLevel(logging.DEBUG)
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return
    
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return
    
    def _log_llm_event(self, payload: Dict) -> None:
        from llama_index.llms import ChatMessage
        #self.logger.debug(f"{payload}")
        pass

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id
    
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Count the LLM or Embedding tokens as needed."""
        self.logger.debug(f"Event type: {event_type}, payload: {payload}")
        if event_type == CBEventType.LLM and payload is not None:
            self._log_llm_event(payload)

