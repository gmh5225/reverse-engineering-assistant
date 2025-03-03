#!/usr/bin/env python3

"""
This file contains the main assistant logic.
It provides a number of APIs for
- embedding data into the model
- performing inference on the model
"""

from __future__ import annotations
from functools import cached_property, cache
from abc import ABC, abstractmethod, abstractproperty
import logging
from pathlib import Path
import json
import tempfile
import datetime
import random

from rich.prompt import Prompt
from rich.logging import RichHandler


from typing import Any, Callable, List, Optional, Type, Dict

import llama_index
from llama_index import PromptTemplate, ServiceContext
from llama_index import StorageContext, VectorStoreIndex
from llama_index.indices.base import BaseIndex
from llama_index.indices.loading import load_index_from_storage
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import ChatMessage
from llama_index.response_synthesizers.tree_summarize import TreeSummarize
from llama_index.schema import Document
from llama_index.memory import ChatMemoryBuffer, BaseMemory

# Agent
from llama_index.agent import ReActAgent
from llama_index.tools import BaseTool
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools.function_tool import FunctionTool

from .tool import AssistantProject
from .model import ModelType, get_model
from .configuration import load_configuration, AssistantConfiguration
from .documents import AssistantDocument, CrossReferenceDocument, DecompiledFunctionDocument

from .llama_index_overrides import RevaSelectionOutputParser, REVA_SELECTION_OUTPUT_PARSER, RevaReActOutputParser, RevaReActChatFormatter


logger = logging.getLogger('reverse_engineering_assistant')

"""
List of RevaIndex classes to be registered with the assistant.
"""
_reva_index_list: List[Type[RevaIndex]] = []

def register_index(cls: Type[RevaIndex]) -> Type[RevaIndex]:
    _reva_index_list.append(cls)
    return cls

"""
List of RevaTool classes to be registered with the assistant.
"""
_reva_tool_list: List[Type[RevaTool]] = []

def register_tool(cls: Type[RevaTool]) -> Type[RevaTool]:
    _reva_tool_list.append(cls)
    return cls

class RevaTool(ABC):
    """
    A tool for performing exact queries on
    the data from the reverse engineering integration
    output.
    """
    project: AssistantProject
    service_context: ServiceContext

    tool_name: str
    description: str

    tool_functions: List[Callable]

    def __str__(self) -> str:
        return f"{self.tool_name}"

    def __init__(self, project: AssistantProject, service_context: ServiceContext) -> None:
        self.project = project
        self.service_context = service_context

    @cache
    def as_tools(self) -> List[FunctionTool]:
        """
        Returns a list of tools usable by the assistant
        based on the value of self.tool_functions.
        """
        tools: List[FunctionTool] = []
        for tool_function in self.tool_functions:
            tool = FunctionTool.from_defaults(
                fn=tool_function,
            )
            tools.append(tool)
        return tools



class RevaIndex(ABC):
    """
    An index of documents available to the
    reverse engineering assistant.
    """
    # The project we will operate on
    project: AssistantProject
    # Service context for the index
    service_context: ServiceContext

    index_name: str
    description: str

    index_directory: Path

    def __str__(self) -> str:
        return f"{self.index_name} @ {self.index_directory}"

    def __init__(self, project: AssistantProject, service_context: ServiceContext) -> None:
        self.project = project
        self.service_context = service_context
    
    @cached_property
    @abstractmethod
    def index(self) -> BaseIndex:
        return self.update_embeddings()

    @abstractmethod
    def get_documents(self) -> List[AssistantDocument]:
        raise NotImplementedError()

    @cache
    def as_query_engine(self) -> BaseQueryEngine:
        """
        Return a query engine for this index
        """
        configuration = load_configuration()

        # Unfortunately some models think reverse engineering is illegal and immoral
        # so we need a custom prompt to tell them we are allowed to reverse engineer
        # software... Without this the model sometimes decides to talk about tennis
        # due to a prompt deep within llama-index...
        prompt = PromptTemplate(configuration.prompt_template.index_query_prompt)
        query_engine = self.index.as_query_engine(
                text_qa_template=prompt,
                service_context=self.service_context,
                similarity_top_k=5,
                show_progress=False,
                verbose=logger.level == logging.DEBUG,
        )

        return query_engine

    @cache
    def as_tool(self) -> QueryEngineTool:
        """
        Return a query engine tool for this index
        """
        tool = QueryEngineTool.from_defaults(
                query_engine=self.as_query_engine(),
                description=self.description,
        )
        return tool

    def update_embeddings(self) -> BaseIndex:
        """
        Retrieve the index from disk, or generate it if it does not exist.
        """
        index = self.load_index()
        if not index:
            logger.info(f"No index on disk. Generating...")
            documents = self.get_documents()
            index = self.persist_index(documents)
        return index

    def load_index(self) -> Optional[BaseIndex]:
        if self.index_directory.exists():
            # Load the index from disk
            logger.info(f"Loading index from {self.project.get_index_directory()}")
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_directory),
            )
            # load_index_from_storage passes its kwargs to the index constructor
            # if we don't pass service_context, we get the default AI model (OpenAI)
            index = load_index_from_storage(storage_context, service_context=self.service_context)
            return index

    def persist_index(self, documents: List[AssistantDocument]) -> BaseIndex:
        """
        Given a list of documents, create an index and persist it to disk,
        return the index.

        Note that this is embedding model specific on disk, if you change the
        embedding model, these need to be updated to match. For now we set
        the embedding model to 'local' in models.py for all configurations.

        This is not fast, but it is consistent. In the future we can alter
        the path to contain the embedding model name, so these will be stored
        separately and managed appropriately.
        """

        embedding_documents: List[Document] = []
        for assistant_document in documents:
            # Transform from an AssistantDocument (our type) to a Document (llama-index type)

            document = Document(
                name=assistant_document.name,
                text=assistant_document.content,
                metadata=assistant_document.metadata,
            )
            embedding_documents.append(document)
        logger.info(f"Embedding {len(embedding_documents)} documents")
        index = VectorStoreIndex(
                embedding_documents,
                service_context=self.service_context,
                show_progress=False,
        )
        logger.info(f"Saving index to {self.project.get_index_directory()}")
        self.index_directory.mkdir(parents=True, exist_ok=False)
        index.storage_context.persist(str(self.index_directory))
        return index

#@register_index
@register_tool
class RevaDecompilationIndex(RevaIndex, RevaTool):
    """
    An index of decompiled functions available to the
    reverse engineering assistant.
    """
    index_name = "decompilation"
    description = "Used for retrieving decompiled functions"
    index_directory: Path
    def __init__(self, project: AssistantProject, service_context: ServiceContext) -> None:
        super().__init__(project, service_context)
        self.index_directory = self.project.get_index_directory() / "decompiled_functions"
        self.description = "Used for retrieveing decompiled functions"
        self.tool_functions = [
            self.get_decompilation_for_function,
            self.get_defined_function_list_paginated,
            self.get_defined_function_count,
        ]

    @cache
    def get_documents(self) -> List[DecompiledFunctionDocument]:
        """
        Filter documents in the project to just the DecompiledFunctionDocuments
        """
        assistant_documents = self.project.get_documents()
        decompiled_functions: List[DecompiledFunctionDocument] = []
        for document in assistant_documents:
            #logger.info(f"Checking {document}")
            if document.type == DecompiledFunctionDocument:
                decompiled_functions.append(document)
        return decompiled_functions
    
    @cache
    def get_decompilation_for_function(self, function_name_or_address: str) -> Dict[str, str]:
        """
        Return the decompilation for the given function. The function can be specified by name or address.
        """
        for document in self.get_documents():
            # In some cases the function name will be passed in
            if document.name == function_name_or_address:
                return document.to_json()
            # In some cases the function signature will be different to the name
            if document.function_signature == function_name_or_address:
                return document.to_json()
            # TODO: We want to surface an exact match first, but this is not working
            # because we do an `in` here.
            #if function_name_or_address in document.function_signature:
            #    return document.to_json()
            try:
                int(function_name_or_address, 16)
            except ValueError:
                continue
            if int(function_name_or_address, 16) >= int(document.function_start_address, 16) and int(function_name_or_address, 16) <= int(document.function_end_address, 16):
                return document.to_json()
                
    @cache
    def get_defined_function_list_paginated(self, page: int, page_size: int = 20) -> List[str]:
        """
        Return a paginated list of functions in the index. Use get_defined_function_count to get the total number of functions.
        """
        start = (page - 1) * page_size
        end = start + page_size
        if start > len(self.get_documents()):
            return []
        return [document.name for document in self.get_documents()[start:end] if document.is_external == False]
    
    @cache
    def get_defined_function_count(self) -> int:
        """
        Return the total number of defined functions in the index.
        """
        return len([document for document in self.get_documents() if document.is_external == False])

@register_tool
class RevaCrossReferenceTool(RevaTool):
    """
    An tool to retrieve cross references, to and from, addresses.
    """
    index_directory: Path
    def __init__(self, project: AssistantProject, service_context: ServiceContext) -> None:
        super().__init__(project, service_context)
        self.index_directory = self.project.get_index_directory() / "cross_references"
        self.description = "Used for retrieving cross references to and from addresses"

        self.tool_functions = [
            self.get_references_to_address,
            self.get_references_from_address,
        ]

    def get_documents(self) -> List[CrossReferenceDocument]:
        assistant_documents = self.project.get_documents()
        cross_references: List[CrossReferenceDocument] = []
        for document in assistant_documents:
            if document.type == CrossReferenceDocument:
                cross_references.append(document)
        return cross_references

    def get_references_to_address(self, address: str) -> Optional[List[str]]:
        """
        Return a list of references to the given address from other locations.
        These might be calls from other functions, or data references to this address.
        """
        logger.debug(f"Searching for {address}")
        for document in self.get_documents():
            if document.subject_address == address or document.symbol == address:
                logger.debug(f"Found document: {document}")
                return document.references_to

    def get_references_from_address(self, address: str) -> Optional[List[str]]:
        """
        Return a list of references from the given address to other locations.
        These might be calls to other functions, or data references from this address.
        """
        for document in self.get_documents():
            if document.subject_address == address or document.symbol == address:
                return document.references_from




class RevaSummaryIndex(RevaIndex):
    """
    An index of summaries available to the
    reverse engineering assistant.
    """
    index_directory: Path
    def __init__(self, project: AssistantProject, service_context: ServiceContext) -> None:
        super().__init__(project, service_context)
        self.index_directory = self.project.get_index_directory() / "summaries"
        self.description = "Used for retrieving summaries"

    def get_documents(self) -> List[AssistantDocument]:
        """
        Summarises the document and embeds the summary into the vector store.
        """
        summeriser = TreeSummarize(
            service_context=self.service_context,
        ) 

        summarised_documents: List[AssistantDocument] = []

        for document in self.project.get_documents():
            if document.type == DecompiledFunctionDocument:
                summary = summeriser.get_response(
                        query_str="Summarise the following function",
                        text_chunks=[document.content],
                )
                logger.debug(f"Summary {document}: {summary}")

                # TODO: Implement the SummaryDocument type?
                raise NotImplementedError()


class ReverseEngineeringAssistant(object):
    """
    A class representing the Reverse Engineering Assistant.

    This class provides functionality for querying a reverse engineering project, including loading indexes and tools,
    updating embeddings, and querying the query engine.

    Attributes:
        project (AssistantProject): The reverse engineering project to query.
        service_context (ServiceContext): The service context for the reverse engineering assistant.
        query_engine (Optional[BaseQueryEngine]): The query engine for the reverse engineering assistant.
        indexes (List[RevaIndex]): The indexes for the reverse engineering assistant.
        tools (List[RevaTool]): The tools for the reverse engineering assistant.
    """

    project: AssistantProject
    service_context: ServiceContext

    query_engine: Optional[ReActAgent] = None

    indexes: List[RevaIndex]
    tools: List[RevaTool]

    model_memory: BaseMemory


    @classmethod
    def get_projects(cls) -> List[str]:
        """
        Gets the names of the projects.

        Returns:
            List[str]: A list of project names.
        """
        return AssistantProject.get_projects()

    def __init__(self, project: str | AssistantProject, model_type: Optional[ModelType] = None) -> None:
        """
        Initializes a new instance of the ReverseEngineeringAssistant class.

        Args:
            project (str | AssistantProject): The reverse engineering project to query.
            model_type (Optional[ModelType], optional): The model type for the reverse engineering assistant. Defaults to None.
        """
        if isinstance(project, str):
            self.project = AssistantProject(project)
        else:
            self.project = project



        self.service_context = get_model(model_type)

        self.model_memory = ChatMemoryBuffer.from_defaults(
            llm=self.service_context.llm,
        )
        # We take the registered index types and construct concrete indexes from them
        self.indexes = [ index_type(self.project, self.service_context) for index_type in _reva_index_list]
        # and the same for tools
        self.tools = [ tool_type(self.project, self.service_context) for tool_type in _reva_tool_list]
        logger.debug(f"Loaded indexes: {self.indexes}")
        logger.debug(f"Loaded tools: {self.tools}")
        
    def update_embeddings(self):
        """
        Updates the embeddings for the reverse engineering assistant.
        """
        # Summarise all summaries together, to try to derive a high level description of the program
        summeriser = TreeSummarize(
            service_context=self.service_context,
            verbose=False,
        ) 

        # Here I pull our own prompt

        configuration: AssistantConfiguration = load_configuration()

        # TODO: Add more tools
        # - Strings in the binary. This should use a high k of n value for the index search
        #   and return many results.
        # - Cross references. This should return a graph like view of the callers and callees of a function
        #   Similar to the function call tree in Ghidra

        logger.debug("Building query engine")
        for index in self.indexes:
            logger.debug(f"Loading index: {index}")

        #chat_history: List[ChatMessage] = [
        #     ChatMessage(role="system", content=configuration.prompt_template.system_prompt),
        #]

        base_tools: List[BaseTool] = []
        for tool in self.tools:
            for function in tool.as_tools():
                base_tools.append(function)
        for index in self.indexes:
            base_tools.append(index.as_tool())

        self.query_engine = ReActAgent.from_tools(
            tools=base_tools,
            service_context=self.service_context,
            llm=self.service_context.llm,
            #chat_history=chat_history,
            verbose=False,
            max_iterations=30,
            # We need to override the output parser to fix a bug in llama-index
            react_chat_formatter=RevaReActChatFormatter(),
            output_parser=RevaReActOutputParser(),
            memory=self.model_memory,
            )

    def query(self, query: str) -> str:
        """
        Queries the reverse engineering assistant with the given query.

        Args:
            query (str): The query to execute.

        Returns:
            str: The result of the query.
        """
        if not self.query_engine:
            self.update_embeddings()
        if not self.query_engine:
            raise Exception("No query engine available")
        try:
            answer = self.query_engine.chat(query)
            return str(answer)
        except json.JSONDecodeError as e:
            logger.exception(f"Failed to parse JSON response from query engine: {e.doc}")
            return "Failed to parse JSON response from query engine"
        except ValueError as e:
            logger.exception(f"Failed to query engine: {e}")
            return "Failed to query engine... Try again?"
        except Exception as e:
            logger.exception(f"Failed to query engine: {e}")
            return "Failed to query engine... Try again?"

def get_thinking_emoji() -> str:
    """
    Returns a random thinking emoji.
    """
    return random.choice([
        "🤔",
        "🧐",
        "🤨",
        "👩‍💻",
        "😖",
        "✨",
        "🔮",
        "🔍",
        "🧙‍♀️",
    ])
    

def main():
    import argparse

    default_log_filename = f"ReVa-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.reva"
    default_log_path = Path(tempfile.gettempdir()) / Path(default_log_filename+".log")
    default_chat_path = Path(tempfile.gettempdir()) / Path(default_log_filename+".chat.txt")
    default_html_path = Path(tempfile.gettempdir()) / Path(default_log_filename+".html")

    parser = argparse.ArgumentParser(description="Reverse Engineering Assistant")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--debug', action='store_true', help="Debug output, useful during development")
    parser.add_argument("--project", required=False, type=str, help="Project name")
    parser.add_argument("-i", "--interactive", action="store_true", help="Enter interactive mode after processing queries")

    parser.add_argument('-f', '--file', default=default_log_path, type=Path, help=f"Save output to file. Defaults to {default_log_path}")

    parser.add_argument("-p", "--provider", required=False, choices=ModelType._member_names_, help="The model provider to use, defaults to the value of `model_type` in the config file.")

    parser.add_argument("QUERY", nargs="*", help="Queries to run, if not specified, enter interactive mode")

    args = parser.parse_args()

    model_type = ModelType._member_map_[args.provider] if args.provider else None

    from rich.console import Console
    console = Console(record=True)
    console.print(f"Welcome to ReVa! The Reverse Engineering Assistant", style="bold green")
    console.print(f"Logging to {args.file}")

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logger.level = logging.DEBUG

    rich_handler = RichHandler(
        console=console,
        level=logging_level,
    )
    logger.addHandler(rich_handler)

    # Create a logger for logging to a file. We'll log everything to the file.
    file_handler = logging.FileHandler(args.file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)

    # If the debug flag is enabled we turn these on.
    from . import llama_index_overrides
    llama_index.global_handler = llama_index_overrides.RevaLLMLog()

    if args.debug:
        logging.getLogger('httpx').addHandler(rich_handler)
        logging.getLogger('openai._base_client').addHandler(rich_handler)
        logging.getLogger('httpcore').addHandler(rich_handler)

    

    if not args.project:
        args.project = Prompt.ask("No project specified, please select from the following:", choices=ReverseEngineeringAssistant.get_projects())

    logger.info(f"Loading project {args.project}")
    assistant = ReverseEngineeringAssistant(args.project, model_type)
    assistant.update_embeddings()
    logger.info(f"Project loaded!")

    # Enter into a loop answering questions


    for query in args.QUERY:
        logger.debug(query)
        console.print(f"> {query}")
        with console.status(f"Thinking..."):
            result = assistant.query(query)
            console.print(result)

    if args.interactive or not args.QUERY:
        try:
            while True:
                query = Prompt.ask("> ")
                logger.debug(query)
                console.print(f"[green]{query}[/green]")
                with console.status(f"{get_thinking_emoji()} Thinking..."):
                    result = assistant.query(query)
                    console.print(result)
        except KeyboardInterrupt:
            console.print("Finished!")
        except EOFError:
            console.print("Finished")

    if args.file:
        logger.info(f"Output saved to {args.file}")
        logger.info(f"Chat saved to {default_chat_path}")
        console.save_text(default_chat_path, clear=False)
        logger.info(f"HTML saved to {default_html_path}")
        console.save_html(default_html_path, clear=False)
    
if __name__ == '__main__':
    main()
