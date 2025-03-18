import os
import logging
import sys

from dotenv import load_dotenv
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain import hub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import START, END, StateGraph


import openlit

openlit.init(application_name="AdaptiveAgentChat",disable_metrics=True)

# Logging
logger = logging.getLogger("agent-chatbot")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# Load API keys from the environment
# load_dotenv()

print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))


logger.info("---LOAD ENV---")

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Handling Route
class QuestionRouter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(RouteQuery)

    def run(self, question: str) -> RouteQuery:
        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""

        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        question_router = route_prompt | self.llm
        return question_router.invoke({"question": question})


# Documents Grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class DocumentsGrader:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(GradeDocuments)

    def run(self, question: str, document: str) -> GradeDocuments:
        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | self.llm
        # docs = self.retriever.invoke(question)
        # doc_txt = docs[1].page_content
        return retrieval_grader.invoke({"question": question, "document": document})

class Generator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, question: str, docs: List[Document]) -> str:
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({"context": docs, "question": question})
        return generation

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeHallucinationsHandler:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(GradeHallucinations)

    def run(self, question: str, documents: List[Document], generation: str) -> GradeHallucinations:
        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        hallucination_grader = hallucination_prompt | self.llm
        return hallucination_grader.invoke({"documents": documents, "generation": generation})

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GradeAnswerGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(GradeAnswer)

    def run(self, question: str, generation: str) -> GradeAnswer:
        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        answer_grader = answer_prompt | self.llm
        return answer_grader.invoke({"question": question, "generation": generation})

class QuestionReWriter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, question: str) -> str:
        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        question_rewriter = re_write_prompt | self.llm
        return question_rewriter.invoke({"question": question})

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

# Main Agent
class AdaptiveRagAgent:
    def __init__(self, llm: ChatOpenAI, retriever: VectorStoreRetriever):
        # Initializing generators
        self.question_router = QuestionRouter(llm=llm)
        self.documents_grader = DocumentsGrader(llm=llm)
        self.generator = Generator(llm=llm)
        self.grade_hallucinations_handler = GradeHallucinationsHandler(llm=llm)
        self.grade_answer_generator = GradeAnswerGenerator(llm=llm)
        self.question_rewriter = QuestionReWriter(llm=llm)
        self.retriever = retriever

        # Create Graph
        self.graph = self._create_graph()

        # Websearch tool
        self.web_search_tool = TavilySearchResults(k=3)

    def run(self, question: str) -> str:
        # Initisl state
        initial_state = GraphState(question=question)
        # Execution
        final_state = self.graph.invoke(initial_state)
        # Final generation
        return final_state["generation"]


    def _create_graph(self) -> GraphState:

        workflow = StateGraph(GraphState)

        workflow.add_node("web_search", self._web_search)  # web search
        workflow.add_node("retrieve", self._retrieve)  # retrieve
        workflow.add_node("grade_documents", self._grade_documents)  # grade documents
        workflow.add_node("generate", self._generate)  # generatae
        workflow.add_node("transform_query", self._transform_query)  # transform_query

        # Build graph
        workflow.add_conditional_edges(
            START,
            self._route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self._grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        return workflow.compile()

    @openlit.trace
    def _retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        logger.info("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    @openlit.trace
    def _generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        logger.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation: str = self.generator.run(question, documents)
        return {"documents": documents, "question": question, "generation": generation}

    @openlit.trace
    def _grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.documents_grader.run(question=question, document=d)
            grade = score.binary_score
            if grade == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    @openlit.trace
    def _transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        logger.info("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.run({"question": question})
        return {"documents": documents, "question": better_question}

    @openlit.trace
    def _web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        logger.info("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

### Edges ###

    @openlit.trace
    def _route_question(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        logger.info("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.run({"question": question})
        if source.datasource == "web_search":
            logger.info("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            logger.info("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    @openlit.trace
    def _decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        logger.info("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            logger.info(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            logger.info("---DECISION: GENERATE---")
            return "generate"

    @openlit.trace
    def _grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        logger.info("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.grade_hallucinations_handler.run(
            question=question, documents=documents, generation=generation
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            logger.info("---GRADE GENERATION vs QUESTION---")
            score = self.grade_answer_generator.run(question=question, generation=generation)
            grade = score.binary_score
            if grade == "yes":
                logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

@openlit.trace
def load_docs(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


def main():

    embd = OpenAIEmbeddings()

    # Docs to index
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load
    docs_list = load_docs(urls)

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )
    retriever = vectorstore.as_retriever()

    # Model initialize
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = AdaptiveRagAgent(llm=llm, retriever=retriever)

    # Slack integration
    # Initializes your app with your bot token
    app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

    #handle message events with any text
    @app.message("")
    def message(message, say):
        response = agent.run(message["text"])
        say(response)

    try:
        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        handler.logger = logger
        handler.start()
    except Exception as e:
        logger.error(f"Error starting Slack bot: {e}")

if __name__ == "__main__":
    main()
