# ------------------------------
# AI Research Agent using LangGraph
# ------------------------------

import os, getpass
import operator
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import ArxivQueryRun
from langsmith import traceable


# ------------------------------
# Set Environment Variables for Tracing & API
# ------------------------------

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = ""  # Add your LangSmith API Key here
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

# Secure API Key for OpenAI
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# ------------------------------
# Define Shared Graph State
# ------------------------------

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


# ------------------------------
# Initialize LLM
# ------------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ------------------------------
# Graph Nodes
# ------------------------------

@traceable
def search_web(state):
    """Search recent papers from Arxiv API based on user question."""
    arxiv_search = ArxivQueryRun()
    search_docs = arxiv_search.run(state['question'])
    formatted_search_docs = f"<Document>\n{search_docs}\n</Document>"
    return {"context": [formatted_search_docs]}


@traceable
def search_wikipedia(state):
    """Search Wikipedia articles based on user question."""
    search_docs = WikipediaLoader(query=state['question'], load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"context": [formatted_search_docs]}


@traceable
def generate_answer(state):
    """Generate answer using LLM and gathered context."""
    context = state["context"]
    question = state["question"]
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, context=context)

    answer = llm.invoke([
        SystemMessage(content=answer_instructions),
        HumanMessage(content=question)
    ])

    return {"answer": answer}


# ------------------------------
# Build Graph
# ------------------------------

builder = StateGraph(State)
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Graph Flow
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()


# ------------------------------
# Run Graph
# ------------------------------

# Example Question
user_question = "What are the most recent advancements in AI chatbots and agents?"
result = graph.invoke({"question": user_question})
print("\n--- Final Answer ---\n")
print(result['answer'].content)
