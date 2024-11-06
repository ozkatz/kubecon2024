#!/usr/bin/env python
import os
import json
from pathlib import Path

import click  # CLI helper
from kfp import dsl  # Kubeflow

# LangChain
from langchain.schema import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import LakeFSLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()
SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use only the following pieces of retrieved context to answer "
    "the question. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


def download_documents(repo: str, ref: str, prefix: str, output_path: Path):
    """
    Download documents from remote storage to local FS
    """
    output_path.mkdir(parents=True, exist_ok=True)
    loader = LakeFSLoader(
        lakefs_access_key=os.environ.get('LAKEFS_ACCESS_KEY_ID'),
        lakefs_secret_key=os.environ.get('LAKEFS_SECRET_KEY'),
        lakefs_endpoint=os.environ.get('LAKEFS_ENDPOINT_URL'),
    )
    loader.set_repo(repo)
    loader.set_ref(ref)
    loader.set_path(prefix)
    docs = loader.load()
    # save
    for doc in docs:
        name = Path(doc.metadata.get('path')).name
        with Path(output_path / (name + '.json')).open('w', encoding='utf-8') as out:
            out.write(doc.json())


def transform_to_vectors(input_path: Path, index_path: Path) -> None:
    """
    Chunk and index the documents in input_path into a vector DB at index_path
    """
    docs = []
    for fname in os.listdir(input_path):
        with Path(input_path / fname).open('r', encoding='utf-8') as f:
            docs.append(Document(**json.load(f)))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(str(index_path))


def get_retriever(index_path: Path) -> VectorStoreRetriever:
    """
    Open the vector DB at index_path and build a lanchain retriever from it
    """
    vector_store = FAISS.load_local(str(index_path), embeddings, 
        allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 20, 'fetch_k': 300})
    return retriever


def get_chain(retriever: VectorStoreRetriever) -> Runnable:
    """
    Incorporate the retriever into a question-answering chain.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def chat(rag_chain: Runnable, user_prompt: str) -> str:
    """
    Given a rag_chain, pass a user_prompt and return model's answer answer
    """
    response = rag_chain.invoke({"input": user_prompt})
    return response["answer"]


@click.group()
def cli():
    """
    CLI for our RAG application
    """


@cli.command('extract')
@click.option('--input-repository', default='peter-pan-data')
@click.option('--input-ref', default='main')
@click.option('--input-path', default='data/books/')
@click.option('--output-path', default='books', type=click.Path(path_type=Path, resolve_path=True))
def cli_extract(input_repository: str, input_ref: str, input_path: str, output_path: Path):
    """
    Extract books from remote storage and store locally as langchain Documents
    """
    download_documents(
        repo=input_repository,
        ref=input_ref,
        prefix=input_path,
        output_path=output_path,
    )


@cli.command('load')
@click.option('--input-path', default='books', type=click.Path(path_type=Path, resolve_path=True))
@click.option('--index-path', default='index', type=click.Path(path_type=Path, resolve_path=True))
def cli_load(input_path: Path, index_path: Path):
    """
    Load a set of documents from a local directory into a vector DB
    """
    transform_to_vectors(
        input_path=input_path,
        index_path=index_path,
    )


@cli.command('ask')
@click.option('--index-path', default='index', 
    type=click.Path(exists=True, path_type=Path, resolve_path=True))
@click.option('--prompt', required=True, help='Prompt to pass to the chatbot')
def cli_ask(index_path: Path, prompt: str):
    """
    Pass a question to our RAG application and print out the answer
    """
    retriever = get_retriever(index_path=index_path)
    chain = get_chain(retriever)
    answer = chat(chain, prompt)
    click.echo(answer)


if __name__ == '__main__':
    cli()
