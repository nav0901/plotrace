#!/usr/bin/env python3
# PlotTrace — Chapter‑Aware RAG for Books
# Generated from novel_plottrace.ipynb on 2025-10-20 07:03:44
#
# This script mirrors the Jupyter notebook code in a linear, runnable format.
# It includes light comments and cell markers to aid readability.
# If functions/classes are already documented, those docstrings are preserved.

"""
An AI-powered reading companion that creates chapter-specific summaries and answers questions about your books using a Retrieval-Augmented Generation (RAG) workflow.

Usage:
  python plottrace_commented.py --help

Notes:
  - This file was generated from the notebook; consider refactoring into
    modules (e.g., data_ingest.py, chunking.py, rag_pipeline.py) for production use.
"""

# =========================
# Imports
# =========================
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import cassio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# =========================
# Helper Functions / Main Logic
# =========================

# %% Cell 1: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
!pip install -q cassandra-driver

# %% Cell 2: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
!pip install -q langchain
!pip install -q openai
!pip install -q pypdf
!pip install -q faiss-cpu
!pip install -q tiktoken
!pip install -q cassio


# %% Cell 3: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
!pip install -q cassandra-driver
!pip install -q cassio>=0.1.1
!pip install -q tiktoken==0.4.0

# %% Cell 4: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
import cassandra

# %% Cell 5: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
from cassandra.cluster import Cluster

# %% Cell 6: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
from cassandra.auth import PlainTextAuthProvider

# %% Cell 7: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
import json

# %% Cell 8: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
cloud_config = {'secure_connect_bundle': "/content/secure-connect-pdf-qna-rag (1) (1).zip"}

# %% Cell 9: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json


with open("/content/xcroxx3@gmail.com-token.json") as f:
    secrets = json.load(f)

CLIENT_ID = secrets['clientId']
CLIENT_SECRET = secrets['secret']

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

# %% Cell 10: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
row = session.execute("select release_version from system.local").one()

if row:
  print(row[0])
else:
  print("Error")

# %% Cell 11: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
# 1) Install the split packages (and CassIO/Astra)
!pip install -U langchain langchain-core langchain-community langchain-openai langchain-text-splitters cassio astrapy pypdf

# 2) Restart runtime so the new packages are picked up
# import os, sys, time
# os.kill(os.getpid(), 9)


# %% Cell 12: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
# ✅ OpenAI integrations live here now (NOT in langchain_community.llms)
from langchain_openai import OpenAI, OpenAIEmbeddings

# ✅ Cassandra vector store lives under community
from langchain_community.vectorstores import Cassandra

# ✅ Correct splitter names & package
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# ✅ Document/loaders locations
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader


# %% Cell 13: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
import os
os.environ['OPENAI_API_KEY'] = "API-KEY"

# %% Cell 14: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
llm = OpenAI(temperature=0)
openai_embeddings = OpenAIEmbeddings()

# %% Cell 15: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_text_splitters import RecursiveCharacterTextSplitter


# %% Cell 16: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
KEYSPACE = "default_keyspace"      # Must exist in your Astra DB
TABLE_NAME = "pdf_novel_table"

# %% Cell 17: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
import cassio

# Tell CassIO to use this CQL session
cassio.init(session=session)

# %% Cell 18: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
vectorstore = Cassandra.from_documents(
    documents=docs,
    embedding=embeddings,
    session=session,          # keep passing it
    keyspace=KEYSPACE,
    table_name="pdf_novel_table"
)

# %% Cell 19: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
# 1. Split the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
docs = splitter.split_documents(pages)

# %% Cell 20: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
loader = PyPDFLoader("/content/harry-potter-and-the-philosophers-stone-by-jk-rowling.pdf")
pages = loader.load_and_split()

# %% Cell 21: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
embeddings = OpenAIEmbeddings()

# %% Cell 22: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
# assumes `vectorstore` is already created
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# %% Cell 23: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
query = "Summarize the main topic of the PDF."
docs = retriever.invoke(query)          # returns List[Document]

for i, d in enumerate(docs, 1):
    print(f"[{i}] {d.metadata.get('source','unknown')} | chunk_len={len(d.page_content)}")

# %% Cell 24: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# %% Cell 25: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
prompt = ChatPromptTemplate.from_template(
    """You are a novel analyzer assistant. Use only the context to answer.
If the answer isn't in the context, say you don't know.

Question: {question}

Context:
{context}"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %% Cell 26: (extracted from notebook)
# Tip: Consider moving this block into a function and adding docstrings if it defines reusable logic.
print(rag_chain.invoke("who does Harry lives with?"))