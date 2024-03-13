from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.vectorstores import faiss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--code_path", type=str, help="code_path")
args = parser.parse_args()

code_path = args.code_path

loader = GenericLoader.from_filesystem(code_path, parser=LanguageParser(language=Language.CPP))

documents = loader.load()

print(len(documents))

print(documents[0])

db = faiss.FAISS.from_documents(documents)
retriever = db.as_retriever()