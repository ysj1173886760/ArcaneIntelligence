from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser()
parser.add_argument("--code_path", type=str, help="code_path")
args = parser.parse_args()

code_path = args.code_path

loader = GenericLoader.from_filesystem(code_path, parser=LanguageParser(language=Language.CPP))

documents = loader.load()

logging.info("file cnt: {}".format(len(documents)))

cpp_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size = 128, chunk_overlap=24)
code_segments = cpp_splitter.split_documents(documents)

logging.info("code segment cnt: {}".format(len(code_segments)))

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(code_segments, hf)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})