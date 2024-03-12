from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

import tree_sitter
import tree_sitter_languages

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--code_path", type=str, help="code_path")
args = parser.parse_args()

code_path = args.code_path

loader = GenericLoader.from_filesystem(code_path, parser=LanguageParser(language=Language.CPP))

documents = loader.load()

print(len(documents))

print(documents[0])