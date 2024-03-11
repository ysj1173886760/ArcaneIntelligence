from llama_index.core.node_parser import CodeSplitter
from llama_index.core import SimpleDirectoryReader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--code_path', type=str, help='code_path')
args = parser.parse_args()

documents = SimpleDirectoryReader(args.code_path).load_data()

print(documents[0].text)