from llama_index.readers.file.docs import PDFReader
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pdf_path', type=str, help='pdf_path')
args = parser.parse_args()

Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')

loader = PDFReader()
document: Document = loader.load_data(args.pdf_path)

index = VectorStoreIndex.from_documents(document)

query_engine = index.as_query_engine()

response = query_engine.query("who is the author?")
print(response)