from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


def main():
    directory = '/Users/gabe/Documents/Github/text-crawler/output/nextjs'
    documents = load_docs(directory)
    print(f"Number of documents are : {len(documents)}")

    # small count
    documents = documents[:10]

    splited_docs = split_docs(documents)
    print(f"Number of splited documents are : {len(splited_docs)}")

    # Create Chroma instance with client 
    db = Chroma.from_documents(splited_docs, OpenAIEmbeddings())

    query = "What Nextjs core principle?"
    docs = db.similarity_search(query)
    print("============== docs ==============")
    print(docs)

if __name__ == '__main__':
   main()