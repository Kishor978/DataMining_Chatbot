from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

DATA_PATH="data/"
DB_FAISS_PATH="vectorstores/db_faiss"


def create_vector_db():
    loader=DirectoryLoader(DATA_PATH,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    texts=text_splitter.split_documents(documents)
    db=FAISS.from_documents(texts[:15],OllamaEmbeddings())
    db.save_local(DB_FAISS_PATH)
    
if __name__=='__main__':
    create_vector_db()