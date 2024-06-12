import os
import getpass
import urllib.request
from dotenv import load_dotenv
from IPython.display import display
from IPython.display import Markdown
import textwrap
import urllib
import warnings
import time
from pprint import pprint
from pathlib import Path as p
import numpy as np

# model related loads
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings

class Model:
    def __init__(self) -> None:
        load_dotenv('api.env')
        self.client = chromadb.PersistentClient()
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def load_model(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(model_name="gemini-pro")
        return model

    def to_markdown(self,text: str):
        text = text.replace('•', '  *')
        return text

    def download_document(self,document_url:str):
        start = time.time()
        data_folder = p.cwd() / "data"
        p(data_folder).mkdir(parents=True , exist_ok=True)
        pdf_file = str(p(data_folder , document_url.split('/')[-1]))
        urllib.request.urlretrieve(document_url,pdf_file)
        end = time.time()
        return pdf_file , end - start

    def make_db_collection(self,docuement_id , context):
        start = time.time()

        collection = self.client.get_or_create_collection(docuement_id,embedding_function=self.embeddings.embed_documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 10)
        
        texts = text_splitter.split_text(context)
        ids = list(np.arange(0,len(texts),1).astype(str))
        print(texts)
        print("adding to collection")
        collection.add(
            ids=ids,
            documents=texts
        )

        end = time.time()
        return collection , end - start


    def store_file(self,document_id , document_data):
        # download the file and store it in data folder
        print("start")
        collection = self.client.get_or_create_collection(
            document_id,
            embedding_function= self.embeddings.embed_documents
        )
        metadata = collection.get()
        if len(metadata['documents']) > 0:
            print("collection exists")
            return -1,-1
        
        print("start download")
        #pdf_file,down_time = self.download_document(document_url)
        pages = textwrap.wrap(document_data,width=1000)
        context = "\n\n".join(p for p in pages)

        print("start making collection")
        collection , db_time = self.make_db_collection(document_id,context)
        return 0,db_time

    def make_chain(self):
        prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                        not contained in the context, say "answer not available in context" \n\n
                        Context: \n {context}?\n
                        Question: \n {question} \n
                        Answer:
                    """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context" , "question"]
        )

        stuff_chain = load_qa_chain(
            self.model,
            chain_type="stuff",
            prompt=prompt
        )
        return stuff_chain

    def ask_model(self, document_id , query):
        stuff_chain = self.make_chain()
        print("let's begin")
        start = time.time()
        collection = self.client.get_collection(
            document_id,
            embedding_function=self.embeddings.embed_documents
        )

        db_index = Chroma(
            client=self.client,
            collection_name=document_id,
            embedding_function=self.embeddings
        ).as_retriever()

        docs = db_index.get_relevant_documents(query=query)
        end = time.time()
        db_time = end - start
        print(f"db took {end - start} sec")

        start = time.time()
        answer = stuff_chain(
            {
                "input_documents" : docs,
                "question" : query
            },
            return_only_outputs=True
        )
        end = time.time()
        print(f"model took {end - start} sec")
        return [answer , db_time , end - start]
