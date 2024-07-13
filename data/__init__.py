
import os
from dotenv import load_dotenv
load_dotenv()


import html


from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/out1.txt")
loader.load()

class ChainSetup:
    def __init__(self, gemini_model:str = "gemini-1.5-flash" ,temperature: float = .2 ): 
        
        context = self.load_txt_files( )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        texts = text_splitter.split_text(context)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})

        system_prompt = (
            f"**I am Ankur Warikoo, here to help with your financial queries!**\n\n"
            "I can analyze the context and assist you with calculations if you provide monetary information.\n\n"
            "I will use information you provide, to help you plan and craft a strategy base on your needs."
            "I'll speak in a friendly and informative tone, primarily using Hinglish. However, I can switch to English or Hindi if needed.\n\n"
            "Based on the context, I'll adjust my tone, sentiment, and overall communication style.\n\n"
            "**Here's the retrieved context to answer your question:**\n\n"
            "<p id='context'>{context}</p>\n\n"
            "If I can't answer your question, I'll honestly tell you. My responses will be concise, with a maximum of 100 words.\n\n"
        )

        model = ChatGoogleGenerativeAI(
            model=gemini_model, 
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=temperature, 
            # convert_system_message_to_human=True
        )

        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
        )
        llm = model
        retriever = vector_index
        
        self.question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, self.question_answer_chain)

        self.qa_chain = RetrievalQA.from_chain_type(
            model, 
            retriever=vector_index,
            return_source_documents=True
        )

        print("""RAG is Ready""")

    def load_txt_files(self):
        context = ""
        # Get the current directory
        current_dir = os.getcwd()
        
        # Iterate through all files in the current directory
        for filename in os.listdir(current_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(current_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        context += file.read() + "\n\n"  # Add a separator between files
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        
        return context

    def get_rag_output(self, input: str): 
        response = self.rag_chain.invoke({"input": f"{input}"})
        escaped_res = html.unescape(response["answer"])
        return escaped_res
    

