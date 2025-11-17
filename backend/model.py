import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain.chains import create_retrieval_chain
#from langchain.chains.retrieval import create_retrieval_chain
from langchain_classic.chains import create_retrieval_chain
# from langchain.chains.question_answering import load_qa_chain
from langchain_classic.chains.question_answering import load_qa_chain
import gdown


# Download .env file from gdrive
file_id = "1UuwbRQzHBA0L4TkLLUl-D_EaMBc2PmNu"
url = f"https://drive.google.com/uc?id={file_id}"
output = ".env"  

gdown.download(url, output, quiet=False)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")

# Ensure API key is set
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in .env")


# Load Stored Vector Database
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever()

# Use GPT-4 for answer generation
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Create Retrieval-Augmented Generation (RAG) Chain
# Load the QA chain properly
qa_chain = load_qa_chain(
    llm = llm,
    chain_type = "stuff"
)

def get_response(query):
    docs = retriever.invoke(query)
    return qa_chain.run(input_documents=docs, question=query)