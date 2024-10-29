import warnings
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug = True

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core._api.deprecation")

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db,
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("What is an interesting fact about the English language?")
print(result)
