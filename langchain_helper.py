from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import WebBaseLoader
from dotenv import load_dotenv


load_dotenv()
embeddings = OpenAIEmbeddings()


def create_db_from_web_url(url: str) -> FAISS:
    loader = WebBaseLoader.from_url(url)
    page_content = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(page_content)

    db = FAISS.from_documents(docs, embeddings)
    return db



def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about Superjack: The Game 
        based on the game's official website content.
        
        Answer the following question: {question}
        By searching the following web page content: {docs}
        
        Only use the factual information from the content to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know."
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
