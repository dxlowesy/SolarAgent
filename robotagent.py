from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvapi_key = input("Enter your NVIDIA API key: ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key



import re
from typing import List, Union

import requests
from bs4 import BeautifulSoup

#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

#embedding_model = NVIDIAEmbeddings(model="ai-embed-qa-4")
embedding_model = NVIDIAEmbeddings(model = "nvidia/nv-embedqa-e5-v5")
def html_document_loader(url: Union[str, bytes]) -> str:
    """
    Loads the HTML content of a document from a given URL and return it's content.

    Args:
        url: The URL of the document.

    Returns:
        The content of the document.

    Raises:
        Exception: If there is an error while making the HTTP request.

    """
    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        # 创建Beautiful Soup对象用来解析html
        soup = BeautifulSoup(html_content, "html.parser")

        # 删除脚本和样式标签
        for script in soup(["script", "style"]):
            script.extract()

        # 从 HTML 文档中获取纯文本
        text = soup.get_text()

        # 去除空格换行符
        text = re.sub("\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Exception {e} while loading document")
        return ""

def index_docs(url: Union[str, bytes], splitter, documents: List[str], dest_embed_dir) -> None:
    """
    Split the document into chunks and create embeddings for the document

    Args:
        url: Source url for the document.
        splitter: Splitter used to split the document
        documents: list of documents whose embeddings needs to be created
        dest_embed_dir: destination directory for embeddings

    Returns:
        None
    """
    # 通过NVIDIAEmbeddings工具类调用NIM中的"ai-embed-qa-4"向量化模型
    for document in documents:
        texts = splitter.split_text(document.page_content)

        # 根据url清洗好的文档内容构建元数据
        print(type(document))
        metadatas = [document.metadata]

        # 创建embeddings嵌入并通过FAISS进行向量存储
        if os.path.exists(dest_embed_dir):
            update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
            update.add_texts(texts, metadatas=metadatas)
            update.save_local(folder_path=dest_embed_dir)
        else:
            docsearch = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
            docsearch.save_local(folder_path=dest_embed_dir)

def create_embeddings(embedding_path: str = "./embed"):

    embedding_path = "./embed"
    print(f"Storing embeddings to {embedding_path}")

    # 包含 NVIDIA NeMo toolkit技术文档的网页列表
    
    #Chinese baike
    urls = [
         "https://baike.baidu.com/item/%E5%A4%AA%E9%98%B3%E7%B3%BB/173281",
        "https://baike.baidu.com/item/%E9%87%91%E6%98%9F/19410?fromModule=lemma_search-box",
        "https://baike.baidu.com/item/%E6%B0%B4%E6%98%9F/135917?fromModule=lemma_search-box",
        "https://baike.baidu.com/item/%E7%81%AB%E6%98%9F/5627?fromModule=lemma_search-box",
        "https://baike.baidu.com/item/%E6%9C%A8%E6%98%9F/222105?fromModule=lemma_search-box",
        "https://baike.baidu.com/item/%E5%9C%9F%E6%98%9F/136354?fromModule=lemma_search-box",
        "https://baike.baidu.com/item/%E5%9C%B0%E7%90%83/6431?fromModule=lemma_search-box",
        "https://baike.baidu.com/item/%E5%A4%A9%E7%8E%8B%E6%98%9F/21805",
        "https://baike.baidu.com/item/%E6%B5%B7%E7%8E%8B%E6%98%9F/30351?fromModule=lemma_search-box"
    ]
    
    #English baike


    # 使用html_document_loader对NeMo toolkit技术文档数据进行加载
    documents = []
    for url in urls:
        document = html_document_loader(url)
        documents.append(document)

    #进行chunk分词分块处理
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=510,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.create_documents(documents)
    index_docs(url, text_splitter, texts, embedding_path)
    print("Generated embedding successfully")

create_embeddings()

embedding_path = "embed/"
docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct")
result = llm.invoke("Tell me something about Jupiter")
print(result.content)

llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

chat = ChatNVIDIA(model="ai-mixtral-8x7b-instruct", temperature=0.1, max_tokens=1000, top_p=1.0)

doc_chain = load_qa_chain(chat , chain_type="stuff", prompt=QA_PROMPT)

qa = ConversationalRetrievalChain(
    retriever=docsearch.as_retriever(),
    combine_docs_chain=doc_chain,
    memory=memory,
    question_generator=question_generator,
)


import speech_recognition as sr

r = sr.Recognizer()

print("Recognizer")
while(True):
    query = input("You asked:",)
    #print("you asked:")
    if(query == "v"):
        with sr.Microphone() as source:
        # read the audio data from the default microphone
            audio_data = r.record(source, duration=5)
            print("Please input your question by voice...")
        # convert speech to text
            query = r.recognize_sphinx(audio_data)
    if(query[0] == '\n' or query == "quit"):
        break
    result = qa({"question": query})
    rag_result = result.get("answer")
    print("Expert of Solar System:", rag_result)