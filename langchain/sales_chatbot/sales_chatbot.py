import gradio as gr
import os

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ["OPENAI_BASE_URL"] = 'https://api.xiaoai.plus/v1'

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return chat_with_ai_for_unknow_question(message=message) or "这个问题我要问问领导"

def chat_with_ai_for_unknow_question(message):
    model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    prompt = ChatPromptTemplate.from_template("""你是一个专业的汽车销售，现在在与客人进行汽车销售交谈，但是客户问了你一个问题，
    你却一时不知道答案，这个时候你需要用你过硬的销售话术婉转的回答客人且不能让客人丧失兴趣。客人的问题是： {q} """)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    result = chain.invoke({"q": message})
    print(f"[ChatResult]{result}")
    return result
    
def reload_faiss(path):
    with open("real_estate_sales_data.txt", encoding='utf-8') as f:
        real_estate_sales = f.read()
    text_splitter = CharacterTextSplitter(        
    separator = r'\d+\.',
    chunk_size = 100,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = True,)
    docs = text_splitter.create_documents([real_estate_sales])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(path)


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="汽车销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # faiss_index_path = os.path.join(current_dir, 'real_estates_sale')
    # initialize_sales_bot(vector_store_dir=faiss_index_path)

    path = "test_data"
    reload_faiss(path);
    initialize_sales_bot(vector_store_dir=path)
    # 启动 Gradio 服务
    launch_gradio()
