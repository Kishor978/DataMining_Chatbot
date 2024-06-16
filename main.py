from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

import chainlit as cl
DB_FAISS_PATH="vectorstores/db_faiss"

custom_prompt_template="""Use the following pieces of information(i.e the context provided) to answer the given question.
If You dont knuw the answer, please just say thatyou don't know the answer, please don't try to makeup the answer.
<context>{context}</context>
Questions:{question}
Only return the helpgul answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """Promt for QA retrivial for each vector stores""" 
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])
    return prompt


def load_llm():
    llm=Ollama(model="llama2")
    return llm

def retrieval_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt}
)
    return qa_chain

def qa_bot():
    embeddings=OllamaEmbeddings()
    db=FAISS.load_local(DB_FAISS_PATH,embeddings,allow_dangerous_deserialization=True)
    llm=load_llm()
    qa_prompt=set_custom_prompt()
    qa=retrieval_chain(llm,qa_prompt,db)
    return qa



def final_result(query):
    qa_result=qa_bot()
    response=qa_result({'query':query})
    return response


@cl.on_chat_start
async def start():
    chain1=qa_bot()
    msg=cl.Message(content="starting the bot!!!!!!")
    await msg.send()
    msg.content="Hi, Welcome to the datamining assistant. What is your query?"
    await msg.update()
    cl.user_session.set("chain",chain1)
    
    
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()



