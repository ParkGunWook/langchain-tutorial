from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.llm import model
from src.qna_rag_chain import conversational_rag_chain
from src.store import get_message_history

parser = StrOutputParser()
app = FastAPI()


@app.get("/model")
async def hello_model():
    return {"model": model.to_json()}


@app.get("/translate/{sentence}")
async def say_hello(sentence: str):
    messages = [
        SystemMessage(content="Translate the following from English into Korean"),
        HumanMessage(content=sentence),
    ]
    result = model.invoke(messages)
    return {"message": {parser.invoke(result)}}


@app.get("/{language}/translate")
async def say_hello(language: str, sentence: str):
    system_template = "Translate the following into {language}:"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{sentence}")]
    )
    chain = prompt_template | model

    result = chain.invoke({"language": f"{language}", "sentence": f"{sentence}"})

    return {"message": {parser.invoke(result)}}


class SentenceRequest(BaseModel):
    sentence: str


@app.post("/chat/{userId}")
async def say_hello(userId: str, sentenceRequest: SentenceRequest):
    result = conversational_rag_chain.invoke(
        {"input": sentenceRequest.sentence},
        config={
            "configurable": {"session_id": userId}
        }
    )
    return {"message": result["answer"]}


@app.get("/chat/{userId}")
async def get_chat(userId: str):
    return get_message_history(userId)
