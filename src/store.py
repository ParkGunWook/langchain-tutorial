from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_message_history(session_id: str):
    message_list = []

    if session_id not in store:
        return message_list

    for message in store[session_id].messages:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"

        message_list.append(f"{prefix}: {message.content}\n")
    return message_list
