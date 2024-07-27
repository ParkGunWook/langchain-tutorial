# Chains

대화형 RAG 어플리케이션에서, retriever로부터 발행된 queries는 대화의 맥락을 가져야한다. LangChain은 이를 간단하게 하기 위해서 create_history_aware_retriever 생성자를 제공한다. chain이 `input`과 `chat_history`를 입력의 키로 받게 해주고 retriever의 출력에도 같은 스키마를 적용한다. create_history_aware_retriever가 필요한 입력은 다음과 같다.
