# langchain-conceptual guide

> source : https://python.langchain.com/v0.2/docs/concepts

# Architecture

## langchain-core

해당 패키지는 각 컴포넌트의 추상화와 그들을 아우르는 방법을 포함하고 있다. LLM, vector stores, retrievers 외 등등 코어 컴포넌트들의 인터페이스가 포함되어있다. 서드파티 integration 기능은 여기에 정의되어 있지 않다. 디펜던시들은 매우 가볍게 유지된다.

## Partner packages

80%의 integration은 `langchain-community`에 존재한다. 유명한 integration은 별개로 자체의 패키지로 분리해두었다.(`langchain-openai`, `langchain-anthropic` 등등) 중요한 integration에 대한 지원을 더 하기 위해서이다.

## langchain

`langchain` 패키지는 어플리케이션의 [cognitive architecture](https://en.wikipedia.org/wiki/Cognitive_architecture)를 만들기 위해서 chains, agents, and retrieval strategies을 포함한다. 서드 파티 integration은 여기에 속하지 않는다. 모든 chains, agents, and retrieval strategies은 특정 integration에 제한적으로 사용되지 않고 integration 전체에 보편적으로 사용된다.

## langchain-community

LangChain community에 의해서 관리되는 이 패키지는 서드 파티 integration을 포함한다. 주요 파트너의 패키지는 분리되어 있다. 이것은 모든 integration을 위한 다양한 컴포넌트(LLMs, vector stores, retrievers)들이 포함되어 있다. 이 패키지 안의 모든 디펜던시들은 패키지를 최대한 가볍게 가져가기 위해서 선택적으로 가져온다.

## langgraph

`langgraph`는 `langchain`의 그래프의 노드와 엣지를 모델링함으로서 robust and stateful multi-actor 어플리케이션 확장판이다.
LangGraph는 사용자가 정의한 플로우를 만드는 저수준의 API뿐 아니라, 다양한 공통 에이전트를 만드는 고수준의 인터페이스를 제공한다.

## langserve

LangChain의 chain을 REST APIs로 배포하는 패키지이다. API 서버를 쉽게 만들어준다.

# LangChain Expression Language (LCEL)

LangChain Expression Language 또는 LCEL은 LangChain 컴포넌트를 chain하는 declarative way이다. LCEL은 코드 변경 없이 시작부터 프로토타입을 프로덕션에 투입할 수 있도록 설계되었습니다. 가장 단순한 "prompt + LLM" chain부터 가장 복잡한 chain(사람들이 100개의 생산 단계를 거쳐 LCEL 체인을 성공적으로 운영하는 것을 보았다.)
)을 지원한다. 너가 LECL을 사용하고 싶을 몇가지 이유를 보여주겠다.

## First-class streaming support 

LECL과 함께 chain을 만들 때, 첫 토큰을 얻는 가능한 최고의 시간(첫 chunk의 출력이 나오는데 걸린 시간)을 얻을 수 있다. 토큰을 LLM에서 streaming 출력 파서로 직접 streaming하면 LLM 공급자가 원본 토큰을 출력하는 것과 동일한 속도로 역파싱된 증가하는 출력 chunk를 얻을 수 있습니다. 

## Async support

LECL로 만들어진 chain은 동기적인 API(Jupyter notebook으로 프로토타입을 만들 때)와 비동기적인 API(LangServer 서버로 호스팅할 때)로 로 호출될 수 있다. 프로토타입과 프로덕트를 같은 코드로 사용하면서 좋은 성능을 유지하고 같은 서버에서 많은 concurrent 요청을 다룰 수 있게 한다. 

## Optimized parallel execution 

LECL chain들은 언제든지 병렬적으로 실행 가능하다.(예를 들어, 여러 개의 retrievers로부터 문서를 불러올 때) LECL은 최소한의 지연을 위해서 동기/비동기 인터페이스에서 자동적으로 실행한다.

## Retries and fallbacks

LECL chain의 모든 부분에 retries와 fallbacks를 설정할 수 있다. 이러한 방법들은 chain들을 규모에 맞춰 신뢰성있게 한다. 최근에는 retries/fallbacks 기능에 대해서도 streming 지원을 준비하고 있고 해당 기능은 지연 시간을 없애면서도 신뢰성을 추가할 수 있게 할 것이다.

## Access intermediate results

복잡한 chain에서는 최종 결과물을 생성하기 이전에 중간 결과물에 접근하는 것은 유용하다. 이것은 엔드 유저로 하여금 무엇을 하는지 알리거나 chain을 디버깅하는데 사용할 수 있다. 중간 결과를 스트리밍할 수 있고 그것은 모든 LangServe 서버에서 가능하다.

## Input and output schemas

입력과 출력의 스키마는 LCEL chain이 Pydantic and JSONSchema으로 chain의 구조를 추론하게 해준다. 이것은 입력과 출력의 검증과 LangServe에서 중요한 기능에 사용된다.

## Seamless LangSmith tracing

복잡한 chain에 따라서 LangSmith로 모니터링을 해준다.

## Runnable interface

사용자 정의의 chain을 가능한 쉽게 만들기 위해서 "Runnable" 프로토콜을 구현했다. chat models, LLMs, output parsers, retrievers, prompt templates을 포함한 다양한 LangChain 컴포넌트들은 `Runnable`을 구현했다.
아래에는 runnable과 함께 작동하는 유용한 원시 타입이 있다.

표준 방식으로 그들을 실행함으로서 사용자 정의 chains를 쉽게 정의할 수 있다. 다음과 같은 표준 인터페이스가 있다.

- `stream` : 응답의 chunk를 stream해서 가져온다.
- `invoke` : 단일 입력값으로 chain으로 호출한다.
- `batch` : 복수 입력값으로 chain으로 호출한다.

동시성을 위해서 사용하는 비동기 메서드는 다음과 같다.

- `astream` : 응답을 비동기적으로 가져온다.
- `ainvoke` : 비동기적으로 요청한다(단수).
- `abatch` : 비동기적으로 요청한다.(복수).
- `astream_log` : 최종 응답 뿐 아니라, 중간 단계를 stream해서 가져온다.

컴포넌트에 따라서 인풋과 아웃풋의 타입은 다르다.

| Component      | Input Type                                              | Output Type             |
|----------------|---------------------------------------------------------|-------------------------|
| Prompt         | Dictionary                                              | PromptValue             |
| ChatModel      | Single string, list of chat messages or a PromptValue   | ChatMessage             |
| LLM            | Single string, list of chat messages or a PromptValue   | String                  |
| OutputParser   | The output of an LLM or ChatModel                       | Depends on the parser   |
| Retriever      | Single string                                           | List of Documents       |
| Tool           | Single string or dictionary, depending on the tool      | Depends on the tool     |

# Components

LangChain은 표준이면서도 확장성이 있는 인터페이스와 LLMs를 만들 때 유용한 다양한 컴포넌트의 외부 integrations을 제공한다. 몇몇 컴포넌트는 LangChain이 구현하고 몇몇은 서드 파티 integrations에 의존하기도 하고 섞여있기도 하다.

## Chat models

입력으로 일련의 메시지를 사용하고 출력으로 대화 메시지를 반환하는 Language 모델들이다. 이것들은 보통 최신형의 모델이다.(오래된 모델들은 LLMs를 사용한다.) Chat 모델들은 대화 메시지에 특정 역할을 할당해서, AI, 유저, 시스템 메시지와 같은 명령을 구별하게 해준다.

근본적인 모델은 메시지가 들어오고 나가지만, LangChain 래퍼는 모델이 문자열을 입력으로 가지게 해준다. 이렇게 LLMs 대신해서 chat models로 쉽게 사용하게 해준다.

문자열을 입력으로 전달하면, `HumanMessage`로 변환되고 모델로 전달된다.

LangChain은 Chat Models을 호스팅하지 않고 모두 서드 파티에서 지원한다.

ChatModels을 생성자는 다음과 같은 표준 파라미터를 가진다.

- model
- temperature(?)
- timeout
- max_tokens
- stop 
- max_retries
- api_key
- base_url

> 표준 파라미터는 의도한 기능에만 지원을 한다. 예를 들어 몇몇 LLMs 제공자는 최대 토큰 아웃풋에 대한 설정을 제공하지 않을 수 있다. 그러므로 max_tokens은 설정이 불가하다.
> 자체 패키지의 경우에는 표준 파라미터를 강제하지만 아니라면 강제하지는 않는다.

### Multimodality

몇몇 chat models은 이미지, 오디오와 비디오 등을 받는 multimodal이다. 아직은 흔치 않지만, model 제공자가 최고의 표준화된 API를 정의하지 못했다고 볼 수 있다. Multimodal 출력은 더욱 없다. 그러므로 multimodal 추상화는 매우 가볍고 발전할 예정이다.

LangChain에서 multimodal 입력을 지원하는 대부분의 chat models은 OpenAI의 content 블럭 포맷과 같이 받아들인다. 지금까지 이것은 이미지 입력으로 제한된다. Gemini와 같이 비디오와 바이트 입력을 지원하는 model에서, API는 native, model-specific representations을 지원한다.

## Messages

몇몇 language models은 입력으로 messages로 받고 message를 반환한다. 여러가지 타입의 messages가 있다. 모든 messages들은 `role`, `content`, `response_metadata`를 가진다.

- role : **누가** 메시지를 말하는지 설명한다. LangChain은 다양한 roles을 위한 다양한 message 클래스들이 존재한다.
- content : message의 내용을 설명한다. 다음과 같다.
  - string(대부분 models의 content이다.)
  - A List of dictionaries(multimodal 입력을 위해 사용된다. dictionary는 입력 타입과 입력의 위치 정보를 포함한다.)

### HumanMessage

유저로부터 만들어진 메시지이다.

### AIMessage

모델로부터 만들어진 메시지이다. 

- response_metadata : 응답에 대한 메타데이터를 가진다. 모델 제공자에 따라 특정한 값이 있다. log-probs and token usage와 같은 것이 사용될 수 있다.
- tool_calls : language model이 tool을 사용할지 결정하는데 사용한다. `AIMessage`의 결과에 포함된다. 아래와 같은 값이 있다.
  - name : 사용되어야할 tool의 이름
  - args : tool에 전달될 arguments
  - id : tool 호출의 id

### SystemMessage

model이 작동하는 방식을 보여준다. 모든 model 제공자가 지원하지는 않는다.

### FunctionMessage

`role`과 `content`뿐 아니라, 결과값을 만드는데 사용된 함수의 이름을 전달하는 `name`을 가진다. 

### ToolMessage

OpenAI's의 `function`과 `tool` message 타입을 구별한다. message는 `tool_call_id`를 가지고 있고 사용된 tool의 id를 전달하는데 사용된다.

## Prompt templates

Prompt templates은 유저의 입력과 파라미터를 language model을 위한 명령어로 변경한다. model의 응답에 대한 이해와 신뢰성있고 일관적인 언어 기반의 출력으로 생성한다.
Prompt templates은 dictionary로 입력을 받고, key는 prompt templates이 채워야하는 변수를 의미한다.
Prompt templates의 출력은 PromptValue이다. PromptValue은 LLM 또는 ChatModel로 전달될 수 있고 일련의 메시지로 캐스팅된다. PromptValue이 존재하는 이유는 strings과 messages사이에서 변환되기 쉽기 때문이다.
다음과 같은 Prompt templates이 있다.

### String PromptTemplates

단일 문자열을 format하고 단순한 입력에서 사용된다. 예시는 다음과 같다.
```python
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

prompt_template.invoke({"topic": "cats"})
```

### ChatPromptTemplates

문자열들을 format한다. 여러개의 템플릿들을 가지고 있다. 예시는 다음과 같다.
```python
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

prompt_template.invoke({"topic": "cats"})
```

위의 예시에서, ChatPromptTemplate은 호출과 동시에 두개의 메시지를 생성한다. 첫번째는 system message이고 format할 변수가 없다. 두번째는 HumanMessage이고 유저가 입력한 `topic` 변수로 format된다.

### MessagesPlaceholder

특정한 위치에 메시지들을 추가하는 prompt template이다. 위의 ChatPromptTemplate에서 두개의 메시지를 format하는 방법을 알아보았다. 유저가 특정 위치에 메시지들을 넘겨주게 하고 싶을 때는 어떻게 해야할까? 아래는 MessagesPlaceholder의 예시이다.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})
```

위의 코드는 두개의 메시지를 만들 것이다. 첫번째는 system message이고 우리가 전달할 HumanMessage이다. 만약 5개의 메시지를 전달한다면, 6개의 메시지를 생성한다. 이것은 특정 위치에 메시지를 배치하는데 유용하다.

MessagesPlaceholder 없이 사용하는 방식은 다음과 같다.
```python
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("placeholder", "{msgs}") # <-- This is the changed part
])
```

## Example selectors

더 나은 성능을 얻기 위한 일반적인 prompting technique은 예시를 prompt에 포함하는 것이다. language model에게 어떻게 행동해야하는지 예시를 주는 것이다. 이러한 예시들은 prompt에 하드코딩되어야 할 수 있지만, 동적으로 그들을 선택할 수 있을 것이다. Example Selectors은 prompts에 examples를 고르고 formatting하는 클래스이다.

## Output parsers

> 다양한 모델들이 점차 function (or tool) calling을 자동으로 지원한다. output parsing을 사용하기보다는 function/tool calling를 추천한다.

모델의 출력을 받고 다음 작업을 위한 포맷으로 변환하는 클래스이다. LLMs을 사용하면서 구조화된 데이터를 생성하거나 chat models과 LLMs으로부터의 출력을 정규화하는 데에 유용하다.
LangChain은 다양한 output parsers를 가지고 있다.

[다양한 output parser는 여기에서](https://python.langchain.com/v0.2/docs/concepts/#output-parsers)

## Chat history

대부분의 LLM 어플리케이션은 대화형 인터페이스를 가지고 있다. 대화에서의 중요한 점은 이전의 대화에서 얻은 정보이다. 최소한 대화형 시스템은 이전 메시지에 접근할 수 있어야 한다. 
ChatHistory의 개념은 임의의 chain을 wrap할 수 있는 LangChain내의 클래스이다. ChatHistory은 chain의 입력과 출력을 따라가고, message 데이터 베이스에 저장된다. 앞으로의 상호작용은 메시지를 가져오고 입력의 일부로서 chain에 전달될 것이다.

## Documents

LangChain에서의 Document는 몇몇 데이터에 대한 정보를 저장한다. 두개의 attributes가 있다.

- `page_content : str` : document의 내용이다. 현재는 문자열만 포함한다.
- `metadata: dict` : document와 관련된 임의의 메타데이터이다. document id, filename 등을 포함한다.

## Document loaders

Document 객체를 불러오는 클래스이다. LangChain은 다양한 데이터 소스(슬랙, 노션, 구글 드라이브)로부터 다양한 데이터를 가져올 수 있는 integrations이 있다. 
각각의 DocumentLoader은 그들의 파라미터를 가지고 있지만, `.load` 메서드를 통해서 실행가능하다.

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    ...  # <-- Integration specific parameters here
)
data = loader.load()
```
## Text splitters

documents를 불러올 떄, 너의 어플리케이션에 맞게 변환하고 싶을 것이다. 가장 쉬운 방법은 긴 document를 너의 모델의 컨텍스트에 맞춰서 작은 조각으로 나누는 것이다. LangChain은 내부에 쉽게 쪼개고, 합치고, 필터링하는 document transformers가 있다.
긴 텍스트를 다룰 때, 텍스트를 조각으로 나누는 것은 중요하다. 쉽게 들리지만, 여기에는 잠재적 복잡성이 존재한다. 이상적으로, 문맥적으로 연관 있는(semantically related) 텍스트를 같이 두고 싶을 것이다. "semantically related"는 텍스트의 종류에 따라 다르다. 몇가지 방법을 소개하겠다.

고수준에서의 text splitters는 다음과 같이 동작한다.

1. 텍스트를 semantically meaningful한 조각으로 나눈다.(대개 문장 단위로)
2. 작은 조각들을 특정 사이즈가 될 때까지 합친다.
3. 특정 사이즈가 되었을 때, 조각을 하나의 텍스트로 만들고 덮어쓸 몇개의 텍스트 청크를 만든다.

즉, text splitter를 커스터마이즈할 두가지 다른 방법이 있다.

1. 어떻게 텍스트를 나눌 것인가
2. 조각의 크기(다시 합칠)는 어떻게 되는가

## Embedding models

Embedding models은 텍스트를 벡터로 표현한다. 벡터는 일련의 숫자 또는 텍스트의 문맥적 의미를 저장한다. 텍스트를 이렇게 표현함으로서, 수학 연산을 적용할 수 있다. 자연어 검색 기능은 많은 유형의 context retrieval을 뒷받침하고 LLM에게 쿼리에 효과적으로 응답하는 데 필요한 관련 데이터를 제공합니다.

`Embeddings` 클래스는 text embedding models을 중재하기(interfacing) 위해서 설계되었다. 다양한 model 제공자(OpenAI, Cohere, Hugging Face, etc)와 로컬 model이 있다. 클래스는 모든 model에 표준 interface를 제공한다.

LangChain의 base Embeddings는 두개의 메서드를 제공한다. documents를 embedding하는 것과 쿼리를 embedding하는 것이다. 전자는 여러개의 텍스트를 가져오고, 후자는 단일 텍스트를 가져온다. 분리된 메서드를 가지는 이유는 몇몇 embedding 제공자가 documents(검색될)와 queries(검색 쿼리 그 자체)에 대해서 다른 embedding 메서드를 가지기 때문이다.

## Vector stores

비정형 데이터를 저장하고 검색하는 일반적인 방법은 embed하거나 resulting embedding vectors를 저장한 후에, 비정형 쿼리를 embed하고 embedded query와 '가장 유사한' embedding vectors를 가져오는 것이다.(?) vector store는 embedded data를 저장하고 벡터 검색을 수행한다.

대부분의 vector stores는 embedded vectors에 대한 메타데이터를 저장하고 유사도 검색 이전에, 메타데이터를 필터링하고 리턴된 documents에 대한 조작을 허용한다.

Vector stores는 다음과 같이 변환될 수 있다.

```python
vectorstore = MyVectorStore()
retriever = vectorstore.as_retriever()
```

## Retrievers

retriever는 비정형 쿼리에서 documents를 리턴하는 인터페이스이다. vector store보다 일반적이다. retriever는 documents를 저장할 필요는 없고, 그들을 return (or retrieve) 하기만 하면 된다. Retrievers는 vector stores로부터 생성되고, [위키피디아 검색](https://python.langchain.com/v0.2/docs/integrations/retrievers/wikipedia/)과 [Amazon Kendra](https://python.langchain.com/v0.2/docs/integrations/retrievers/amazon_kendra_retriever/)를 포함하기에 충분하다.

