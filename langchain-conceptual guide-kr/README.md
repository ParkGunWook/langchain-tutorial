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

## Tools

Tools은 agent, chain, 또는 chat model / LLM을 사용하게 해주는 인터페이스이다. 

tool은 아래의 컴포넌트를 포함한다.
1. tool의 이름
2. tool이 하는 일에 대한 설명
3. tool이 필요한 json 구조의 입력
4. 호출해야 하는 함수
5. tool이 유저에게 결과를 리턴해주어야하는지 여부

이름, 설명과 json 구조는 LLM의 컨텍스트에 포함되고 LLM이 툴을 적절하게 사용하게 한다.

사용 가능한 tools과 prompt를 제공 받으면, LLM은 적절한 arguments와 함께 하나 또는 이상의 tools을 실행하는 요청을 할 수 있다.

일반적으로, chat model 또는 LLM에 사용 될 tools을 디자인할 때, 다음 사항을 주의하는 것이 중요하다.

- tool 호출에 파인 튜닝된 Chat models은 파인 튜닝이 되지 않은 tool 호출보다 낫다.
- 파인 튜닝 되지 않은 models은 tools을 전혀 쓰지 못할 수 있고 tools이 복잡하거나 다양한 tool 호출이 필요하면 그렇다.
- Models은 tools이 컴포넌트를 적절하게 가질 때 더 좋은 성능을 가진다.
- 단순한 tools은 복잡한 tools보다 models에 적용하기가 쉽다.

## Toolkits

Toolkits들은 특정한 업무에 사용되기 좋은 tools 집합이다. 불러오는 것 또한 단순하다.

모든 Toolkits은 tools 리스트를 리턴하는 `get_tools` 메서드를 노출(expose)한다. 아래 코드와 같다.

```python
# Initialize a toolkit
toolkit = ExampleTookit(...)

# Get list of tools
tools = toolkit.get_tools()
```

## Agents

language models 자체만으로는 오직 출력 결과만 얻을 수 있고 별도의 행동을 취하지 않는다. LangChain의 주요한 use case는 agents 생성이다. Agents들은 LLM을 어떤 행동을 취할지와 이런 행동의 입력이 어떻게 되어야할지 결정하는 [reasoning engine](https://en.wikipedia.org/wiki/Semantic_reasoner)으로 사용하는 시스템이다. 행동의 결과는 agent에 피드백되고 추가 행동이 필요한지 결정하고 마쳐도 되는지 결정한다.

LangGraph는 제어 가능하고 사용자화 가능한 agents 생성에 특화된 확장판이다. LangGraph에서 agent 컨셉의 깊은 이해를 확인할 수 있다.

LangChain에는 deprecating하는 레거시 agent가 있다. AgentExecutor는 agents안의 런타임이다. 그것은 시작할 때는 큰 도움이 되지만, 사용자화된 agents를 시작하기에는 유연하지 못하다. 그것을 해결하기 위해서 LangGraph를 만들었다.

### ReAct agents

agents를 만들기에 유명한 아키텍처로서 ReAct가 있다. ReAct은 reasoning과 acting을 반복하는 프로세스로 구성된다. 그래서 "ReAct"를 풀어쓰면 "Reason"와 "Act"이다.

일반적인 flow는 다음과 같다.

- model은 입력과 이전 관측을 통해서 응답에 대해서 어떤 step을 가질지 생각한다.
- model은 가용한 tools로부터 action을 선택한다.(유저가 선택할 수도 있다.)
- model은 tool에 필요한 arguments를 생성한다.
- agent 런타임(executor)는 선택된 tool을 parse out하고 생성된 arguments와 함께 호출한다.
- executor는 tool 호출의 결과를 모델에게 가져온다.
- agent가 반응할 때까지 위의 flow를 반복한다.

model 특화 기능 없이 구현된 prompting이 있다. 그러나 가장 신뢰성 있는 구현은 tool calling 과 같은 기술을 사용한다.

## Callbacks

LangChain은 LLM 어플리케이션의 다양한 단계를 가로채는(hook) callbacks 시스템을 지원한다. logging, monitoring, streaming 외 다양한 작업에 유용하다. 

API 전체에서 `callbacks` argument를 통해서 아래의 다양한 이벤트를 subscribe할 수 있다. argument는 handler 객체의 리스트이고 하나 이상의 methods를 구현하도록 만들어졌다.

### Callback Events

https://python.langchain.com/v0.2/docs/concepts/#callback-events

### Callback handlers

Callback handlers는 `sync` 또는 `async`이다.

- Sync callback handlers는 BaseCallbackHandler 인터페이스를 구현한다.
- Async callback handlers는 AsyncCallbackHandler 인터페이스를 구현한다.

LangChain 런타임 중에 적절한 callback manager 설정이 가능하다.

### Passing callbacks

`callbacks` property는 API 대부분의 객체에 존재한다.(Models, Tools, Agents, 등등) 

- Request time callbacks : 입력 데이터가 요청에 추가될 때의 시간에 전달된다. 모든 `Runnable` 객체에서 사용가능하다. 그들이 정의된 모든 객체의 자식들에게 INHERITED된다. `chain.invoke({"number": 25}, {"callbacks": [handler]})` 다음과 같다.
- Constructor callbacks : `chain = TheNameOfSomeChain(callbacks=[handler])`. 객체의 생성자에 전달된다. 그들이 정의된 객체에만 한정되고 객체의 자식에게 inherited되지 않는다.

만약에, 커스텀 chain또는 runnable을 만들 때, request time callbacks이 자식 객체에게 전달(propagate)되는 것을 기억해야한다.

# Techniques

## Streaming

개별 LLM 호출은 긴 시간 동안 실행된다. 다양한 추론 과정이 필요할 때, 복잡한 chain 또는 agents를 만들면 더욱 심화된다.

다행히도, LLMs은 결과를 반복적으로 생성하고 최종 응답이 준비되기 전에, 중간 결과물을 보여줄 수 있다. 결과물이 가용해질 때마다 Consuming하는 것은 LLMs을 사용하는 앱의 레이턴시 문제를 경감하는데 도움을 준다.

LangChain안의 streaming에 대한 concepts과 considerations을 보여주겠다.

### .stream()과 .astream()

LangChain안의 대부분의 모듈은 `.stream()` 메서드를 포함한다.(비동기 환경에서는 `.astream()`을 포함한다.)
`.stream()`은 iterator를 리턴하고 for 루프로 소비가 가능하다.

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-sonnet-20240229")

for chunk in model.stream("what color is the sky?"):
    print(chunk.content, end="|", flush=True)
```

streaming을 지원하지 않는 models에서, iterator는 단일 chunk를 생성하지만, 여전히 그들을 호출하며 사용이 가능하다. `.stream()`를 사용하면 추가적인 설정없이 model이 streaming 모드로 설정된다.

결과 chunk는 컴포넌트의 타입에 따라 달라진다. 예를 들어서, chat models은 AIMessageChunks를 생성한다. 이 메서드는 LangChain Expression Language의 일부이기에, output parser를 통해서 생성된 chunk를 다양한 결과물로 변환 가능하다.

### .astream_events()

`.stream()`은 직관적이지만, chain의 마지막 값만 리턴한다. 단일 LLM 호출에서는 문제가 없지만, 다양한 LLM 호출을 통해서 복잡한 chain을 만들때, chain의 중간 값에 따라서 최종값을 사용하고 싶을 수 있다. 예를 들어서, 문서 앱 통해서 챗을 만들 때 최종 산출물과 소스를 리턴한다.

콜백을 사용하거나 chain을 만들 때 `.assgin()` 함수를 통해서 중간 값을 끝에 전달할 수 있지만, LangChain은 `.stream()`의 복잡한 콜백과 결합가능한 `.astream_events()` 메서드를 포함한다. 호출될 때, 프로젝트의 니즈에 맞게 필터링하고 전처리된 다양한 타입의 이벤트를 생성하는 이터레이터를 리턴한다.

streamed chat model 결과를 리턴하는 예시이다.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-sonnet-20240229")

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser

async for event in chain.astream_events({"topic": "parrot"}, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(event, end="|", flush=True)
```

### Callbacks

LangChain에서 LLM의 결과물로부터 스트리밍하는 lowest level 방법은 callback 시스템이다. LangChain 컴포넌트에 `on_llm_new_token` 이벤트를 처리하는 callback handler를 전달하면 된다. 컴포넌트가 실행되면, LLM과 chat moedls는 생성된 토큰과 함께 콜백을 호출한다. 콜백 내에서, HTTP 응답과 같은 목적지로 토큰을 연결(pipe)하면 된다. on_llm_end 이벤트로 정리도 할 수 있다.

Callbacks은 LangChain이 출시했을 때 streaming의 첫 기술이다. 강하고 일반적인 기술이지만, 개발자들이 다루기 어려운 편이다.

- 결과를 모으기 위해서 몇가지 aggregator 또는 다른 stream을 초기화하고 관리해야한다.
- execution 순서가 보장되지 않고 이론적으로 `.invoke()` 메서드가 종료된 후에, callback 실행이 된다.
- 제공자는 한번에 리턴할 때와 달리 추가적인 파라미터를 stream 출력에서 요구할 수 있다.
- callback 결과에 따라서 model 호출의 결과를 무시할 수도 있다.

### Tokens

대부분의 모델 제공자가 입력과 출력을 측정하는 최소 단위는 token으로 불린다. Tokens은 language models이 텍스트를 생성할 때 읽고 생성하는 최소단위이다. token의 정확한 정의는 model이 어떤 훈련을 거쳤는지에 달려있다. English의 경우에, token은 apple과 같은 단일 단어일 수 있다.

모델에 prompt를 보낼 때, prompt안의 words와 characters는 tokenizer를 통해서 tokens으로 인코딩된다. model은 생성된 출력 tokens을 tokenizer가 사람이 읽을 수 있는 텍스트로 streams back한다.

`LangChain is cool!`은 `/Lang/Chain/ /is/ /cool/!/`로 5개의 tokens으로 분리된다. 그리고 실제 단어의 경계와는 다르다.

language models이 글자와 같은 직관적이지 않은 tokens을 사용하는 이유는 텍스트를 처리하고 이해하는데에 있다. 고수준에서, language models은 초기 입력과 이전 결과물을 바탕으로 그들의 다음 결과물을 예측한다. tokens language models이 개별 글자보다는 의미를 가지는 언어의 기본 단위인 토큰을 이용해서 모델을 훈련하면, model이 언어의 구조(문법과 문맥)를 이해하고 배우게 하는게 더욱 쉽다. 더 나아가, model이 글자 단위 처리를 하는것 보다 tokens을 사용해서 처리하면 더욱 효율적이다.

## Structured output

LLMs은 임의의 텍스트를 생성할 수 있다. model이 다양한 범위의 입력에 적절한 대답을 하도록 하지만, 몇몇 경우에는 LLM의 결과의 포맷이나 구조를 제한하는 것이 좋다. 이것을 structured output라고 한다.

예를 들어서, output이 RDB에 저장되려면 model이 정의된 스키마나 포맷에 맞게 생성되면 된다. unstructured 텍스트로부터 특정 정보를 추출하는 것은 유용한 케이스 중에 하나이다. 대부분, 출력은 JSON이고, YAML또한 유용할 수 있다. LangChain의 몇가지 structured output에 대해서 알아보겠다.

### .with_structured_output()

편의를 위해, LangChain chat models은 `.with_structured_output()` 메서드를 지원한다. 이 메서드는 입력으로 스키마가 필요하고 dict 또는 Pydantic 객체를 리턴한다. 보통, 이 메서드는 아래에 설명할 메서드를 지원하는 모델에만 존재한다. model을 위한 적절한 output parser와 schema를 포매팅하는 것에 집중하겠다.

```python
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")
```

이 메서드를 structured output를 처음 사용할 때 추천한다.

- output parser 없이 내부적으로 모델 특화 기능을 사용한다.
- tool calling을 사용하는 models에서 특별한 prompting이 필요하지 않다.
- 몇가지 기술이 뒷받침해준다면, 어떤 것을 사용할지 토글할 수도 있다.

다른 기술이 필요하거나 원할 수 있다.

- 사용중인 chat model이 tool calling을 지원하지 않는다.
- 복잡한 스키마를 사용하고 model이 만족할만한 결과를 가져오지 못하고 있다.

### Raw prompting

structure structure을 얻는 가장 쉬운 방법은 잘 질문하는 것이다. 질의에, 어떤 출력을 원하는지 설명하는 것을 추가하고 output parser를 이용해서 raw model message또는 string output을 조작하기 쉬운 방향으로 변환하는 것이다.

raw prompting의 최대 강점은 유연함이다. 

- Raw prompting은 특별한 model 기능이 필요하지 않고, 전달된 스키마를 이해할 추론 능력만 있으면 된다.
- 어떠한 포맷으로도 prompt할 수 있다. XML또는 YAML 같은 특정한 타입의 데이터로 훈련된 모델이라면 더욱 유용하다.

그러나 몇가지 단점이 있다.

- LLMs은 [non-deterministic](https://en.wikipedia.org/wiki/Nondeterministic_algorithm)이고 LLM에게 부드러운 parsing을 위해서 정확하게 맞는 포맷으로 지속적으로 prompting하는 것은 놀라울 정도로 어렵고 제한적인 model이다.
- 개별 models은 그들이 훈련한 데이터에 맞게 강점이 있고, prompts 최적화는 어려운 작업이다. 몇몇은 json 구조를 잘 해석하고 몇몇은 TypeScript를 정의하고 여전히 몇몇은 XML을 선호한다.

model 제공자에 의해서 제공된 기능이 reliability를 높이지만, prompting 기술은 어떠한 메서드를 선택하든 결과를 튜닝하는데 중요한 기술이다.

### JSON mode

몇몇 models은 JSON mode라 불리는 기능을 지원한다. 보통 설정으로 활성화된다.

활성화되면, JSON 모드는 model의 출력을 항상 유효한 JSON으로 제한한다. 가끔 커스텀 prompting이 필요하지만, raw prompting을 하는 것보다는 부담이 훨씬 적고 "you must always return JSON"보다 낫다. 또한 parse하기도 쉽다.

tool calling보다 직접 사용하기 쉽고 일반적으로 사용되기에, 결과를 prompting과 shaping하는 것이 tool calling보다 유연하다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser

model = ChatOpenAI(
    model="gpt-4o",
    model_kwargs={ "response_format": { "type": "json_object" } },
)

prompt = ChatPromptTemplate.from_template(
    "Answer the user's question to the best of your ability."
    'You must always output a JSON object with an "answer" key and a "followup_question" key.'
    "{question}"
)

chain = prompt | model | SimpleJsonOutputParser()

chain.invoke({ "question": "What is the powerhouse of the cell?" })
```

```json
{"answer": "The powerhouse of the cell is the mitochondrion. It is responsible for producing energy in the form of ATP through cellular respiration.",
 "followup_question": "Would you like to know more about how mitochondria produce energy?"}
```

### Function/tool calling

> tool calling을 function calling와 혼용해서 부른다. function calling이 단일 함수 호출을 의미하지만, 모든 models을 그들이 각 메시지에서 다양한 tool과 function calls을 리턴할 수 있다고 가정한다.

Tool calling은 유저가 정의한 스키마에 일치하는 model이 주어진 prompt에 응답하게 한다. model이 어떤 행동을 한다고 이름에서 내포하지만, 실제로는 아니다. model은 tool에 arguments와 함께 오고 실제 tool 실행은 유저에게 달려있다. 예를 들어서, unstructured 텍스트로부터 스키마에 일치하는 출력을 추출하기를 원한다면, 모델에게 원하는 스키마에 매치되는 파라미터를 extraction tool을 줄 수 있다. 그렇게 최종 결과에 반영될 것이다.

models에서 tool calling을 지원한다면 매우 간단하다. 내장 모델의 스키마를 선호하는 기능을 제거한다. 자연적으로 대리 flows를 지원하고 숫자나 조합에서 씨름하지 않고 tool schemas를 전달만하면 된다.

많은 LLM 제공자들은 다양한 tool calling 기능을 지원한다. 이런 기능들은 LLM이 가용한 tools과 schemas를 요청에 포함하도록 허용하고 이러한 tools을 응답에 포함하도록 허용한다. 예를 들어서, 검색 엔진 tool이 주어지면, LLM은 검색 엔진에 먼저 요청한다. LLM을 호출하는 시스템은 tool call을 받고 실행하고 LLM에게 응답을 알려주기 위해서 그것의 출력을 리턴한다. LangChain은 내장 tools가 있고 커스텀 tool을 만들기 위한 몇가지 메서드가 있다.

LangChain은 다양한 models들에서 사용할 수 있는 tool calling 표준 인터페이스를 제공한다.

표준 인터페이스는 다음과 같다.

- ChatModel.bind_tools() : model이 사용가능한 tools을 알려주는 메서드이다. 
- AIMessage.tool_calls : AIMessage의 속성이다. model에 의해서 요청된 tool call을 접근하는 model에 의해 리턴된다.

## Retrieval

LLMs은 크지만 고정된 데이터에서 훈련하고 개인적이거나 최근 정보를 바탕으로 그들의 추론할 능력을 제한한다. 몇몇 사실을 기반으로 LLM을 파인 튜닝하는 것은 이것을 이완하지만, 종종 사실에 맞지 않거나 비용이 많이 들 수 있다. Retrieval은 주어진 입력에서 LLM이 응답을 향상시켜주는 관련 정보 제공의 절차이다. Retrieval augmented generation (RAG)는 retrieved된 정보를 사용해서 LLM 생성하는 밑작업이다.

RAG는 retrieved된 문서의 관련성과 품질만큼 좋다. 다행히, 떠오른 기술들이 RAG시스템을 향상시키고 디자인하는데 사용된다. 우리는 다양한 기술을 분류하고 요약하는데 집중했고 이번 절에서 고수준의 전략적 가이드를 주겠다. 다양한 조각들로 실험할 수 있을 것이다. 

### Query Translation

먼저, RAG 시스템에서 유저의 입력을 고려해보겠다. 이상적으로, RAG 시스템은 단어로만 이루어진 질문부터 복잡한 질의를 가진 다양한 범위의 입력을 관리한다. 입력을 검토하고 선택적으로 수정하기 위해서 LLM을 사용하는 것은 query translation의 핵심 아이디어이다. 일반적인 버퍼를 제공하고 retrieval 시스템을 위한 유저 인풋을 최적화한다. 예를 들어서, keywords를 추출하는 단순 작업이거나 복잡한 쿼리에서 다양한 서브 질문을 생성하는 것이다.

| Name          | When to use                            | Description                                                                                                          |
|---------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Multi-query   | 질문의 다양한 관점을 담당해야할 필요가 있을 때             | 유저의 질문을 다양한 관점으로 재생성하고 각각의 재생성된 질문을 위한 문서를 retrieve하고 모든 질문을 위한 유일한 문서를 리턴한다.                                        |
| Decomposition | 질문이 작은 범위의 문제로 나눠질 수 있을 떄              | 순차적 또는 병렬로 문제를 해결할 수 있는 서브 문제/질문으로 질문을 분해한다.                                                                         |
| Step-back     | 고수준의 개념적 이해가 필요할 때                     | 고수준의 개념과 원리에 대해서 일반적인 step-back 질문을 LLM에 prompt하고 그들에 대한 사실을 retrieve한다. 이 배경지식을 통해서 유저에게 대답한다.                      |
| HyDE          | 유저 입력만으로 관련된 문서를 retrieving하는 것이 어려울 때 | 질문을 질문을 답하는 가상의 문서로 변환하는데 LLM을 사용한다. 내재된 가상의 문서를 전제로 실제 문서를 retrieve하는데 사용한다. 전제는 문서와 문서간의 유사도 검색을 통해서 신뢰성을 높일 수 있다. |

### Routing

두번쨰로, RAG 시스템에 적합한 데이터 소스를 고려해보겠다. 정형 데이터 소스부터 비정형까지 하나 이상의 데이터베이스에 질의하고 싶을 것이다. LLM이 입력을 검토하고 적절한 데이터 소스에 접근하는 것은 소스에 질의하는 단순하고 효과적인 접근이다.

| Name              | When to use                                  | Description                                                                  |
|-------------------|----------------------------------------------|------------------------------------------------------------------------------|
| Logical routing   | LLM에게 입력을 어디로 route할지 정하는 prompt를 생성할 수 있을 때 | Logical routing은 LLM이 어떤 데이터 저장소로 질의하고 선택하게 할지 추론할 수 있게 한다.                  |
| Semantic routing  | semantic similarity가 입력을 어디로 route할지 효과적일 때  | Semantic routing은 질의와 prompts의 집합을 가지고 있다. similarity에 기반해서 적절한 prompt를 고른다. |


### Query Construction

세번째로, 데이터 소스가 특정한 query formats을 필요로 하는지이다. 많은 정형 데이터베이스들은 SQL을 사용한다. Vector stores는 보통 문서 메타데이터에 키워드 필터를 적용하는 특정 구문이 있다. 자연어 질의를 질의 구문으로 변환할 때 LLM을 사용하는 것은 유명하고 강력한 접근이다. 특히 text-to-SQL, text-to-Cypher와 query analysis for metadata filters들은 정형화되고 그래프, 벡터 데이터베이스 각각에 효과적으로 소통하는 방법이다.

### Indexing

네번쨰로, document index의 디자인을 살펴보겠다. LLM에 전달하는 documents의 retrieval을 위해서 index하는 documents를 분해하는 것이다. Indexing은 vector stores와 함께 내재된 models을 사용한다. vector stores는 고정된 vector로 documents안의 의미를 담은 정보를 압축한다.

많은 RAG 접근들이 documents를 chunks로 쪼개고 LLM을 위해서 입력 질문에서 유사도를 기반으로 몇가지 숫자를 retrieving한다. 그러나 chunk 크기와 chunk 숫자는 설정하기가 어렵고 LLM에게 질문에 대한 전체 맥락을 전달하지 못한다면 원하는 결과를 주지 못할 수 있다. 더 나아가, LLMs은 수백만개의 tokens을 처리할 수 있게한다.

두가지 방법이 답을 찾아준다. 
- multi vector : LLM이 documents를 indexing에 적합한 어떠한 형식으로 주는 retriever이다. 그러나 생성을 위해서 LLM에게 전체 documents를 리턴한다.
- ParentDocument : document chunks를 내포하면서 전체 documents를 리턴한다. retrieval을 위한 간결한 대표성 정보(요약 또는 chunks)를 전달한다.

| Name                       | 	Index Type                   | Uses an LLM               | When to use                                                        | Description                                                                                                               |
|----------------------------|-------------------------------|---------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Vector store               | Vector store                  | No                        | 쉽고 단순한 결과를 얻을 때 사용한다.                                              | 가장 단순한 메서드이고 시작하기가 가장 단순하다. 각 단어의 조각에서 embeddings을 생성하는 것을 포함한다.                                                          |
| ParentDocument             | Vector store + Document Store | No                        | pages가 직접 indexed할 수 있는 많은 숫자의 개별 작은 조각을 가지면서 모든 것을 retrieved해야할 때 | 각 document를 위해서 다양한 chunks를 indexing하는 것을 포함한다. embedding 공간에서 가장 유사한 chunks를 찾고 전체 parent document를 retrieve하고 그것을 리턴한다. |
| Multi Vector               | Vector store + Document Store | Sometimes during indexing | 텍스트 자체보다는 index할 연관된 documents로부터 정보를 추출할 때 사용한다.                  | 각 document에 대한 많은 vectors를 생성하는 것을 포함한다. 각 vector는 텍스트의 요약과 가상의 질문을 통해서 무수히 생성된다.                                         |
| Time-Weighted Vector store | Vector store                  | No                        | documents에 timestamps가 포함되고 가장 최근의 것을 retrieve할 때 사용한다.            | 문맥적(semantic) 유사도와 최근의 조합을 바탕으로 documents를 부른다.                                                                           |

5번째로, 유사도(similarity) 검색의 질을 향상시키는 것을 고려한다. 내포된 models은 텍스트를 document의 의미있는 컨텐트를 가지고 있는 고정 길이(vector) 대표 정보로 압축한다. 압축은 검색/retrieval에 유용하지만, document의 의미있는 늬앙스/상세 정보를 단일 vector 대표정보에 넣기 위해서 큰 부담을 전가한다. 몇가지 경우에는, 관계 없고 불필요한 컨텐트가 embedding의 의미있는 유용성을 희석시킨다.

ColBERT는 세분화된 embeddings으로 문제를 해결하는 방식이다. 

1. document와 query의 각 token에 문맥적으로 영향력 있는 embedding을 생성한다.
2. 각 query token과 모든 document tokens에 유사도를 점수 매긴다.
3. 최대값을 가져온다.
4. 모든 query tokens에 반복한다.
5. query-document similarity 점수를 가져오기 위해서 모든 query tokens의 최대 점수(3번 과정)의 합을 가져온다. 
6. 좋은 결과가 나온다.

retrieval의 질을 향상시키는 몇가지 트릭이 있다. Embeddings은 의미있는 정보를 가져오는 것을 향상시키지만, 키워드 형식의 queries에서는 성능을 잘 발휘하지 못한다. 많은 vector stores는 키워드와 의미 유사도를 섞어서 두개의 장점을 취한 빌트인 hybrid search를 제공한다. 더 나아가, 많은 vector stores는 maximal marginal relevance(MMR)가 있고 유사하고 불필요한 documents를 결과로 리턴하는 것을 피하게하고 결과를 다양하게 한다.

### Post-processing

여섯번째로, retrieved된 documents를 필터링하고 순위 매기는 것을 살펴보겠다. 다양한 소스로부터 리턴된 documents를 합칠 때 도움이 된다. 관련성이 적은 documents를 down-rank하고 유사한 documents를 압축한다.

| Name                   | 	Index Type | Uses an LLM | When to use                                                                          | Description                                                                                               |
|------------------------|-------------|-------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Contextual Compression | Any         | Sometimes   | retrieved된 결과가 관련성이 적은 정보가 너무 많이 포함되고 그것이 LLM에 혼동을 줄때 사용한다.                          | 다른 retriever의 post-processing 단계에 진행되고 retrieved된 documents의 가장 관련성 높은 정보만 추출한다. embeddings또는 LLM에서 실행된다. |
| Ensemble               | Any         | 	No         | 다양한 retrieval 메서드가 있고 그들을 합칠 때 사용한다.                                                 | 다양한 retrievers로부터 documents를 가져오고 그들을 합친다.                                                                |
| Re-ranking             | Any         | Yes         | 관련성을 바탕으로 retrieved된 documents를 순위매기는 것을 원할 때, 특히 다양한 retrieval 메서드에서 결과를 합칠 때 사용한다. | query와 documents가 주어졌을 때, Rerank는 query에 의미가 가장 관련있는 순서로 documents를 indexes한다.                            |

### Generation

마지막으로, RAG 시스템에서 자체 수정을 만드는 방법을 고려해보겠다. RAG 시스템은 저품질의 retrieval(유저의 질문이 index내의 domain 밖이다.)  또는 생성 과정에서의 환영(hallucinations)에서 고통 받을 수 있다. retrieve 생성 파이프라인에서는 이런 종류의 에러를 감지하거나 자체적으로 수정할 능력이 없다. "flow engineering"의 개념은 [context of code generation](https://arxiv.org/abs/2401.08500)에서 소개되었다. 오류를 확인하고 자체 수정을 하기 위해서 단위 테스트를 통해 code question에 대답하는 것을 반복한다. 몇가지 작업은 Self-RAG와 Corrective-RAG에 추가되었다. 두가지 경우에, document 관련성, 환영과 답변 질을 체크가 RAG 답변 생성 플로우 중에 수행된다.

| Name           | When to use                     | Description                                                                                               |
|----------------|---------------------------------|-----------------------------------------------------------------------------------------------------------|
| Self-RAG       | 환영 또는 관련 없는 콘텐트를 고칠 필요가 있을 때    | Self-RAG는 document 관련성, 환영과 RAG 답변 생성 플로우 중의 답변 질 확인을 실행한다. 답변을 만들고 자체                                    |
| Corrective-RAG | 관련성 적은 docs를 위한 fallback이 필요할 때 | Corrective-RAG은 retrieved된 documents가 query에 관련이 없을 때 web search와 같은 fallback을 포함한다. 높은 품질과 높은 관련성을 보장한다. |

## Text splitting

LangChain은 다양한 종류의 text splitters를 지원한다. `langchain-text-splitters` 패키지에서 확인 가능하다.

https://python.langchain.com/v0.2/docs/concepts/#text-splitting