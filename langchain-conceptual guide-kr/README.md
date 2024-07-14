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