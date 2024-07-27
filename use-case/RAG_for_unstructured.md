# Build a Retrieval Augmented Generation (RAG) App

> https://python.langchain.com/v0.2/docs/tutorials/rag/

LLMs에 의해서 가능해진 가장 유용한 어플리케이션은 정교한 QnA 챗봇이다. 특정 원천 정보에 대한 질문을 대답할 수 있는 어플리케이션이다. 챗봇 어플리케이션은 Retrieval Augmented Generation라 불리는 기술을 사용한다.

이번 튜토리얼에서 텍스트 데이터에서 단순한 Q&A 어플리케이션을 만드는 방법을 보여주겠다. 그 과정에서 특정한 Q&A 아키텍처를 보고 더 나은 Q&A 기술을 위해서 필요한 추가 리소스를 보겠다.

# What is RAG?

RAG는 추가적인 데이터로 LLM 지식을 보완하는 기술이다. LLMs은 넓은 범위의 주제를 추론하지만, 그들이 그 시점에 학습한 특정한 공공 데이터에 대한 지식만 있다. 사적인 데이터나 모델의 출시일 이후의 데이터에 대한 AI 어플리케이션을 만드려면, 그것이 필요한 특정 데이터로 model의 지식을 보완할 필요가 있다. 적절한 데이터를 가져오는 과정과 model prompt에 그것을 넣는 것을 RAG라 한다.

LangChain은 Q&A 어플리케이션 빌드를 위해서 디자인된 컴포넌트가 많고 RAG은 더욱 일반적이다. 

# Concepts

일반적인 RAG는 두개의 주요 컴포넌트로 이루어진다.

1. Indexing : 소스로부터 데이터를 소화하고 indexing하는 파이프라인이다. 
2. Retrieval and generation : RAG 체인은 런타임에서 유저의 쿼리를 가져오고 index로부터 데이터를 가져오고 model에게 전달한다.

초기 데이터에서 답변으로의 단계는 다음과 같다.

## Indexing

1. Load : 우리의 데이터를 가져올 필요가 있다. Document Loader를 통해서 가져온다.
2. Split : Text splitters가 큰 `Documents`를 작은 chunks로 쪼갠다. 큰 청크는 검색하기가 힘들고 model의 제한된 context window에 적합하지 않기에, 작은 chunks 데이터는 indexing하고 모델에게 전달하기에 유리하다.
3. Store : 쪼개둔 데이터를 검색하기 위해서 저장하고 index할 공간이 필요하다.VectorStore와 Embeddings model을 이용해서 해결한다.

## Retrieval and generation

4. Retrieve : user 입력이 주어졌을 때, Retriever를 이용해서 저장소로부터 쪼개진 데이터를 retrieve한다.
5. Generate : ChatModel과 LLM에게 질문과 retrieve된 데이터를 포함한 prompt를 보내서 답변을 생성한다.

# Preview

웹사이트의 내용에 대해서 질문을 답하는 앱을 만들겠다. 웹사이트는 [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)를 사용한다. 

20줄 정도의 코드로 간단한 RAG 체인과 indexing 파이프라인을 만들어 보겠다.

```python

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")
```

# Detailed walkthrough

위의 코드를 단계별로 알아보겠다.

## 1. Indexing: Load

블로그 포스트 내용을 가져와야한다. source로부터 데이터를 가져오고 Document 리스트를 가져오는 DocumentLoaders를 사용할 것이다. `Document` 객체는 `page_content`와 `metadata`를 가진다.

WebBaseLoader를 사용할 것이다. web URLs로부터 `urllib`를 이용해서 HTML을 가져오고 `BeautifulSoup`를 이용해서 내용을 텍스트로 parse한다. HTML -> text 과정에서 `bs_kwargs` 파라미터를 통해서 약간의 커스터마이즈를 하겠다. 이 경우에는, HTML 태그 중에서 “post-content”, “post-title”, 또는 “post-header” 클래스를 가진 것만 가져오고 아닌 경우에는 모두 삭제한다.

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
```

## 2. Indexing: Split

불러온 document는 42k가 넘는 길이를 가진다. 이것은 models의 context window에 적합하지 않다. context window의 전체 포스트를 가져올 수 있더라도, models이 큰 입력에 대한 정보를 찾는 것이 힘들 것이다.

이것을 해결하기 위해서, `Document`를 embedding과 vector storage에 저장할 수 있도록 쪼갤 것이다. 런타임 이내에, 가장 관련있는 블로그 포스트의 조각을 가져올 것이다.

chunks간에 200자가 겹쳐진 1000자의 chunks로 documents를 나눌 것이다. 겹쳐진 부분은 관련된 중요한 context가 쪼개진 문장으로 부터 사라질 가능성을 완화한다. 개행문자와 같은 일반적인 구분자를 이용해서 document를 특정 크기에 도달할 때까지 재귀적으로 쪼개는RecursiveCharacterTextSplitter를 사용한다. 일반적인 text 사용에서 추천되는 text splitter이다.

`add_start_index=True`를 세팅한다. 초기 Document에서 쪼개진 개별 Document는 metadata 속성으로 "start_index"을 가진다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)
```

## 3. Indexing: Store

runtime 중에, 검색할 텍스트 chunks를 index할 필요가 있다. 각 document split의 내용을 embed하고 vector 데이터베이스에 이러한 embeddings을 넣는 것이다. splits을 검색할 때, 텍스트 검색 쿼리를 가져오고, embed하고 query embedding에 가장 유사한 embeddings을 찾기 위한 몇가지 유사도 검색을 한다. 가장 단순한 유사도 검색은 cosine 유사도이다. embeddings의 각 pair 사이에서 각의 cosine값을 가져온다.

Chroma vector store와 OpenAIEmbeddings model을 사용해서 단일 command안에 document splits을 embed하고 저장할 수 있다.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
```

## 4. Retrieval and Generation: Retrieve

실제 어플리케이션 로직을 시작한다. 유저의 질문을 가져오고 질문에 연관된 documents를 찾고 retrieved된 documents와 초기 질문을 model에게 전달하고 답변을 얻는다.

우선 documents를 검색하는 로직을 정의한다. LangChain은 주어진 문자열 쿼리로 관련된 Documents를 가져오는 index를 만드는 Retriever 인터페이스를 가지고 있다.

가장 일반적인 Retriever는 retrieval를 촉진하는 vector store의 유사도 검색을 사용하는 VectorStoreRetriever이다. VectorStore는 `VectorStore.as_retriever()`로 단순하게 튜닝될 수 있다.

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
```

### 5. Retrieval and Generation: Generate

질문을 가져오고 유사한 documents를 retrieves하고 prompt를 생성하고 model에게 전달하고 출력을 parses하는 모든 요소를 chain에 넣는다.

LCEL Runnable 프로토콜을 이용해서 chain을 정의할 것이다.
- 컴포넌트와 함수를 투명한 방법으로 연결한다.
- LangSmith가 자동으로 chain을 추적한다.
- 외부에서 streaming, async와  batched calling를 가져온다.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)
```

어떤 일이 일어나는지 보기 위해서 LCEL을 분해해보겠다.

각 components(retriever, prompt, llm)는 Runnalble 인스턴스이다. 그들은 모두 같은 메서드를 구현하고 있기에 연결이 용이하다. 그들은 RunnableSequence로 연결되고 `|` 연산자를 사용한다.

LangChain은 `|`를 만났을 때 모든 객체를 runnables로 캐스팅한다. `format_docs`는  RunnableLambda로 캐스팅되고 "context" 와 "question"을 가진 dict는 RunnableParallel로 치환된다. 

이제 입력 질문이 runnables위에서 어떻게 흘러가는지 보겠다.

위에서 보았듯이, prompt로의 입력은 "context"와 "question" key를 가진 dict이다. 그래서 chain의 첫 요소는 입력 질문으로부터 이 두개를 계산할 runnables을 생성한다.

- `retriever | format_docs`는 retriever를 통해서 질문을 전달하고 Document 객체를 만들고 `format_docs`가 문자열을 만든다.
- `RunnablePassthrough()`는 입력 질문을 변경되지 않은채로 전달한다.

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
)
```

위와 같이 만들고 `chain.invoke(question)` 실행은 formatted prompt를 만들고 추론할 준비가 된다.(개발 중에 이와 같이 만들면 테스트 하기에 실용적이다.)

chain의 마지막은 추론을 실행하는 llm이다. StrOutputParser()는 LLM의 출력 메시지를 문자열로 만들어준다.