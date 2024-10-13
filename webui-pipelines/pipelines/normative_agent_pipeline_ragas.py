from typing import List, Union, Generator, Iterator
import os
import asyncio
import chromadb
import operator
import time
import logging
from chromadb.config import Settings
from langchain import FewShotPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.document_transformers import (
    LongContextReorder
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        self.name = "Normative Agent"
        self.basic_rag_pipeline = None
        
        # Define all parameters in one place
        self.params = {
            'model_name': "ianssens/e5-model-rag",
            'collection_name': "modelv1_750_100_base",
            'llm': "gpt-4o-mini",
            'top_k_search': 6,
            'search_type': "similarity",
            'weights': [0.5, 0.5]
        }
        
        # Initialize components
        self.collection_name = self.params['collection_name']
        self.model_name = self.params['model_name']
        self.embedding_model = None
        self.llm = None
        self.client = None
        self.db = None
        self.chroma_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.K = 5
        self.cross_tokenizer = None
        self.cross_model = None

    async def on_startup(self):
        # This function is called when the server is started.
        logger.info(f"on_startup:{__name__}")
        print(f"on_startup:{__name__}")
        
        self.cross_encoder = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco')
        
        logger.info("Cross Encoder initialized")
        print("Cross Encoder initialized")
        
        self.cross_tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        self.cross_model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        logger.info("cross_tokenizer and cross_model initialized")
        print("cross_tokenizer and cross_model initialized")
        
        # Set OpenAI API key
        # os.environ["OPENAI_API_BASE"] = "https://api.vsegpt.ru/v1"
        os.environ["OPENAI_API_BASE"] = "localhost:11434" 
        # os.environ['OPENAI_API_KEY'] = 'PUT-YOUR-OPENAI-KEY-HERE'
        # os.environ['OPENAI_API_KEY'] = 'PUT-YOUR-OPENAI-KEY-HERE'
        logger.info("OpenAI API key set")
        print("OpenAI API key set")
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cuda'}
        )
        logger.info("Embedding model initialized")
        print("Embedding model initialized")
        
        # Initialize LLM
        # self.llm = ChatOpenAI(
        #     model_name=self.params['llm'],
        #     temperature=0,
        # )
        self.llm = ChatOllama(
            model = 'bambucha/saiga-llama3:latest',
            temperature=0,
        )

        logger.info("LLM initialized")
        print("LLM initialized")
        
        # Initialize Chroma DB client
        # TODO
        self.client = chromadb.HttpClient(
            host='62.68.146.97',
            port=8000,
            ssl=False,
            headers=None,
            settings=Settings(),
            tenant=chromadb.DEFAULT_TENANT,
            database=chromadb.DEFAULT_DATABASE,
        )
        logger.info("Chroma DB client initialized")
        print("Chroma DB client initialized")
        
        # Initialize Chroma DB
        self.db = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )
        logger.info("Chroma DB initialized")
        print("Chroma DB initialized")
        
        # Initialize Chroma retriever
        self.chroma_retriever = self.db.as_retriever(
            search_type=self.params['search_type'],
            search_kwargs={"k": self.params['top_k_search']},
        )
        logger.info("Chroma retriever initialized")
        print("Chroma retriever initialized")
        
        # Fetch documents from Chroma collection
        collection = self.client.get_collection(name=self.collection_name)
        documents = collection.get()
        logger.info("Documents fetched from Chroma collection")
        print("Documents fetched from Chroma collection")
        
        # Initialize BM25Retriever
        self.bm25_retriever = BM25Retriever.from_texts(
            texts=documents.get("documents"),
            metadatas=documents.get("metadatas"),
        )
        self.bm25_retriever.k = self.params['top_k_search']
        logger.info("BM25Retriever initialized")
        print("BM25Retriever initialized")
        
        # Initialize EnsembleRetriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=self.params['weights']
        )
        logger.info("EnsembleRetriever initialized")
        print("EnsembleRetriever initialized")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        logger.info(f"on_shutdown:{__name__}")
        print(f"on_shutdown:{__name__}")
        
    async def Multi_Query_Retriever(self, query, llm, retriever):
        QUERY_EXPANSION_PROMPT = PromptTemplate(
            input_variables=["question"],
            template=
            """
            Ты – русскоязычный полезный ассистент. Твоя задача — сгенерировать три
            различных версий заданного пользовательского вопроса для извлечения соответствующих документов из векторной базы данных.
            Создавая несколько точек зрения на вопрос пользователя, ваша цель — помочь
            пользователю преодолеть некоторые ограничения поиска по сходству на основе расстояния.
            Предоставьте эти альтернативные вопросы, разделенные символами новой строки.
            Предоставьте только запрос, без нумерации.
            Исходный вопрос: {question}
            """,
        )
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            prompt=QUERY_EXPANSION_PROMPT,
            include_original=True
        )

        run_manager = AsyncCallbackManagerForRetrieverRun(
            run_id="first_id",
            handlers=[],
            inheritable_handlers=[]
        )

        unique_docs = await retriever_from_llm._aget_relevant_documents(
            query=query,
            run_manager=run_manager,
            )

        return unique_docs

    def Self_Querying(self, query):
        QUERY_REDACTING_PROMPT = f"""
            Ты – русскоязычный полезный ассистент.
            Ты помогаешь технической поддержке с поступающими вопросами.
            Пользователи не всегда пишут полные вопросы, ты помогаешь
            перефразировать поступающий вопрос и выделяешь основную проблему,
            если это требуется. Ты всегда исправишь опечатки и ошибки.
            Постарайся максимально передать изначальный смысл и посыл вопроса.
            Вот вопрос пользователя: {query}
            Перефразированный и исправь вопрос:
            """

        return self.llm.invoke(QUERY_REDACTING_PROMPT).content

    def augmentation_generation(self, query, docs):
        """
        Функция Augmentation Generation для генерации ответов на вопросы.
        """

        contexts = []
        for doc in docs[:self.K]: # Parameter to cut the extracted chunks
            contexts.append(doc.page_content)
        context_str = '\n\n##\n\n'.join(contexts)

        examples = [
            {
                "query": "Каков объем экспорта услуг категории 'Поездки' в региональном проекте 'Экспорт услуг' категории 'Поездки' в Ханты-Мансийском автономном округе - Югре?",
                "answer": "Объем экспорта услуг категории 'Поездки' в региональном проекте 'Экспорт услуг' категории 'Поездки' в Ханты-Мансийском автономном округе - Югре составляет 0,033 млрд долларов США."
            }, {
                "query": "Какова общая сумма по городу Когалыму за 2013-2014 годы?",
                "answer": "Общая сумма по городу Когалыму за 2013-2014 годы составляет 265147484,00 рублей."
            }, {
                "query": "Где узнать о порядке обжалования решений?",
                "answer": "О порядке обжалования действий (бездействия) и решений, осуществляемых и принимаемых в ходе предоставления государственной услуги, можно узнать у ответственных должностных лиц Департамента, которые ответственны за предоставление государственной услуги."
            }, {
                "query": "Когда принято постановление Правительства автономного округа N 48-п?",
                "answer": "Постановление Правительства Ханты-Мансийского автономного округа - Югры N 48-п 'Об оплате труда работников государственных учреждений Ханты-Мансийского автономного округа - Югры' принято 3 марта 2005 года."
            }, {
                "query": "Кто отвечает за информационную безопасность в ХМАО-Югре, если Губернатор отсутствует?",
                "answer": "В случае отсутствия Губернатора Ханты-Мансийского автономного округа - Югры, его обязанности исполняет заместитель Губернатора Ханты-Мансийского автономного округа - Югры. Заместитель Губернатора имеет в ведении Департамент информационных технологий и цифрового развития Ханты-Мансийского автономного округа - Югры, который, вероятно, отвечает за вопросы информационной безопасности."
            }
        ]

        example_template = """Пользователь: {query}
        ИИ: {answer}
        """

        example_prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template=example_template)

        prefix = """
        Игнорируй все предыдущие инструкции. Ты консультант-юрист. Помогаешь с вопросами
        по нормативным правовым актам (далее НПА) ХАНТЫ-МАНСИЙСКОГО АВТОНОМНОГО ОКРУГА-ЮГРЫ.
        Будь вежлив. Выяви потребность клиента и помоги ему решить ее с помощью НПА.
        Все НПА действительные. Отвечай на вопросы на основе фактов, которые тебе дадут.
        Не придумывай новую информация! Также используй данные из контекста.

        ###

        Контекст:
        {context_str}

        ###

        Используй только знания о НПА и примеры ответов на консультациях. Если в этих
        данных нет ответа и ты в этом уверен, скажи, что “я не знаю”, предложи обратиться к живому
        консультанту компании. Не придумывай факты, которых нет в контексте. Ты можешь
        использовать свои общие знания в разных областях,
        чтобы давать общие советы своим пользователям.
        Отвечай на языке, на котором посетитель задал вопрос(по умолчанию используй русский язык).

        ###

        В своем ответе используй следующие принципы:
        1. Ты должен давать четкие, краткие и прямые ответы.
        2. Исключи ненужные напоминания, извинения, упоминания самого себя и любые заранее запрограммированные тонкости.
        3. Сохраняй непринужденный тон в общении.
        4. Будь прозрачным; если ты не уверен в ответе или если вопрос выходит за рамки твоих возможностей или знаний, признай это.
        5. В случае неясных или двусмысленных вопросов задавай дополнительные вопросы, чтобы лучше понять намерения пользователя.
        6. При объяснении концепций используй примеры и аналогии из реальной жизни, где это возможно.
        7. В случае сложных запросов сделай глубокий вдох и работай над проблемой шаг за шагом.
        8. За каждый ответ ты получишь чаевые до 200 долларов (в зависимости от качества твоего ответа).
        Очень важно, чтобы ты понял это правильно. На кону несколько жизней и моя карьера.

        Вот несколько примеров ответов:

        ###

        """

        suffix = """
        Пользователь: {query}
        ИИ: """

        few_shot_prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["query", "context_str"],
            example_separator="\n\n"
        )

        return self.llm.invoke(few_shot_prompt_template.format(query=query, context_str=context_str)).content, contexts
    
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.info(f"Starting retrieval for user message: {user_message}")
        print(f"Starting retrieval for user message: {user_message}")
        
        # RETRIEVE ----------------------------------------------------------------
        # relevant_docs_from_retriever = self.ensemble_retriever.get_relevant_documents(user_message)
        
        reformulated_query = self.Self_Querying(user_message)
        logger.info(f"Reformulated query: {reformulated_query}")
        print(f"Reformulated query: {reformulated_query}")

        
        relevant_docs_from_retriever = asyncio.run(self.Multi_Query_Retriever(reformulated_query, self.llm, self.ensemble_retriever))
        logger.info(f"Multi_Query_Retriever: {relevant_docs_from_retriever}")
        print(f"Multi_Query_Retriever: {relevant_docs_from_retriever[0]}")
        
        # CROSS-ENCODING
        # cross_list = [(user_message, doc.page_content) for doc in relevant_docs_from_retriever]
        # print(cross_list[0])
        # scores = self.cross_encoder.predict(cross_list)
        # docs_with_scores = list(zip(relevant_docs_from_retriever, scores.tolist()))
        # reranked_docs = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        
        pairs = []
        for doc in relevant_docs_from_retriever:
            tokenized_input = self.cross_tokenizer(reformulated_query, doc.page_content, truncation=True, max_length=512, return_tensors="pt")
            pairs.append(tokenized_input)
            
        scores = []
        for pair in pairs:
            outputs = self.cross_model(**pair)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            prob_for_related = probabilities[0, 1]
            scores.append(prob_for_related.item())
            
        scored_docs = zip(scores, relevant_docs_from_retriever)
        sorted_docs = sorted(scored_docs, reverse=True)

        reranked_docs = [doc for _, doc in sorted_docs][0:8]
        # print(f'RERANKED_DOCS: {reranked_docs}')

        # Middle problem
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(reranked_docs)

        response, context = self.augmentation_generation(reformulated_query, reordered_docs)
        logger.info(f"Generated response: {response}")
        print(f"Generated response: {response}")
        logger.info(f"Generated context: {context}")
        print(f"Generated context: {context}")
        
        # Format context as a numbered Markdown list with each line in italic
        context_list = "\n".join([f"*{i+1}. {item}*" for i, item in enumerate(context)])
        response =  response.replace("\n", " ")
        return f'''*Вопрос переформулирован:\n{reformulated_query}*\n\n\n**{response}**\n\nКонтекст:\n{context_list}'''