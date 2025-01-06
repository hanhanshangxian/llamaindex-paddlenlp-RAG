import os
import paddlenlp
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices import DocumentSummaryIndex
from llama_index.core import Document, VectorStoreIndex,SimpleDirectoryReader, SimpleKeywordTableIndex
from llama_index.core import Settings,SummaryIndex,load_index_from_storage,StorageContext,Settings
from typing import Optional, List, Mapping, Any
import torch
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.response_synthesizers import get_response_synthesizer
from transformers.generation import GenerationConfig
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.indices.document_summary import DocumentSummaryIndexLLMRetriever
from llama_index.core.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core import QueryBundle
# import NodeWithScore
from llama_index.core.schema import NodeWithScore
# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)



paddle.set_device('gpu:0')

# 一、加载本地模型
model_name = "/home/wwhh/RAGtest2/RAGtest/models/checkpoint-699"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer is None:
    raise ValueError("Tokenizer加载失败，请检查路径是否正确。")
model = AutoModelForCausalLM.from_pretrained(model_name)
if model is None:
    raise ValueError("模型加载失败，请检查路径是否正确。")
model.eval() 



#二、自定义LLM类

class OurLLM(CustomLLM):
    context_window: int = 4096
    num_output: int = 256
    model_name: str = "custom"
 
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )
 
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 定义提示模板
        prompt_template = (
            "根据以下上下文回答问题。\n"
            "上下文信息：\n"
            "{context_str}\n"
            "问题：{query_str}\n"
            "回答：\n"
        )
        
        # 生成提示文本
        context_str = kwargs.get("context_str", "")
        query_str = kwargs.get("query_str", prompt)
        formatted_prompt = prompt_template.format(context_str=context_str, query_str=query_str)
        # print(f"格式化后的提示: {formatted_prompt}") # 调试
        # print(f"上下文: {context_str}")  # 调试
        # print(f"问题: {query_str}")  # 调试
        
        inputs = tokenizer(formatted_prompt, return_tensors="pd", truncation=True, max_length=self.context_window)
        # print(f"传递给模型的输入: {inputs}") # 调试
        outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.3, top_k=50, top_p=0.9, batch_size=1)
        token_ids = outputs[0].numpy().flatten()
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        # print(f"生成的输出: {outputs}") # 调试
        # print(f"生成的 tokens: {token_ids}") # 调试
        # print(f"解码的文本: {text}") # 调试
        return CompletionResponse(text=text)
 
    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError()


# 设置使用本地模型
Settings.llm = OurLLM()

# 三、加载embedding模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="/home/wwhh/RAGtest2/RAGtest/models/bge-large-zh-v1.5"
)

# 加载文档
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
filepaths = [os.path.join(dirpath, name) for dirpath, dirnames, filenames in os.walk("/home/wwhh/RAGtest2/RAGtest/data") for name in filenames]

docs = []
for filepath in filepaths:
    _docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
    _docs[0].doc_id = os.path.basename(filepath).split('.txt')[0]
    docs.extend(_docs)

# 创建向量和关键词索引
vector_index = VectorStoreIndex.from_documents(docs)
keyword_index = SimpleKeywordTableIndex.from_documents(docs)

# 将索引保存到磁盘
vector_index.set_index_id("vector_index")
vector_index.storage_context.persist("./indextest1")
keyword_index.set_index_id("keyword_index")
keyword_index.storage_context.persist("./indextest2")

# 重新构建存储上下文
storage_context1 = StorageContext.from_defaults(persist_dir="/home/wwhh/RAGtest2/RAGtest/models/indextest1")
storage_context2 = StorageContext.from_defaults(persist_dir="/home/wwhh/RAGtest2/RAGtest/models/indextest2")

# 从存储中加载索引
vector_index = load_index_from_storage(storage_context1, index_id="vector_index")
keyword_index = load_index_from_storage(storage_context2, index_id="keyword_index")

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        llm: CustomLLM,  # 加入模型
        mode: str = "OR",
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._llm = llm  # 初始化模型
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        if not vector_nodes and not keyword_nodes:
            # 如果没有找到任何上下文，使用模型直接生成回答
            print("没有找到任何匹配的节点，使用模型生成回答。")
            context_str = ""
            query_str = query_bundle.query_str
            response = self._llm.complete(query_str, context_str=context_str)
            return [NodeWithScore(node=None, score=1.0, text=response.text)]

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

# define custom retriever
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever, Settings.llm)

# define response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

# vector query engine
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)
# keyword query engine
keyword_query_engine = RetrieverQueryEngine(
    retriever=keyword_retriever,
    response_synthesizer=response_synthesizer,
)


def chat_loop_with_query_engine():
    print("欢迎使用自定义聊天引擎！输入'退出'以结束对话。")

    while True:
        user_input = input("\n你: ")

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("聊天已结束，再见！")
            break

        try:
            query_bundle = QueryBundle(query_str=user_input)
            nodes = custom_retriever._retrieve(query_bundle)
            
            if nodes and nodes[0].node:  # 如果找到了上下文
                context = nodes[0].node.text
                response = Settings.llm.complete(user_input, context_str=context)
                print(f"回答: {response.text}")
            else:  # 如果没有找到任何节点或上下文为空
                print("没有找到相关的上下文，使用模型生成回答。")
                response = Settings.llm.complete(user_input)
                print(f"回答: {response.text}")
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    chat_loop_with_query_engine()