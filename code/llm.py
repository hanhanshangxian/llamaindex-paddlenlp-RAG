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
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever


# 设置设备
paddle.set_device('gpu:5')

# 一、加载本地模型
model_name = "/home/wwhh/RAGtest2/RAGtest/models/checkpoint-699"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer is None:
    raise ValueError("Tokenizer加载失败，请检查路径是否正确。")
model = AutoModelForCausalLM.from_pretrained(model_name)
if model is None:
    raise ValueError("模型加载失败，请检查路径是否正确。")
model.eval()

# 二、自定义LLM类
class OurLLM(CustomLLM):
    context_window: int = 2048
    num_output: int = 256
    model_name: str = "custom"
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_template = (
            "请根据以下上下文，提供简明且准确的回答。\n"
            "上下文信息：\n"
            "{context_str}\n"
            "问题：{query_str}\n"
            "回答应直接回答问题，避免重复。\n"
            "回答：\n"
        )
        
        context_str = kwargs.get("context_str", "")
        query_str = kwargs.get("query_str", prompt)
        formatted_prompt = prompt_template.format(context_str=context_str, query_str=query_str)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pd")
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0, top_k=50, top_p=0.9)
        token_ids = outputs[0].numpy().flatten()
        response = tokenizer.decode(token_ids, skip_special_tokens=True)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)

# 设置使用本地模型
Settings.llm = OurLLM()

# 三、加载embedding模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="/home/wwhh/RAGtest2/RAGtest/models/bge-large-zh-v1.5"
)

# # 四、加载文档并生成节点
# splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
# filepaths = [os.path.join(dirpath, name) for dirpath, dirnames, filenames in os.walk("/home/wwhh/RAGtest2/RAGtest/data") for name in filenames]

# docs = []
# for filepath in filepaths:
#     _docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
#     _docs[0].doc_id = os.path.basename(filepath).split('.txt')[0]
#     docs.extend(_docs)

# # 解析文档生成节点
# nodes = splitter.get_nodes_from_documents(docs)

# # 五、存储上下文和索引构建
# docstore = SimpleDocumentStore()
# docstore.add_documents(nodes)

# # 初始化其他存储结构
# index_store = SimpleIndexStore()
# vector_store = SimpleVectorStore()
# graph_store = SimpleGraphStore()

# # 创建 StorageContext 并构建索引
# storage_context = StorageContext(
#     docstore=docstore,
#     index_store=index_store,
#     vector_stores={"default": vector_store},
#     graph_store=graph_store
# )

# response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)
# doc_summary_index = DocumentSummaryIndex.from_documents(
#     docs,
#     llm=OurLLM(),
#     transformations=[splitter],
#     response_synthesizer=response_synthesizer,
#     show_progress=True,
# )

# # 六、创建文档摘要索引和持久化存储
# doc_summary_index.storage_context.persist("index")


# 加载存储上下文
storage_context = StorageContext.from_defaults(persist_dir="index")
doc_summary_index = load_index_from_storage(storage_context)


# 七、配置查询引擎
summaryindexretriever = DocumentSummaryIndexLLMRetriever(
    doc_summary_index,
    choice_select_prompt=None,
    choice_batch_size=1000,
    choice_top_k=3,
    format_node_batch_fn=None,
    parse_choice_select_answer_fn=None,
)



class SimpleBM25Retriever(BM25Retriever):
    @classmethod
    def from_defaults(cls, index, similarity_top_k, **kwargs) -> "BM25Retriever":
        docstore = index.docstore
        return BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=5,
            verbose=True,
            tokenizer=chinese_tokenizer,
            **kwargs
        )

retriever = QueryFusionRetriever(
    [summaryindexretriever, SimpleBM25Retriever],
    retriever_weights=[0.6, 0.4],
    similarity_top_k=10,
    num_queries=1,  # set this to 1 to disable query generation
    mode="dist_based_score",
    use_async=True,
    verbose=True,
)

# class SimpleFusionRetriever(QueryFusionRetriever):
#     def __init__(self, vector_index, top_k=2, mode=FUSION_MODES.DIST_BASED_SCORE):
#         self.top_k = top_k
#         self.mode = mode

#         # Build vector retriever from vector index
#         self.summaryindexretriever = DocumentSummaryIndexLLMRetriever(
#             doc_summary_index,
#             choice_select_prompt=None,
#             choice_batch_size=50,
#             choice_top_k=1,
#             format_node_batch_fn=None,
#             parse_choice_select_answer_fn=None,
#         )

#         # Build BM25 retriever from document storage
#         self.bm25_retriever = SimpleBM25Retriever.from_defaults(
#             index=vector_index,
#             similarity_top_k=top_k,
#         )

#         super().__init__(
#             [self.summaryindexretriever, self.bm25_retriever],
#             retriever_weights=[0.6, 0.4],
#             similarity_top_k=top_k,
#             num_queries=1,  # set this to 1 to disable query generation
#             mode=mode,
#             use_async=True,
#             verbose=True,
#         )

# response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)

query_engine = RetrieverQueryEngine.from_args(retriever)

# 八、聊天循环
def chat_loop_with_query_engine():
    print("欢迎使用自定义聊天引擎！输入'退出'以结束对话。")
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("聊天已结束，再见！")
            break
        
        try:
            response = query_engine.query(user_input)
            print(f"回答: {response.response}")
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    chat_loop_with_query_engine()
