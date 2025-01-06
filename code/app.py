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


paddle.set_device('gpu:1')

# 一、加载本地模型
model_name = "/home/wwhh/RAGtest2/RAGtest/models/checkpoint-273"
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
        outputs = model.generate(**inputs, max_new_tokens=500, temperature=0, top_k=50, top_p=0.9, batch_size=1)
        token_ids = outputs[0].numpy().flatten()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
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
# response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)
# doc_summary_index = DocumentSummaryIndex.from_documents(
#     docs,
#     llm=OurLLM(),
#     transformations=[splitter],
#     response_synthesizer=response_synthesizer,
#     show_progress=True,
# )

# doc_summary_index.storage_context.persist("index")



# 加载存储上下文
storage_context = StorageContext.from_defaults(persist_dir="index")
doc_summary_index = load_index_from_storage(storage_context)





# 八、配置查询引擎

retriever = DocumentSummaryIndexLLMRetriever(
    doc_summary_index,
    choice_select_prompt=None,
    choice_batch_size=10000,
    choice_top_k=5,
    format_node_batch_fn=None,
    parse_choice_select_answer_fn=None,
)

response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)

# # 自定义的提示模板，用于生成独立问题
# custom_prompt = PromptTemplate(
#     """\
#     请根据以下人类和助理的对话历史以及人类的后续消息，将该消息重写为一个独立的、完整的问题，并且包含所有相关的上下文。

#     <对话历史>
#     {chat_history}

#     <后续问题>
#     {question}

#     <独立问题>
#     """
# )

# # 聊天历史记录
# custom_chat_history = [
#     ChatMessage(
#         role=MessageRole.USER,
#         content="Hello assistant, we are having a insightful discussion about Paul Graham today.",
#     ),
#     ChatMessage(role=MessageRole.ASSISTANT, content="Okay, sounds good."),
# ]


query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)


# # 五、交互式对话
# def interactive_chat():
#     print("开始对话！输入你的问题，输入 'exit' 退出。")
#     while True:
#         user_input = input("你：")
#         if user_input.lower() == "exit":
#             print("对话结束。")
#             break
#         # 使用 query_engine 处理用户输入并生成回复
#         response = query_engine.query(user_input)
#         print(f"助理：{response}")

# # 启动对话
# interactive_chat()



# # 实例化 CondenseQuestionChatEngine
# chat_engine = CondenseQuestionChatEngine.from_defaults(
#     query_engine=query_engine,
#     condense_question_prompt=custom_prompt,
#     chat_history=custom_chat_history,
#     verbose=True,
# )

# chat_engine.chat_repl()


#九、启动聊天引擎



                        
# chat_engine = doc_summary_index.as_chat_engine(
#     chat_mode="condense_plus_context", 
#     llm=Settings.llm, 
#     context_prompt=(
#         "你是一位金融领域的智能助手，能够与用户进行正常的交互，"
#         "并就金融领域的相关话题进行讨论。"
#         "以下是与问题相关的文档内容供你参考：\n"
#         "{context_str}"
#         "\n指令：请结合以上文档内容，或之前的对话记录，帮助用户解答他们的金融相关问题。"
#     ),
#     verbose=False
#     )

# chat_engine.chat_repl()



