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
import jieba
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

paddle.set_device('gpu:3')

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
    num_output: int = 1024
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
        "你是一位专业的银行业务人员，请根据以下提供的上下文内容结合自身知识，准确、清晰地回答客户的问题。"
        "请务必提供详尽的回答，列出关键点及必要步骤，不得进行任何概括或简化描述。"
        "若问题涉及金融法规或规定，请准确引用并说明相关规定的具体名称和内容。\n"
        "【上下文】:\n"
        "{context_str}\n"
        "【客户问题】:\n"
        "{query_str}\n"
        "【专业回答】:\n"
        )

        # prompt_template = (
        #     "根据以下上下文回答问题。\n"
        #     "上下文信息：\n"
        #     "{context_str}\n"
        #     "问题：{query_str}\n"
        #     "回答：\n"
        # )
        
        # 生成提示文本
        context_str = kwargs.get("context_str", "")
        query_str = kwargs.get("query_str", prompt)
        formatted_prompt = prompt_template.format(context_str=context_str, query_str=query_str)
        print(f"格式化后的提示: {formatted_prompt}") # 调试
        # print(f"上下文: {context_str}")  # 调试
        # print(f"问题: {query_str}")  # 调试
        
        inputs = tokenizer(formatted_prompt, return_tensors="pd", truncation=True, max_length=self.context_window)
        # print(f"传递给模型的输入: {inputs}") # 调试
        outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.3, top_k=50, top_p=0.7, batch_size=1)
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

# splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
# filepaths = [os.path.join(dirpath, name) for dirpath, dirnames, filenames in os.walk("/home/wwhh/RAGtest2/RAGtest/data") for name in filenames]

# docs = []
# for filepath in filepaths:
#     _docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
#     _docs[0].doc_id = os.path.basename(filepath).split('.txt')[0]
#     docs.extend(_docs)

# index = VectorStoreIndex.from_documents(
#     docs, transformations=[splitter], show_progress=True
# )

# index.set_index_id("vector_index")
# index.storage_context.persist("./testindex")

storage_context1 = StorageContext.from_defaults(persist_dir="/home/wwhh/RAGtest2/RAGtest/models/testindex")

index = load_index_from_storage(storage_context1, index_id="vector_index")

vector_retriever = index.as_retriever(similarity_top_k=5)

# 自定义中文BM25检索
def chinese_tokenizer(text: str):
    return list(jieba.cut(docs))

class SimpleBM25Retriever(BM25Retriever):
    @classmethod
    def from_defaults(cls, index, similarity_top_k, **kwargs):
        nodes = list(index.docstore.docs.values())
        return BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=10, verbose=True,
            tokenizer=chinese_tokenizer, **kwargs
        )

bm25_retriever = SimpleBM25Retriever.from_defaults(
    index=index,
    similarity_top_k=10
)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    retriever_weights=[0.6, 0.4],
    similarity_top_k=10,
    num_queries=1,  # set this to 1 to disable query generation
    mode="dist_based_score",
    use_async=True,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(retriever)


def match_and_correct(output_text, query_engine, reference_text):
    splitter = SentenceSplitter()  # Create an instance

    # Extract the text content from the output
    document = Document(text=output_text)  # Create Document object
    text_content = document.text  # Get the text content
    
    # Split the text into chunks
    chunks = splitter.split_text(text_content)  
    corrected_chunks = []
    
    # Loop through each chunk to retrieve relevant information
    for chunk in chunks:
        response = query_engine.query(QueryBundle(reference_text))
        best_match = response.response if response.response else chunk
        corrected_chunks.append(best_match)
    
    print(corrected_chunks)

    # Combine the corrected text with the original query
    combined_input = f"{reference_text}\n{''.join(corrected_chunks)}"
    return combined_input  # Return the combined input instead of just corrected text

def chat_loop_with_query_engine():
    print("欢迎使用自定义聊天引擎！输入'退出'以结束对话。")
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("聊天已结束，再见！")
            break
        try:
            response = query_engine.query(user_input)
            generated_text = response.response if response else "没有生成文本"
            print("模型生成的文本：", generated_text)

            
            # Use query engine to correct generated text
            combined_input = match_and_correct(generated_text, query_engine, user_input)
            print("矫正后的输入：", combined_input)
            
            # Re-query the model with the combined input
            new_response = query_engine.query(combined_input)
            print("模型重新生成的文本：", new_response.response)

        except Exception as e:
            import traceback
            print(f"发生错误: {e}")
            traceback.print_exc()  # 输出完整的堆栈跟踪

if __name__ == "__main__":
    chat_loop_with_query_engine()

# def chat_loop_with_query_engine():
#     print("欢迎使用自定义聊天引擎！输入'退出'以结束对话。")
    
#     while True:
#         user_input = input("\n你: ")
        
#         if user_input.lower() in ["退出", "exit", "quit"]:
#             print("聊天已结束，再见！")
#             break
        
#         try:
#             response = query_engine.query(user_input)
#             print(f"回答: {response.response}")
#         except Exception as e:
#             print(f"发生错误: {e}")

# if __name__ == "__main__":
#     chat_loop_with_query_engine()