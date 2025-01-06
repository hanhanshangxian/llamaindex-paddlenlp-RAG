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
import re
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import concurrent.futures
from paddlenlp.peft import LoRAModel, LoRAConfig 

paddle.set_device('gpu:7')

# 一、加载本地模型
# # 1. 设置模型路径
# model_path = "/home/wwh/PaddleNLP/llm/checkpoints/lora_ckpts1"
# base_model_path = "/home/wwh/xuanyuan-13b-chat"

# # 2. 加载分词器
# tokenizer = AutoTokenizer.from_pretrained(
#     base_model_path,
#     vocab_file=f"{base_model_path}/tokenizer.model"
# )

# # 确保 pad_token_id 被正确设置
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# # 3. 加载基础模型
# base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# # 4. 加载 LoRA 配置和模型
# config = LoRAConfig.from_pretrained(model_path)
# model = LoRAModel.from_pretrained(base_model, model_path)

# # 5. 设置模型为评估模式
# model.eval()

# 一、加载本地模型
model_name = "/home/wwh/PaddleNLP/llm/checkpoints/llama_sft_ckpts2/checkpoint-233"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer is None:
    raise ValueError("Tokenizer加载失败，请检查路径是否正确。")
model = AutoModelForCausalLM.from_pretrained(model_name)
if model is None:
    raise ValueError("模型加载失败，请检查路径是否正确。")
model.eval() 


#二、自定义LLM类

class OurLLM(CustomLLM):
    context_window: int = 8192
    num_output: int = 4096
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
        # prompt_template = (
        # "上下文信息如下。\n"
        # "{context_str}\n"
        # "在回答用户查询时，仅参考上述上下文信息，而不依赖于先前知识。\n"
        # "查询：{query_str}\n"
        # "答案：\n"
        # )

        # prompt_template = (
        # "根据自身已有知识，并参考以下内容,提供准确、详尽的解答。\n"
        # "务必直接回答问题。避免重复输出。\n"
        # "回复如果过短，必须根据自己的知识进行补充。\n"
        # "【上下文】:\n"
        # "{context_str}\n"
        # "【客户问题】:\n"
        # "{query_str}\n"
        # "【专业回答】:\n"
        # )

        prompt_template = (
            "作为专业的银行业务人员，请基于以下上下文内容和自身的专业知识，为客户提供准确、详尽的解答。"
            "回答中请明确列出所有关键点和步骤，并通过具体的示例帮助客户理解。避免任何概括或简化描述。"
            "如果问题涉及到金融法规或政策，请准确引用相关的规定，标明具体名称及内容。\n"
            "如果没有上下文内容，请输出无法回答。\n"
            "【上下文】:\n"
            "{context_str}\n"
            "【客户问题】:\n"
            "{query_str}\n"
            "【专业回答】:\n"
        )

        # prompt_template = (
        #     "根据以下上下文详细回答问题，确保包括所有关键细节和具体要求。\n"
        #     "上下文信息：\n"
        #     "{context_str}\n"
        #     "问题：{query_str}\n"
        #     "回答：\n"
        # )
        # prompt_template = (
        #     "作为一名专业银行业务人员，请详细回答以下问题。回答时请参考上下文内容，"
        #     "不要重复或简化。请引用具体条款和细则，以便用户理解。\n"
        #     "【上下文】:\n"
        #     "{context_str}\n"
        #     "【客户问题】:\n"
        #     "{query_str}\n"
        #     "【回答】:\n"
        #     )
        
        # 生成提示文本
        # qa_template = PromptTemplate(prompt_template)
        context_str = kwargs.get("context_str", "")
        query_str = kwargs.get("query_str", prompt)
        formatted_prompt = prompt_template.format(context_str=context_str, query_str=query_str)
        print(f"格式化后的提示: {formatted_prompt}") # 调试
        # print(f"上下文: {context_str}")  # 调试
        # print(f"问题: {query_str}")  # 调试
        
        inputs = tokenizer(formatted_prompt, return_tensors="pd", truncation=True, max_length=8192, batch_size=1)
        # print(f"传递给模型的输入: {inputs}") # 调试
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_k=50, top_p=0.9)
        token_ids = outputs[0].numpy().flatten()
        text = tokenizer.decode(token_ids, skip_special_tokens=True).replace("</s>", "").strip()
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
    model_name="/home/wwh/RAGtest2/RAGtest/models/bge-large-zh-v1.5"
)

splitter = SentenceSplitter(chunk_size=256, chunk_overlap=100)
filepaths = [os.path.join(dirpath, name) for dirpath, dirnames, filenames in os.walk("/home/wwh/RAGtest2/RAGtest/txt") for name in filenames]

docs = []
for filepath in filepaths:
    _docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
    _docs[0].doc_id = os.path.basename(filepath).split('.txt')[0]
    docs.extend(_docs)

index = VectorStoreIndex.from_documents(
    docs, transformations=[splitter], show_progress=True
)

index.set_index_id("vector_index")
index.storage_context.persist("./testindex")

storage_context1 = StorageContext.from_defaults(persist_dir="/home/wwh/RAGtest2/RAGtest/models/testindex")

index = load_index_from_storage(storage_context1, index_id="vector_index")

vector_retriever = index.as_retriever(similarity_top_k=10)

# 自定义中文BM25检索
def chinese_tokenizer(text: str):
    return list(jieba.cut(text,cut_all=True))

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
    similarity_top_k=5,
    num_queries=1,  # set this to 1 to disable query generation
    mode="dist_based_score",
    use_async=True,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(retriever)


# def match_and_correct(output_text, query_engine, reference_text):
#     splitter = SentenceSplitter()  # Create an instance

#     # Extract the text content from the output
#     document = Document(text=output_text)  # Create Document object
#     text_content = document.text  # Get the text content
    
#     # Split the text into chunks
#     chunks = splitter.split_text(text_content)  
#     corrected_chunks = []
    
#     # Loop through each chunk to retrieve relevant information
#     for chunk in chunks:
#         response = query_engine.query(QueryBundle(reference_text))
#         best_match = response.response if response.response else chunk
#         corrected_chunks.append(best_match)
    
#     print(corrected_chunks)

#     # Combine the corrected text with the original query
#     combined_input = f"{reference_text}\n{''.join(corrected_chunks)}"
#     return combined_input  # Return the combined input instead of just corrected text

# def chat_loop_with_query_engine():
#     print("欢迎使用自定义聊天引擎！输入'退出'以结束对话。")
#     while True:
#         user_input = input("\n你: ")
#         if user_input.lower() in ["退出", "exit", "quit"]:
#             print("聊天已结束，再见！")
#             break
#         try:
#             response = query_engine.query(user_input)
#             generated_text = response.response if response else "没有生成文本"
#             print("模型生成的文本：", generated_text)

            
#             # Use query engine to correct generated text
#             combined_input = match_and_correct(generated_text, query_engine, user_input)
#             print("矫正后的输入：", combined_input)
            
#             # Re-query the model with the combined input
#             new_response = query_engine.query(combined_input)
#             print("模型重新生成的文本：", new_response.response)

#         except Exception as e:
#             import traceback
#             print(f"发生错误: {e}")
#             traceback.print_exc()  # 输出完整的堆栈跟踪

# if __name__ == "__main__":
#     chat_loop_with_query_engine()

# 四、定义策略树和MCTS

class TreeNode:
    def __init__(self, chunks, parent=None, cost=0):
        self.chunks = chunks
        self.parent = parent
        self.children = []
        self.cost = cost
        self.visits = 0
        self.total_utility = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, candidate_chunks, budget, required_chunks=3):
        """扩展节点，生成包含3个不同块的组合"""
        added = False
        n = len(candidate_chunks)
        
        # 生成每个可能的3块组合
        for i in range(n - required_chunks + 1):
            combination = candidate_chunks[i:i + required_chunks]
            
            # 检查是否是3个不同的块
            if len(set(chunk[0] for chunk in combination)) == required_chunks:
                new_text = ' '.join([chunk[0] for chunk in combination])  # 组合后的文本
                new_cost = sum(chunk[1] for chunk in combination)  # 组合后的总成本

                # 确保组合在预算和块数限制内
                if self.cost + new_cost <= budget and len(self.chunks) + required_chunks <= required_chunks:
                    new_chunks = self.chunks + [new_text]
                    child_node = TreeNode(new_chunks, parent=self, cost=self.cost + new_cost)
                    self.add_child(child_node)
                    added = True

                    # 调试信息：输出每个组合的内容和成本
                    print(f"[DEBUG] 新生成的组合: {new_chunks}, 总成本: {new_cost}")
                    
        return added


class MCTS:
    def __init__(self, exploration_constant=1.4, llm=None, query=None, max_chunks=5, c=1.0, lambda_=1.0):
        self.exploration_constant = exploration_constant
        self.llm = llm
        self.query = query
        self.max_chunks = max_chunks
        self.c = c
        self.lambda_ = lambda_

    def search(self, root: TreeNode, candidate_chunks, budget, iterations=100, top_k=5):
        self.budget = budget
        root.expand(candidate_chunks, budget, required_chunks=3)

        for i in range(iterations):
            node = self.select(root)
            if len(node.chunks) < self.max_chunks and node.cost < budget:
                added = node.expand(candidate_chunks, budget, required_chunks=3)
                if added:
                    node = self.select(node)

            reward = self.simulate(node)
            self.backpropagate(node, reward)

        best_nodes = sorted(root.children, key=lambda child: child.total_utility / (child.visits + 1e-5), reverse=True)[:top_k]
        return best_nodes

    def select(self, node: TreeNode) -> TreeNode:
        while not node.is_leaf():
            node = max(
                node.children,
                key=lambda child: (child.total_utility / (child.visits + 1e-5)) +
                (self.exploration_constant * np.sqrt(np.log(child.visits + 2) / (child.visits + 1e-5)))
            )
        return node

    def simulate(self, node: TreeNode) -> float:
        context = "\n".join(node.chunks)
        prompt = f"【上下文】:\n{context}\n【客户问题】:\n{self.query}\n【专业回答】:\n"
        generated_response = self.llm.complete(prompt)
        response_text = generated_response if isinstance(generated_response, str) else generated_response.text
        V_vi = self.evaluate_response(response_text, context)
        N_vi = node.visits + 1
        U_vi = V_vi * N_vi + self.c * np.log(N_vi) - self.lambda_ * node.cost / self.budget
        return U_vi

    def evaluate_response(self, response: str, context: str) -> float:
        length_score = min(len(response) / 100, 1.0)
        similarity_score = self.calculate_similarity(response, context)
        
        # 增加多样性评分（基于块数量和差异性）
        diversity_score = len(set(context.split())) / len(context.split())  # 计算块的多样性比例
        
        total_score = 0.6 * similarity_score + 0.2 * length_score + 0.2 * diversity_score
        
        # 调试信息：输出评分的各个部分
        print(f"[DEBUG] 评分: 相似度: {similarity_score:.4f}, 长度: {length_score:.4f}, 多样性: {diversity_score:.4f}, 总分: {total_score:.4f}")
        
        return total_score

    def calculate_similarity(self, text1: str, text2: str) -> float:
        embeddings1 = extract_embeddings([text1])
        embeddings2 = extract_embeddings([text2])
        if embeddings1.size == 0 or embeddings2.size == 0:
            return 0.0
        similarity = cosine_similarity(embeddings1, embeddings2).flatten()[0]
        return similarity

    def backpropagate(self, node: TreeNode, reward: float):
        while node is not None:
            node.visits += 1
            node.total_utility += reward
            node = node.parent

# 提取嵌入的辅助函数
def extract_embeddings(sentences: List[str]):
    if not sentences:
        return []
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pd')
    outputs = model(**inputs)
    embeddings = outputs[0][:, 0, :].numpy()
    return embeddings

# 主函数
def select_best_chunks_with_mcts(retrieved_nodes: List[Document], query: str, budget: int = 8192, top_k: int = 5) -> List[List[str]]:
    root = TreeNode(chunks=[])
    mcts = MCTS(llm=Settings.llm, query=query, max_chunks=6, c=1.0, lambda_=1.0)
    candidate_chunks = [(node.text, len(node.text)) for node in retrieved_nodes if len(node.text) <= budget]

    if not candidate_chunks:
        print("没有找到符合预算的有效块。")
        return []

    best_nodes = mcts.search(root, candidate_chunks, budget, iterations=10, top_k=top_k)
    return [node.chunks for node in best_nodes]

# 更新主循环，选择3块组合
def chat_loop_with_query_engine():
    print("欢迎使用自定义聊天引擎！输入'退出'以结束对话。")
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("聊天已结束，再见！")
            break
        try:
            query_bundle = QueryBundle(query_str=user_input)
            nodes = retriever._retrieve(query_bundle)

            if not nodes:
                print("没有找到相关内容，请重新输入。")
                continue

            top_k = 3
            selected_chunk_combinations = select_best_chunks_with_mcts(nodes, query=user_input, budget=8192, top_k=top_k)
            
            best_response = None
            best_score = -float("inf")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_chunks = {executor.submit(generate_and_score, chunks, user_input): chunks for chunks in selected_chunk_combinations}

                for future in concurrent.futures.as_completed(future_to_chunks):
                    chunks = future_to_chunks[future]
                    try:
                        response_text, current_score = future.result()
                        print(f"\n模型生成的文本内容（组合 {chunks}）：{response_text}，评分：{current_score}")

                        if current_score > best_score:
                            best_score = current_score
                            best_response = response_text
                    except Exception as e:
                        print(f"生成过程出错：{e}")

            print("\n最佳回复内容：", best_response)

        except Exception as e:
            import traceback
            print(f"发生错误: {e}")
            traceback.print_exc()

def generate_and_score(chunks: List[str], query: str) -> (str, float):
    context = "\n".join(chunks)
    response = query_engine.query(f"{context}\n用户问题: {query}")
    response_text = response.text if hasattr(response, 'text') else str(response)
    current_score = evaluate_response(response_text, context)
    return response_text, current_score

def evaluate_response(response: str, context: str) -> float:
    length_score = min(len(response) / 100, 1.0)
    similarity_score = calculate_similarity(response, context)
    return 0.7 * similarity_score + 0.3 * length_score

def calculate_similarity(text1: str, text2: str) -> float:
    embeddings1 = extract_embeddings([text1])
    embeddings2 = extract_embeddings([text2])
    if embeddings1.size == 0 or embeddings2.size == 0:
        return 0.0
    similarity = cosine_similarity(embeddings1, embeddings2).flatten()[0]
    return similarity

if __name__ == "__main__":
    chat_loop_with_query_engine()