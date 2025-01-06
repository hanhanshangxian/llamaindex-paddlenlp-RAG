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
from datasketch import MinHash, MinHashLSH

paddle.set_device('gpu:6')

# 一、加载本地模型
model_name = "/home/wwhh/RAGtest2/RAGtest/models/model"
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
    num_output: int = 1096
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
        # "你是一位专业的银行业务人员，根据自身已有知识，并参考以下上下文来进行回答,并进行详细解释。\n"
        # "务必直接回答问题。\n"
        # "【上下文】:\n"
        # "{context_str}\n"
        # "【客户问题】:\n"
        # "{query_str}\n"
        # "【专业回答】:\n"
        # )

        # prompt_template = (
        #     "作为专业的银行业务人员，请基于以下上下文内容和自身的专业知识，为客户提供准确、详尽的解答。"
        #     "回答中请明确列出所有关键点和步骤，并通过具体的示例帮助客户理解。避免任何概括或简化描述。"
        #     "如果问题涉及到金融法规或政策，请准确引用相关的规定，标明具体名称及内容。\n"
        #     "如果新查询的内容无差异，请输出原回答。\n"
        #     "【上下文】:\n"
        #     "{context_str}\n"
        #     "【客户问题】:\n"
        #     "{query_str}\n"
        #     "【专业回答】:\n"
        # )

        prompt_template = (
            "根据以下上下文回答问题。\n"
            "上下文信息：\n"
            "{context_str}\n"
            "问题：{query_str}\n"
            "回答：\n"
        )
        
        # 生成提示文本
        qa_template = PromptTemplate(prompt_template)
        context_str = kwargs.get("context_str", "")
        query_str = kwargs.get("query_str", prompt)
        formatted_prompt = prompt_template.format(context_str=context_str, query_str=query_str)
        print(f"格式化后的提示: {formatted_prompt}") # 调试
        # print(f"上下文: {context_str}")  # 调试
        # print(f"问题: {query_str}")  # 调试
        
        inputs = tokenizer(formatted_prompt, return_tensors="pd", truncation=True, max_length=8192, batch_size=1)
        # print(f"传递给模型的输入: {inputs}") # 调试
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, top_k=50, top_p=0.9)
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
    model_name="/home/wwhh/RAGtest2/RAGtest/models/bge-large-zh-v1.5"
)



# splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
# filepaths = [os.path.join(dirpath, name) for dirpath, dirnames, filenames in os.walk("/home/wwhh/RAGtest2/RAGtest/txt") for name in filenames]

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

vector_retriever = index.as_retriever(similarity_top_k=10)

# 自定义中文BM25检索
def chinese_tokenizer(text: str):
    return list(jieba.cut(text,cut_all=True))

class SimpleBM25Retriever(BM25Retriever):
    @classmethod
    def from_defaults(cls, index, similarity_top_k, **kwargs):
        nodes = list(index.docstore.docs.values())
        return BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=5, verbose=True,
            tokenizer=chinese_tokenizer, **kwargs
        )

bm25_retriever = SimpleBM25Retriever.from_defaults(
    index=index,
    similarity_top_k=5
)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    retriever_weights=[0.6, 0.4],
    similarity_top_k=5,
    num_queries=4,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(retriever)


# 设置 MinHash 配置
num_perm = 256
similarity_threshold = 0.9

# 7. MinHash 处理与匹配函数
def generate_minhash(text, num_perm):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)

# 加载评估集
eval_set_path = "/home/wwhh/data/问答对_v3/train.json"
with open(eval_set_path, "r", encoding="utf-8") as f:
    eval_data = [json.loads(line.strip()) for line in f]

minhash_dict = {}
for i, entry in enumerate(eval_data):
    question = entry['src']
    minhash = generate_minhash(question, num_perm)
    lsh.insert(f"eval_{i}", minhash)
    minhash_dict[f"eval_{i}"] = minhash




def remove_punctuation(text):
    """去除文本中的标点符号"""
    return re.sub(r"[^\w\s]", "", text)

def extract_embeddings(sentences):
    """提取每个句子的嵌入向量"""
    if not sentences:
        return []

    with torch.no_grad():
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pd')  
        outputs = model(**inputs)
        embeddings = outputs[0][:, 0, :].numpy()  # 取 [CLS] token 对应的向量
    return embeddings

def rewrite_sentence(sentence):
    """对句子进行重写"""
    inputs = tokenizer(sentence, return_tensors='pd')  
    outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
    token_ids = outputs[0].numpy().tolist()[0]
    rewritten_sentence = tokenizer.decode(token_ids, skip_special_tokens=True)
    return rewritten_sentence

def generate_new_sentences(text, n_sentences=3):
    """对模型生成的回答进行语义提取和重写，生成最相关的句子"""
    text = remove_punctuation(text)

    if not text:
        raise ValueError("处理的文本不能为空")

    sentences = list(jieba.cut(text, cut_all=False))

    if len(sentences) < n_sentences:
        n_sentences = len(sentences)  # 如果句子少于要求的数量，调整为实际数量

    embeddings = extract_embeddings(sentences)
    if embeddings.size == 0:
        print("嵌入向量为空，无法进行处理。")
        return []

    # 计算文本的嵌入
    text_embedding = extract_embeddings([text])[0].reshape(1, -1)

    # 计算每个句子与输入文本之间的余弦相似度
    similarities = cosine_similarity(text_embedding, embeddings).flatten()

    # 获取最相关的句子索引
    top_indices = np.argsort(similarities)[-n_sentences:]

    # 根据索引提取最相关的句子并进行重写
    new_sentences = []
    for idx in top_indices:
        rewritten_sentence = rewrite_sentence(sentences[idx])
        new_sentences.append(rewritten_sentence)

    return new_sentences

def match_and_correct(output_text, query_engine, reference_text):
    splitter = SentenceSplitter()

    text_content = output_text.text if hasattr(output_text, 'text') else str(output_text)

    chunks = splitter.split_text(text_content)
    corrected_chunks = []

    for chunk in chunks:
        query_bundle = QueryBundle(query_str=chunk)
        response = query_engine.query(query_bundle)
        best_match = response.response if response.response else chunk
        corrected_chunks.append(best_match)

    combined_input = f"{reference_text}\n{''.join(corrected_chunks)}"
    return combined_input

def extract_embeddings(sentences):
    """提取每个句子的嵌入向量"""
    if not sentences:
        return []

    with torch.no_grad():
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pd')  
        outputs = model(**inputs)
        embeddings = outputs[0][:, 0, :].numpy()  # 取 [CLS] token 对应的向量
    return embeddings

def calculate_similarity(answer_embedding, minhash_embeddings):
    """计算回答与评估集回答之间的余弦相似度"""
    similarities = cosine_similarity(answer_embedding.reshape(1, -1), minhash_embeddings)
    return similarities.flatten()

def chat_loop_with_query_engine():
    print("欢迎使用自定义聊天引擎！输入'退出'以结束对话。")
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("聊天已结束，再见！")
            break
        try:
            # 预处理用户输入
            input_question = remove_punctuation(user_input)

            # 生成 MinHash 并查找相似问题
            input_minhash = generate_minhash(input_question, num_perm)
            similar_questions = lsh.query(input_minhash)

            # 提取评估集中的答案嵌入
            eval_answers = [entry['tgt'] for entry in eval_data]
            eval_embeddings = extract_embeddings(eval_answers)

            # 检查是否有相似问题
            if similar_questions:
                standard_question = similar_questions[0]  # 取第一个相似的问题
                print(f"匹配到的标准问题: {standard_question}")
                response = query_engine.query(standard_question)
            else:
                print("未找到相似度大于90%的问题，基于原问题生成回复...")
                response = query_engine.query(user_input)

            response_text = response.text if hasattr(response, 'text') else str(response)
            print(f"模型生成的文本内容: {response_text}")

            # 检查生成的内容是否存在，并计算余弦相似度
            if response_text:
                answer_embedding = extract_embeddings([response_text])[0]
                similarities = calculate_similarity(answer_embedding, eval_embeddings)

                # 判断相似度
                if np.max(similarities) > 0.95:
                    print("相似度高于95%，输出该回复。")
                else:
                    print("相似度低于95%，重新生成回答...")
                    response = query_engine.query(user_input)
                    response_text = response.text if hasattr(response, 'text') else str(response)
                    print(f"重新生成的文本内容: {response_text}")
            else:
                print("模型生成的文本为空，跳过分句生成。")
                continue

        except Exception as e:
            import traceback
            print(f"发生错误: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    chat_loop_with_query_engine()

