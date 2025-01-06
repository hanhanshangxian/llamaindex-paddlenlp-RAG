import json
from llama_index.core import QueryBundle
import test6
import concurrent.futures
import traceback
from typing import Optional, List, Mapping, Any
# 假设json文件为 input.json
json_filepath = "/home/wwh/data/data/问答对_v3/train.json"

# 加载问题和目标答案
with open(json_filepath, "r", encoding="utf-8") as f:
    qa_pairs = [json.loads(line) for line in f.readlines()]

# 新回答输出文件
output_filepath = "/home/wwh/data/data/compare_data1/output1(hou_vectorBM25_1024_200_64)13.json"

batch_size = 1
output_data_list = []



def process_qa_pair(qa_pair):
    question = qa_pair['src']
    original_answer = qa_pair['tgt']
    
    try:
        print(f"处理问题: {question}")

        # 调用检索器生成查询结果
        query_bundle = QueryBundle(query_str=question)
        nodes = test6.retriever._retrieve(query_bundle)
        
        # 检查检索器返回结果
        if not nodes:
            print(f"问题：{question} - 检索器未找到相关节点。")
            return {
                "question": question,
                "original_answer": original_answer,
                "new_answer": "没有找到相关内容"
            }

        print(f"检索到的节点数: {len(nodes)}")

        # 选择内容块
        top_k = 3
        selected_chunk_combinations = test6.select_best_chunks_with_mcts(nodes, query=question, budget=8192, top_k=top_k)
        
        if not selected_chunk_combinations:
            print(f"问题：{question} - 无法找到最佳内容块组合。")
            return {
                "question": question,
                "original_answer": original_answer,
                "new_answer": "最佳内容块选择失败"
            }
            
        best_response = None
        best_score = -float("inf")

        # 使用线程池并发生成回答
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_chunks = {executor.submit(generate_and_score, chunks, question): chunks for chunks in selected_chunk_combinations}

            for future in concurrent.futures.as_completed(future_to_chunks):
                chunks = future_to_chunks[future]
                try:
                    response_text, current_score = future.result()
                    print(f"\n生成的文本（组合 {chunks}）：{response_text}，评分：{current_score}")

                    if current_score > best_score:
                        best_score = current_score
                        best_response = response_text
                except Exception as e:
                    print(f"生成过程出错（组合 {chunks}）：{e}")
                    traceback.print_exc()

        # 检查是否生成了有效的回复
        if best_response:
            print(f"最佳回复：{best_response}")
        else:
            print(f"问题：{question} - 无法生成有效的回复。")
        
        return {
            "question": question,
            "original_answer": original_answer,
            "new_answer": best_response if best_response else "生成失败"
        }

    except Exception as e:
        print(f"处理问题 {question} 时发生错误: {e}")
        traceback.print_exc()
        return {
            "question": question,
            "original_answer": original_answer,
            "new_answer": "处理失败"
        }

def generate_and_score(chunks: List[str], query: str) -> (str, float):
    context = "\n".join(chunks)
    response = test6.query_engine.query(f"{context}\n用户问题: {query}")
    response_text = response.text if hasattr(response, 'text') else str(response)
    current_score = evaluate_response(response_text, context)
    return response_text, current_score

def evaluate_response(response: str, context: str) -> float:
    length_score = min(len(response) / 100, 1.0)
    similarity_score = calculate_similarity(response, context)
    return 0.7 * similarity_score + 0.3 * length_score

def calculate_similarity(text1: str, text2: str) -> float:
    embeddings1 = test6.extract_embeddings([text1])
    embeddings2 = test6.extract_embeddings([text2])
    if embeddings1.size == 0 or embeddings2.size == 0:
        return 0.0
    similarity = test6.cosine_similarity(embeddings1, embeddings2).flatten()[0]
    return similarity

# 写入处理结果到文件
with open(output_filepath, "w", encoding="utf-8") as f_out:
    for i, qa_pair in enumerate(qa_pairs, 1):
        output_data = process_qa_pair(qa_pair)
        output_data_list.append(output_data)

        # 每1条数据写入文件
        if i % batch_size == 0:
            for data in output_data_list:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            f_out.flush()  # 刷新缓冲区
            output_data_list = []  # 清空列表

    # 写入最后一批数据
    if output_data_list:
        for data in output_data_list:
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        f_out.flush()

print(f"新生成的回答已存入文件: {output_filepath}")
