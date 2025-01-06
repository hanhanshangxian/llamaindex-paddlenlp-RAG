import json
from test3 import query_engine
from test3 import match_and_correct
from test3 import retriever
from test3 import Settings
from llama_index.core import QueryBundle
  # 假设你的主代码文件名为 main_script.py

# 假设json文件为 input.json
json_filepath = "/home/wwhh/data/问答对_v3/train.json"

# 加载问题和目标答案
with open(json_filepath, "r", encoding="utf-8") as f:
    qa_pairs = [json.loads(line) for line in f.readlines()]

# 新回答输出文件
output_filepath = "/home/wwhh/data/compare_data1/output1(hou_vectorBM25_1024_200_64)8.json"

batch_size = 1
output_data_list = []

with open(output_filepath, "w", encoding="utf-8") as f_out:
    for i, qa_pair in enumerate(qa_pairs, 1):
        question = qa_pair['src']
        original_answer = qa_pair['tgt']

        # 生成新回答
        query_bundle = QueryBundle(query_str=question)
        nodes = retriever._retrieve(query_bundle)
        context = nodes[0].node.text
        response = Settings.llm.complete(question, context_str=context)
        
        combined_input = match_and_correct(response, query_engine, question)

        new_answer = query_engine.query(combined_input).response
        if hasattr(new_answer, 'text'):  # 同样提取可序列化内容
            new_answer = new_answer.text
        
        output_data = {
            "question": question,
            "original_answer": original_answer,
            "new_answer": new_answer
        }
        
        # 暂时存储到列表
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