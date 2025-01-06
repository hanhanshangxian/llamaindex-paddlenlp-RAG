import json
from app import chat_engine

# 定义读取 JSON 文件的函数
def load_questions_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

# 定义将单个结果追加写入 JSON 文件的函数
def append_answer_to_json(data, output_file):
    with open(output_file, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")

# 读取问题数据
input_file = "/home/wwhh/data/train.json"  # 输入文件路径
output_file = "/home/wwhh/data/compare_data/output3.json"  # 输出文件路径
questions_data = load_questions_from_json(input_file)

# 遍历每个问题，使用 chat_engine 生成回答
for entry in questions_data:
    question = entry.get('src')
    correct_answer = entry.get('tgt')
    
    # 检查 question 和 correct_answer 是否存在
    if not question or not correct_answer:
        print(f"Skipping entry due to missing data: {entry}")
        continue
    
    try:
        # 使用 chat_engine 生成回复
        # 如果 chat_engine 不支持 chat 方法，可以改用 query 或其他方法
        response = chat_engine.chat(question)  # 确保方法名称正确
        generated_answer = response.response if hasattr(response, 'response') else response
        
        # 构造结果
        result_entry = {
            "src": question,
            "tgt": correct_answer,
            "generated_answer": generated_answer
        }
        
        # 将结果追加写入 JSON 文件
        append_answer_to_json(result_entry, output_file)
    except Exception as e:
        print(f"Error processing question: {question}. Error: {e}")
        continue
