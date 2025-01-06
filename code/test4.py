import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

# 一、加载本地模型
model_name = "/home/wwh/PaddleNLP/llm/checkpoints/llama_sft_ckpts2/checkpoint-348"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer is None:
    raise ValueError("Tokenizer加载失败，请检查路径是否正确。")
model = AutoModelForCausalLM.from_pretrained(model_name)
if model is None:
    raise ValueError("模型加载失败，请检查路径是否正确。")
model.eval() 

def test_paddle_model(question: str):
    # 构建输入提示
    prompt = f"问题：{question}\n回答：\n"
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pd", truncation=True, max_length=8192)

    # 生成输出
    with paddle.no_grad():  # 不需要计算梯度
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.8, top_k=50, top_p=0.85)
    
    # 解码输出
    token_ids = outputs[0].numpy().flatten()
    response = tokenizer.decode(token_ids, skip_special_tokens=True).replace("</s>", "").strip()
    
    return response

# 测试问题
question = "银行信贷资产收益权转让的基本规范有哪些？"

# 测试模型输出
output = test_paddle_model(question)
print("模型输出:", output)
