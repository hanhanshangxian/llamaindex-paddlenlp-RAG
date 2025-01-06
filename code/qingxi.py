def remove_newlines(input_file, output_file):
    # 读取文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 去除换行符和多余的空格
    content_cleaned = content.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    content_cleaned = ' '.join(content_cleaned.split())  # 去除多余的空格
    
    # 将去除换行后的内容写入新的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content_cleaned)

# 使用示例
input_file = ''  # 输入文件路径
output_file = ''  # 输出文件路径
remove_newlines(input_file, output_file)
