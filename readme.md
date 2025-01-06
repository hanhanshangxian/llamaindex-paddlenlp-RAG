基于llamaindex构建的RAG应用程序
一、文件使用说明：
1、如需加载文档，请将文件放入以下存储路径：./data
2、如需查看文档索引生成的结果，可查看生成的向量数据库存储路径：./index
4、环境搭建：./models/requirements.txt
5、基于外规场景下精调后的金融大模型：checkpoint
6、嵌入式模型：bge-large-zh-1.5

二、如需要使用基础RAG，请使用以下文件：
1、向量检索：base_index.py
2、向量+关键词检索：vector_keyword.py
3、向量+BM25检索：vector_BM25.py
4、向量+BM25+MTCS+动态文档检索：vector_BM25_mtcs_auto.py




提示:paddlenlp和paddlepaddle请从源代码进行安装
相关链接：
paddlenlp：https://github.com/PaddlePaddle/PaddleNLP
paddlepaddle：https://www.paddlepaddle.org.cn/
llamaindex:https://github.com/run-llama/llama_index
