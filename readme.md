基于llamaindex构建的RAG应用程序<br>
该项目是一个基于机器学习的问答系统，使用本地模型和向量检索来提高答复的准确性。<br>
主要用于金融领域的客户支持，提供基于上下文的专业回答。<br>


一、文件使用说明：<br>
1、如需加载文档，请将文件放入以下存储路径：./data<br>
2、如需查看文档索引生成的结果，可查看生成的向量数据库存储路径：./index<br>
4、环境搭建：./models/requirements.txt<br>
5、基于外规场景下精调后的金融大模型：checkpoint<br>
6、嵌入式模型：bge-large-zh-1.5<br>

二、如需要使用基础RAG，请使用以下文件：<br>
1、向量检索：base_index.py<br>
2、向量+关键词检索：vector_keyword.py<br>
3、向量+BM25检索：vector_BM25.py<br>
4、向量+BM25+MTCS+动态文档检索：vector_BM25_mtcs_auto.py<br>


三、向量+BM25+MTCS+动态文档检索代码结构：<br>
一、模型加载和自定义LLM类<br>
本地模型加载：<br>
加载模型：通过paddlenlp的AutoTokenizer和AutoModelForCausalLM从指定路径加载本地模型和tokenizer。<br>
模型切换到评估模式：model.eval()确保模型处于评估模式，避免训练中的不必要操作。<br>
自定义LLM类 OurLLM：<br>
complete 方法：用于生成模型的回答。它首先定义了一个银行业务相关的prompt模板，并将用户问题和上下文插入模板中，生成格式化后的提示。然后通过tokenizer将文本转化为模型输入，利用model.generate()方法生成模型的输出文本，最后将生成的文本解码并返回。<br>
stream_complete 方法：此方法未实现，可能是为未来的流式生成设计的。<br>
设置自定义LLM：<br>
Settings.llm = OurLLM()：将自定义的LLM类应用到系统的设置中，替换默认的LLM。<br>

二、嵌入模型和文档索引<br>
加载嵌入模型：使用HuggingFaceEmbedding加载本地嵌入模型，并设置在Settings.embed_model中。<br>
文档加载与索引：<br>
文本加载：通过SimpleDirectoryReader从指定路径加载文本文件，构建文档对象。<br>
索引构建：使用VectorStoreIndex.from_documents创建基于文档的向量索引，并将其持久化到指定目录中。<br>
索引加载：从持久化存储中加载索引，并转换成检索器。<br>
BM25检索：自定义了SimpleBM25Retriever类，修改了BM25Retriever的tokenizer，使用jieba分词来支持中文检索。创建了bm25_retriever和vector_retriever两个检索器，并组合成一个QueryFusionRetriever，通过加权检索结果来增强查询效果。<br>

三、MCTS（蒙特卡洛树搜索）<br>
TreeNode 类：<br>
该类表示树中的节点，保存了当前节点的文本块、父节点、子节点以及访问次数、总效用等信息。<br>
expand方法用来生成新节点，将多个候选块组合生成新的子节点，并保证成本不超出预算。<br>
MCTS类：<br>
search 方法：这是MCTS的核心搜索函数，通过多次迭代，选择、扩展、模拟和反向传播来搜索最佳的节点组合。<br>
select 方法：选择一个节点进行扩展，使用UCB1公式（即基于效用和访问次数的公式）来平衡探索与开发。<br>
simulate 方法：模拟一个节点的响应并评估其效用，基于生成的文本和上下文计算响应的质量。<br>
backpropagate 方法：将模拟的效用反向传播到树的父节点，用于更新树的状态。<br>

四、聊天引擎的循环与优化<br>
聊天循环：<br>
chat_loop_with_query_engine函数实现了一个用户交互的循环，用户输入问题后，系统将通过检索器（retriever）检索相关文档。<br>
检索结果被传递给MCTS算法，选择出最佳的文本块组合并生成回复。<br>
生成的回复通过query_engine.query进行查询，并计算其相关性得分，选择最佳的回复。<br>
结果处理和评估：<br>
在生成回复之前，首先使用MCTS优化选出最相关的文本块组合，通过generate_and_score方法生成响应，并计算得分。<br>
得分较高的文本被认为是最佳回复，并最终输出给用户。<br>

五、辅助函数与相似度计算<br>
extract_embeddings：提取输入文本的嵌入向量。它使用model对输入文本进行编码，并返回句子的嵌入表示。<br>
相似度计算：<br>
calculate_similarity：通过余弦相似度计算两个文本的相似度。<br>
evaluate_response：综合文本的长度、相似度和多样性评分来评估生成的回答的质量。<br>
calculate_relevance_score：计算文档节点与用户查询的相关性得分，通常是通过嵌入向量的相似度计算。<br>

六、线程池与并发<br>
ThreadPoolExecutor：<br>
在chat_loop_with_query_engine中，使用ThreadPoolExecutor并发地生成和评分多个文本块组合，进一步提升系统的效率。<br>

注意事项:paddlenlp和paddlepaddle请从源代码进行安装<br>
相关链接：<br>
paddlenlp：https://github.com/PaddlePaddle/PaddleNLP<br>
paddlepaddle：https://www.paddlepaddle.org.cn/<br>
llamaindex:https://github.com/run-llama/llama_index<br>

提示：<br>
1、需修改llamaindex中的llama_index/llama-index-core/llama_index/core/chat_engine/condense_plus_context.py文件中的窗口大小<br>
        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window - 100
        )<br>
2、如需对模型进行精调，请参考https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm
