"""KB RAG 子包。

这个子包实现第二阶段的知识库问答能力，负责文档切块、embedding、向量存储、
pgvector/内存检索、reranker 精排，以及把检索结果转换为统一 Evidence。
"""
