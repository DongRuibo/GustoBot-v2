"""LangGraph 节点子包。

这个子包存放各条业务链路的节点实现，例如 General、Clarify、KB RAG、GraphRAG、Text2SQL、Image 和 File。
节点负责执行具体业务动作，并把结果转换成工作流后续节点可以消费的 raw_answer/raw_evidence。
"""
