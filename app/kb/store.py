"""知识库存储模块。

这个文件定义 KnowledgeStore 抽象，并实现内存存储和 PostgreSQL + pgvector 存储适配器。
它让上层 KB 服务不关心底层是本地内存还是真实数据库，从而便于测试和后续生产部署切换。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol

from app.kb.embeddings import cosine_similarity
from app.kb.hybrid import (
    annotate_vector_results,
    bm25_scores,
    build_search_text,
    postgres_tsquery,
    reciprocal_rank_fusion,
)


@dataclass(slots=True)
class StoredChunk:
    # StoredChunk 表示已经完成 embedding、可以被向量检索的 chunk。
    document_id: str
    chunk_id: str
    content: str
    embedding: list[float]
    search_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    # RetrievedChunk 是召回后的候选证据；score 会先来自 pgvector/内存相似度，再被 reranker 更新。
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_score(self, score: float, *, rerank_score: float) -> "RetrievedChunk":
        metadata = {**self.metadata, "rerank_score": rerank_score}
        metadata.setdefault("pre_rerank_score", self.score)
        if "vector_score" not in metadata and "lexical_score" not in metadata:
            metadata["vector_score"] = self.score
        return RetrievedChunk(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            content=self.content,
            score=score,
            metadata=metadata,
        )


class KnowledgeStore(Protocol):
    # KnowledgeStore 屏蔽底层存储差异，工作流只关心“入库”和“向量检索”两个动作。
    store_type: str

    def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        ...

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
        query_text: str | None = None,
        hybrid_enabled: bool = False,
        lexical_top_k: int = 8,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        ...

    def count_chunks(self) -> int:
        ...


class InMemoryKnowledgeStore:
    # 内存存储用于本地开发和测试：它保留完整 KB RAG 行为，但不会要求先安装 PostgreSQL/pgvector。
    # 生产环境应配置 GUSTOBOT_POSTGRES_DSN，让服务切换到 PostgreSQLPgVectorStore。
    store_type = "memory"

    def __init__(self) -> None:
        self._chunks: dict[str, StoredChunk] = {}
        self._lock = RLock()

    def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
        query_text: str | None = None,
        hybrid_enabled: bool = False,
        lexical_top_k: int = 8,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        with self._lock:
            candidates = list(self._chunks.values())

        filtered = [
            chunk for chunk in candidates if _metadata_matches(chunk.metadata, metadata_filter)
        ]
        vector_results = [
            RetrievedChunk(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=max(0.0, cosine_similarity(query_embedding, chunk.embedding)),
                metadata=chunk.metadata,
            )
            for chunk in filtered
        ]
        vector_results = sorted(vector_results, key=lambda item: item.score, reverse=True)[:top_k]
        if not hybrid_enabled or not query_text:
            return annotate_vector_results(vector_results)

        search_documents = {
            chunk.chunk_id: chunk.search_text or build_search_text(chunk.content, chunk.metadata)
            for chunk in filtered
        }
        lexical_scores = bm25_scores(query_text, search_documents)
        if not lexical_scores:
            return annotate_vector_results(vector_results)

        chunks_by_id = {chunk.chunk_id: chunk for chunk in filtered}
        lexical_results = [
            RetrievedChunk(
                document_id=chunks_by_id[chunk_id].document_id,
                chunk_id=chunk_id,
                content=chunks_by_id[chunk_id].content,
                score=score,
                metadata=chunks_by_id[chunk_id].metadata,
            )
            for chunk_id, score in sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)[
                :lexical_top_k
            ]
        ]
        return reciprocal_rank_fusion(
            vector_results,
            lexical_results,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    def count_chunks(self) -> int:
        # 诊断接口使用，帮助确认本地内存知识库当前有多少可检索 chunk。
        with self._lock:
            return len(self._chunks)


class PostgreSQLPgVectorStore:
    # PostgreSQL + pgvector 是第二阶段的目标存储；驱动采用惰性导入，避免未配置数据库时影响本地测试。
    # 表结构保持简单：documents 存文档级元数据，chunks 存文本、metadata 和 vector embedding。
    store_type = "postgres_pgvector"

    def __init__(self, dsn: str, *, embedding_dimension: int, table_name: str = "kb_chunks") -> None:
        self.dsn = dsn
        self.embedding_dimension = embedding_dimension
        self.table_name = _validate_table_name(table_name)
        self._ensure_driver()
        self.ensure_schema()

    def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        if not chunks:
            return

        psycopg = self._ensure_driver()
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                for chunk in chunks:
                    if self.table_name == "searchable_documents":
                        self._upsert_legacy_searchable_document(cursor, chunk)
                    else:
                        self._upsert_v2_chunk(cursor, chunk)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
        query_text: str | None = None,
        hybrid_enabled: bool = False,
        lexical_top_k: int = 8,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        vector_results = self._vector_search(query_embedding, top_k=top_k, metadata_filter=metadata_filter)
        if not hybrid_enabled or not query_text:
            return annotate_vector_results(vector_results)

        lexical_results = self._lexical_search(query_text, top_k=lexical_top_k, metadata_filter=metadata_filter)
        if not lexical_results:
            return annotate_vector_results(vector_results)
        return reciprocal_rank_fusion(vector_results, lexical_results, top_k=top_k, rrf_k=rrf_k)

    def _vector_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        psycopg = self._ensure_driver()
        where_sql = ""
        params: list[Any] = [_vector_literal(query_embedding)]
        if metadata_filter:
            # metadata_filter 目前按 jsonb 包含关系过滤，后续可扩展 company_id、user_id、doc_type 等权限条件。
            where_sql = "WHERE metadata @> %s::jsonb"
            params.append(json.dumps(metadata_filter, ensure_ascii=False))
        params.append(top_k)

        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                if self.table_name == "searchable_documents":
                    cursor.execute(
                        f"""
                        SELECT document_id,
                               id::text AS chunk_id,
                               content,
                               GREATEST(0, 1 - (embedding <=> %s::vector)) AS score,
                               metadata
                        FROM searchable_documents
                        {where_sql}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        _duplicate_query_vector_param(params, metadata_filter),
                    )
                else:
                    cursor.execute(
                        f"""
                        SELECT document_id,
                               chunk_id,
                               content,
                               GREATEST(0, 1 - (embedding <=> %s::vector)) AS score,
                               metadata
                        FROM kb_chunks
                        {where_sql}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        _duplicate_query_vector_param(params, metadata_filter),
                    )
                rows = cursor.fetchall()

        return [
            RetrievedChunk(
                document_id=row[0],
                chunk_id=row[1],
                content=row[2],
                score=float(row[3]),
                metadata=dict(row[4] or {}),
            )
            for row in rows
        ]

    def _lexical_search(
        self,
        query_text: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        tsquery = postgres_tsquery(query_text)
        if tsquery is None:
            return []

        psycopg = self._ensure_driver()
        where_clauses = []
        params: list[Any] = [tsquery]
        if metadata_filter:
            where_clauses.append("metadata @> %s::jsonb")
            params.append(json.dumps(metadata_filter, ensure_ascii=False))

        if self.table_name == "searchable_documents":
            search_expr = "to_tsvector('simple', content)"
            table_sql = "searchable_documents"
            id_sql = "id::text"
        else:
            search_expr = "to_tsvector('simple', COALESCE(NULLIF(search_text, ''), content))"
            table_sql = "kb_chunks"
            id_sql = "chunk_id"
        where_clauses.append(f"{search_expr} @@ q.query")
        where_sql = "WHERE " + " AND ".join(where_clauses)
        params.append(top_k)

        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH q AS (SELECT to_tsquery('simple', %s) AS query)
                    SELECT document_id,
                           {id_sql} AS chunk_id,
                           content,
                           ts_rank_cd({search_expr}, q.query) AS score,
                           metadata
                    FROM {table_sql}, q
                    {where_sql}
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cursor.fetchall()

        return [
            RetrievedChunk(
                document_id=row[0],
                chunk_id=row[1],
                content=row[2],
                score=float(row[3]),
                metadata=dict(row[4] or {}),
            )
            for row in rows
        ]

    def count_chunks(self) -> int:
        # 诊断接口使用，真实 pgvector 接入后可用它确认文档是否已经成功入库。
        psycopg = self._ensure_driver()
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                row = cursor.fetchone()
        return int(row[0]) if row else 0

    def ensure_schema(self) -> None:
        psycopg = self._ensure_driver()
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                # CREATE EXTENSION 可能需要数据库权限；若失败，应由部署阶段提前安装 pgvector。
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                if self.table_name == "searchable_documents":
                    self._ensure_legacy_schema(cursor)
                    return

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_documents (
                        document_id text PRIMARY KEY,
                        title text NOT NULL,
                        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS kb_chunks (
                        chunk_id text PRIMARY KEY,
                        document_id text NOT NULL REFERENCES kb_documents(document_id) ON DELETE CASCADE,
                        content text NOT NULL,
                        embedding vector({self.embedding_dimension}) NOT NULL,
                        search_text text NOT NULL DEFAULT '',
                        metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute("ALTER TABLE kb_chunks ADD COLUMN IF NOT EXISTS search_text text NOT NULL DEFAULT ''")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kb_chunks_metadata ON kb_chunks USING gin (metadata)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_kb_chunks_search_text ON kb_chunks USING gin (to_tsvector('simple', COALESCE(NULLIF(search_text, ''), content)))"
                )

    def _upsert_v2_chunk(self, cursor: Any, chunk: StoredChunk) -> None:
        cursor.execute(
            """
            INSERT INTO kb_documents (document_id, title, metadata)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (document_id) DO UPDATE
            SET title = EXCLUDED.title,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                chunk.document_id,
                chunk.metadata.get("title", chunk.document_id),
                json.dumps(chunk.metadata, ensure_ascii=False),
            ),
        )
        cursor.execute(
            """
            INSERT INTO kb_chunks (chunk_id, document_id, content, embedding, search_text, metadata)
            VALUES (%s, %s, %s, %s::vector, %s, %s::jsonb)
            ON CONFLICT (chunk_id) DO UPDATE
            SET content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                search_text = EXCLUDED.search_text,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.content,
                _vector_literal(chunk.embedding),
                chunk.search_text or build_search_text(chunk.content, chunk.metadata),
                json.dumps(chunk.metadata, ensure_ascii=False),
            ),
        )

    def _upsert_legacy_searchable_document(self, cursor: Any, chunk: StoredChunk) -> None:
        # 旧 GustoBot-develop 的 pgvector 表没有 chunk_id 唯一键，因此这里使用 metadata 中的 chunk_id 做幂等删除再插入。
        metadata = {**chunk.metadata, "chunk_id": chunk.chunk_id}
        cursor.execute(
            "DELETE FROM searchable_documents WHERE metadata ->> 'chunk_id' = %s",
            (chunk.chunk_id,),
        )
        cursor.execute(
            """
            INSERT INTO searchable_documents (document_id, source, content, embedding, metadata)
            VALUES (%s, %s, %s, %s::vector, %s::jsonb)
            """,
            (
                chunk.document_id,
                metadata.get("source_id", metadata.get("title", chunk.document_id)),
                chunk.content,
                _vector_literal(chunk.embedding),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )

    def _ensure_legacy_schema(self, cursor: Any) -> None:
        # 兼容旧项目 docker/pgvector/init.sql 创建的 searchable_documents 表。
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS searchable_documents (
                id SERIAL PRIMARY KEY,
                document_id VARCHAR(255),
                source VARCHAR(255),
                content TEXT,
                embedding vector({self.embedding_dimension}),
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_searchable_documents_source ON searchable_documents(source)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_searchable_documents_embedding ON searchable_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_searchable_documents_content_fts ON searchable_documents USING gin (to_tsvector('simple', content))"
        )

    @staticmethod
    def _ensure_driver():
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "已配置 GUSTOBOT_POSTGRES_DSN，但当前环境缺少 psycopg。"
                "请先安装 psycopg[binary]，或取消该环境变量改用内存存储。"
            ) from exc
        return psycopg


def _metadata_matches(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    return all(metadata.get(key) == value for key, value in metadata_filter.items())


def _vector_literal(vector: list[float]) -> str:
    # pgvector 支持 '[0.1,0.2]' 形式的文本字面量；这里集中格式化，避免 SQL 拼接散落各处。
    return "[" + ",".join(f"{item:.8f}" for item in vector) + "]"


def _validate_table_name(table_name: str) -> str:
    # 表名来自环境变量，必须限制为普通标识符，避免把表名插入 SQL 时产生注入风险。
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(f"非法 pgvector 表名：{table_name}")
    if table_name not in {"kb_chunks", "searchable_documents"}:
        raise ValueError("当前仅支持 kb_chunks 或 searchable_documents 两种 KB pgvector 表结构。")
    return table_name


def _duplicate_query_vector_param(params: list[Any], metadata_filter: dict[str, Any] | None) -> list[Any]:
    # 查询 SQL 中 query vector 出现两次：一次计算 score，一次排序；metadata_filter 位于二者中间。
    if metadata_filter:
        return [params[0], params[1], params[0], params[2]]
    return [params[0], params[0], params[1]]
