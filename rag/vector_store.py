from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os
import re
from collections import OrderedDict


class RerankRetriever:
    def __init__(self, base_retriever, top_n: int, model: str | None = None):
        self.base_retriever = base_retriever
        self.top_n = top_n
        self.model = model

    def invoke(self, query: str) -> list[Document]:
        documents = self.base_retriever.invoke(query)
        if not documents:
            return []

        try:
            from langchain_community.document_compressors import DashScopeRerank
            compressor = DashScopeRerank(model=self.model, top_n=self.top_n)
            return list(compressor.compress_documents(documents, query))
        except Exception as e:
            logger.error(f"[rerank]重排失败，返回未重排结果：{str(e)}", exc_info=True)
            return documents[:self.top_n]


class RRFHybridRetriever:
    def __init__(
            self,
            vector_retriever,
            bm25_retriever,
            k: int,
            rrf_k: int = 60,
            vector_weight: float = 1.0,
            bm25_weight: float = 1.0,
    ):
        self.retrievers = [
            ("vector", vector_retriever, vector_weight),
            ("bm25", bm25_retriever, bm25_weight),
        ]
        self.k = k
        self.rrf_k = rrf_k

    @staticmethod
    def _document_key(document: Document) -> str:
        source = document.metadata.get("source", "")
        page = document.metadata.get("page", "")
        return f"{source}:{page}:{document.page_content}"

    def invoke(self, query: str) -> list[Document]:
        score_map = {}
        doc_map = OrderedDict()

        for _, retriever, weight in self.retrievers:
            documents = retriever.invoke(query)
            for rank, document in enumerate(documents, start=1):
                key = self._document_key(document)
                doc_map.setdefault(key, document)
                score_map[key] = score_map.get(key, 0.0) + weight / (self.rrf_k + rank)

        sorted_keys = sorted(score_map, key=score_map.get, reverse=True)
        return [doc_map[key] for key in sorted_keys[:self.k]]


class VectorStoreService:
    def __init__(self):
        self.backend = chroma_conf.get("vector_store_backend", "chroma")
        self.retrieval_mode = chroma_conf.get("retrieval_mode", "vector")
        self.vector_store = None

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def _init_vector_store(self):
        from model.factory import embed_model

        if self.backend == "chroma":
            return Chroma(
                collection_name=chroma_conf["collection_name"],
                embedding_function=embed_model,
                persist_directory=get_abs_path(chroma_conf["persist_directory"]),
            )

        if self.backend == "pgvector":
            try:
                from langchain_postgres import PGVector
            except ImportError as e:
                raise ImportError(
                    "使用 pgvector 向量检索需要安装依赖：langchain-postgres、psycopg[binary]"
                ) from e

            connection_env = chroma_conf.get("pgvector_connection_env", "PGVECTOR_CONNECTION_STRING")
            connection = os.getenv(connection_env)
            if not connection:
                raise ValueError(
                    f"已启用 pgvector，但环境变量 {connection_env} 未设置。"
                    "请配置 PostgreSQL 连接串，例如："
                    "postgresql+psycopg://user:password@localhost:5432/dbname"
                )

            return PGVector(
                embeddings=embed_model,
                collection_name=chroma_conf["collection_name"],
                connection=connection,
                use_jsonb=True,
            )

        raise ValueError(f"未知的向量库后端：{self.backend}，可选值：chroma、pgvector")

    def _get_vector_store(self):
        if self.vector_store is None:
            self.vector_store = self._init_vector_store()

        return self.vector_store

    def get_retriever(self):
        if self.retrieval_mode == "vector":
            return self._maybe_rerank(self.get_vector_retriever())

        if self.retrieval_mode == "bm25":
            return self._maybe_rerank(self.get_bm25_retriever())

        if self.retrieval_mode == "hybrid_rrf":
            retriever = RRFHybridRetriever(
                vector_retriever=self.get_vector_retriever(
                    k=chroma_conf.get("hybrid_vector_k", chroma_conf["k"])
                ),
                bm25_retriever=self.get_bm25_retriever(
                    k=chroma_conf.get("hybrid_bm25_k", chroma_conf.get("bm25_k", chroma_conf["k"]))
                ),
                k=chroma_conf["k"],
                rrf_k=chroma_conf.get("rrf_k", 60),
                vector_weight=chroma_conf.get("rrf_vector_weight", 1.0),
                bm25_weight=chroma_conf.get("rrf_bm25_weight", 1.0),
            )
            return self._maybe_rerank(retriever)

        raise ValueError(f"未知的检索模式：{self.retrieval_mode}，可选值：vector、bm25、hybrid_rrf")

    def _maybe_rerank(self, retriever):
        if not chroma_conf.get("reranker_enabled", False):
            return retriever

        return RerankRetriever(
            base_retriever=retriever,
            top_n=chroma_conf.get("reranker_top_n", chroma_conf["k"]),
            model=chroma_conf.get("reranker_model"),
        )

    def get_vector_retriever(self, k: int | None = None):
        return self._get_vector_store().as_retriever(search_kwargs={"k": k or chroma_conf["k"]})

    def get_bm25_retriever(self, k: int | None = None):
        try:
            from langchain_community.retrievers import BM25Retriever
        except ImportError as e:
            raise ImportError("使用 BM25 关键词检索需要安装依赖：rank-bm25") from e

        documents = self._load_split_documents()
        if not documents:
            raise RuntimeError("BM25 初始化失败：知识库文档为空")

        retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=self._bm25_tokenize,
        )
        retriever.k = k or chroma_conf.get("bm25_k", chroma_conf["k"])
        return retriever

    @staticmethod
    def _bm25_tokenize(text: str) -> list[str]:
        text = str(text).lower()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
        chinese_chars = [token for token in tokens if re.match(r"[\u4e00-\u9fff]", token)]
        chinese_bigrams = [
            chinese_chars[index] + chinese_chars[index + 1]
            for index in range(len(chinese_chars) - 1)
        ]
        return tokens + chinese_bigrams

    def _get_file_documents(self, read_path: str):
        if read_path.endswith("txt"):
            return txt_loader(read_path)

        if read_path.endswith("pdf"):
            return pdf_loader(read_path)

        return []

    def _allowed_files_path(self) -> tuple[str]:
        return listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

    def _load_split_documents(self) -> list[Document]:
        split_documents = []
        for path in self._allowed_files_path():
            try:
                documents = self._get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_documents.extend(self.spliter.split_documents(documents))
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)

        return split_documents

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def md5_store_key(md5_for_check: str):
            store_id = chroma_conf.get("persist_directory", self.backend)
            return f"{self.backend}:{chroma_conf['collection_name']}:{store_id}:{md5_for_check}"

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            key = md5_store_key(md5_for_check)
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == key:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_store_key(md5_for_check) + "\n")

        for path in self._allowed_files_path():
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = self._get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self._get_vector_store().add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)


