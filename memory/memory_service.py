import os
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model, embed_model
from utils.config_handler import memory_conf
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_memory_summarize_prompts
from utils.logger_handler import logger


class JsonMemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def similarity_search(self, query: str, k: int = 3, filter: dict | None = None) -> list[Document]:
        records = self._load_records()
        user_id = (filter or {}).get("user_id")
        if user_id:
            records = [record for record in records if record.get("metadata", {}).get("user_id") == user_id]

        query_tokens = self._tokenize(query)
        scored_records = []
        for record in records:
            content = str(record.get("page_content", ""))
            content_tokens = self._tokenize(content)
            score = len(query_tokens & content_tokens)
            if query and query in content:
                score += 5
            scored_records.append((score, record.get("metadata", {}).get("created_at", ""), record))

        scored_records.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_records = [item[2] for item in scored_records[:k]]
        return [
            Document(page_content=record.get("page_content", ""), metadata=record.get("metadata", {}))
            for record in selected_records
        ]

    def add_documents(self, documents: list[Document], ids: list[str] | None = None):
        ids = ids or []
        with open(self.path, "a", encoding="utf-8") as f:
            for index, document in enumerate(documents):
                record = {
                    "id": ids[index] if index < len(ids) else str(uuid4()),
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_records(self) -> list[dict]:
        if not self.path.exists():
            return []

        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"[long_term_memory] skipped broken json memory line in {self.path}")
        return records

    def _tokenize(self, text: str) -> set[str]:
        text = str(text or "").lower()
        tokens = set(re.findall(r"[a-z0-9_]+", text))
        tokens.update(char for char in text if "\u4e00" <= char <= "\u9fff")
        return tokens


class LongTermMemoryService:
    def __init__(self):
        self.enabled = bool(memory_conf.get("enabled", True))
        self.user_id = self._get_user_id()
        self.vector_store = None
        self.summary_chain = None

        if not self.enabled:
            return

        self.persist_directory = Path(get_abs_path(memory_conf["persist_directory"]))
        backend = str(memory_conf.get("backend", "auto")).lower()
        if backend == "json":
            self.vector_store = self._create_json_fallback_store()
        elif backend in {"auto", "chroma"}:
            self.vector_store = self._init_vector_store()
        else:
            raise ValueError(f"Unknown memory backend: {backend}. Expected auto, chroma, or json.")
        prompt = PromptTemplate.from_template(load_memory_summarize_prompts())
        self.summary_chain = prompt | chat_model | StrOutputParser()

    def _create_vector_store(self, persist_directory: Path):
        persist_directory.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=memory_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=str(persist_directory),
        )

    def _init_vector_store(self):
        try:
            return self._create_vector_store(self.persist_directory)
        except Exception as e:
            if not bool(memory_conf.get("auto_recover_on_error", True)):
                raise

            logger.error(
                f"[long_term_memory] vector store init failed, trying recovery: {str(e)}",
                exc_info=True,
            )
            recovered_directory = self._backup_broken_persist_directory(self.persist_directory)
            if recovered_directory != self.persist_directory:
                self.persist_directory = recovered_directory
            try:
                return self._create_vector_store(self.persist_directory)
            except Exception as recovery_error:
                logger.error(
                    f"[long_term_memory] recovered vector store init failed, using json fallback: {str(recovery_error)}",
                    exc_info=True,
                )
                return self._create_json_fallback_store()

    def _create_json_fallback_store(self):
        fallback_path = memory_conf.get("json_fallback_path", "data/long_term_memory.jsonl")
        fallback_store = JsonMemoryStore(get_abs_path(fallback_path))
        logger.warning(f"[long_term_memory] using json fallback memory store: {fallback_store.path}")
        return fallback_store

    def _backup_broken_persist_directory(self, persist_directory: Path) -> Path:
        if not persist_directory.exists():
            return persist_directory

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_prefix = memory_conf.get("backup_prefix", f"{persist_directory.name}_failed")
        backup_directory = persist_directory.with_name(f"{backup_prefix}_{timestamp}")

        try:
            shutil.move(str(persist_directory), str(backup_directory))
            logger.warning(
                f"[long_term_memory] broken vector store moved to {backup_directory}; "
                f"fresh store will be created at {persist_directory}"
            )
            return persist_directory
        except Exception as e:
            fallback_directory = persist_directory.with_name(f"{persist_directory.name}_recovered_{timestamp}")
            logger.error(
                f"[long_term_memory] failed to move broken store: {str(e)}; "
                f"using fallback store {fallback_directory}",
                exc_info=True,
            )
            return fallback_directory

    def _get_user_id(self) -> str:
        env_name = memory_conf.get("user_id_env", "MEMORY_USER_ID")
        return os.getenv(env_name) or memory_conf.get("default_user_id", "default_user")

    def retrieve(self, query: str) -> str:
        if not self.enabled:
            return ""

        try:
            documents = self.vector_store.similarity_search(
                query,
                k=memory_conf.get("top_k", 3),
                filter={"user_id": self.user_id},
            )
        except Exception as e:
            logger.error(f"[long_term_memory]记忆检索失败：{str(e)}", exc_info=True)
            return ""

        if not documents:
            return ""

        lines = []
        for index, document in enumerate(documents, start=1):
            lines.append(f"长期记忆{index}：{document.page_content}")

        return "\n".join(lines)

    def save_turn(self, query: str, assistant_reply: str, history: list | None = None):
        if not self.enabled or not query or not assistant_reply:
            return

        try:
            summary = self._summarize_turn(query, assistant_reply, history or [])
            if not summary or summary.strip() == "无":
                return

            now = datetime.now().isoformat(timespec="seconds")
            document = Document(
                page_content=summary.strip(),
                metadata={
                    "user_id": self.user_id,
                    "created_at": now,
                    "source": "conversation_summary",
                },
            )
            self.vector_store.add_documents([document], ids=[str(uuid4())])
            logger.info(f"[long_term_memory]已写入长期记忆 user_id={self.user_id}")
        except Exception as e:
            logger.error(f"[long_term_memory]记忆写入失败：{str(e)}", exc_info=True)

    def _summarize_turn(self, query: str, assistant_reply: str, history: list) -> str:
        recent_history = history[-memory_conf.get("max_history_messages", 8):]
        history_text = "\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in recent_history
        )

        return self.summary_chain.invoke(
            {
                "history": history_text,
                "user_input": query,
                "assistant_reply": assistant_reply,
                "summary_max_chars": memory_conf.get("summary_max_chars", 500),
            }
        )
