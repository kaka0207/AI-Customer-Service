import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_answer_review_prompts


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def history_to_text(history: list | None, max_messages: int = 8) -> str:
    if not history:
        return "无"

    lines = []
    for message in history[-max_messages:]:
        role = message.get("role", "unknown") if isinstance(message, dict) else getattr(message, "type", "unknown")
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        text = _message_content_to_text(content).strip()
        if text:
            lines.append(f"{role}: {text}")

    return "\n".join(lines) or "无"


def trace_messages_to_text(messages: list | None, max_chars: int = 6000) -> str:
    if not messages:
        return "无"

    lines = []
    for message in messages:
        role = getattr(message, "type", None)
        if role is None and isinstance(message, dict):
            role = message.get("role", "unknown")
        role = role or "unknown"

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content", "")
        text = _message_content_to_text(content).strip()

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            text = f"{text}\n工具调用: {tool_calls}".strip()

        if text:
            lines.append(f"{role}: {text}")

    trace_text = "\n".join(lines)
    if len(trace_text) > max_chars:
        return trace_text[-max_chars:]
    return trace_text or "无"


def parse_review_response(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty review response")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class AnswerReviewAgent:
    def __init__(self):
        review_conf = agent_conf.get("answer_review", {}) or {}
        self.enabled = bool(review_conf.get("enabled", True))
        self.max_history_messages = int(review_conf.get("max_history_messages", 8))
        self.max_trace_chars = int(review_conf.get("max_trace_chars", 6000))
        prompt = PromptTemplate.from_template(load_answer_review_prompts())
        self.chain = prompt | chat_model | StrOutputParser()

    def review(
        self,
        query: str,
        draft_answer: str,
        history: list | None = None,
        trace_messages: list | None = None,
    ) -> str:
        draft_answer = (draft_answer or "").strip()
        if not self.enabled or not draft_answer:
            return draft_answer

        try:
            raw_response = self.chain.invoke(
                {
                    "query": query,
                    "history": history_to_text(history, self.max_history_messages),
                    "draft_answer": draft_answer,
                    "trace": trace_messages_to_text(trace_messages, self.max_trace_chars),
                }
            )
            review_result = parse_review_response(raw_response)
            final_answer = str(review_result.get("final_answer") or "").strip()
            issues = review_result.get("issues") or []
            approved = review_result.get("approved")
            logger.info(f"[AnswerReviewAgent]审查完成 approved={approved} issues={issues}")
            return final_answer or draft_answer
        except Exception as e:
            logger.error(f"[AnswerReviewAgent]审查失败，回退主Agent草稿：{str(e)}", exc_info=True)
            return draft_answer
