from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.answer_review_agent import AnswerReviewAgent
from agent.tools.registry import get_agent_tools
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from memory.memory_service import LongTermMemoryService
from utils.logger_handler import logger


def message_content_to_text(content) -> str:
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


class NullMemoryService:
    def retrieve(self, query: str) -> str:
        return ""

    def save_turn(self, query: str, assistant_reply: str, history: list | None = None):
        return None


class ReactAgent:
    def __init__(self):
        try:
            self.memory = LongTermMemoryService()
        except Exception as e:
            logger.error(f"[ReactAgent]long-term memory disabled: {str(e)}", exc_info=True)
            self.memory = NullMemoryService()
        self.reviewer = AnswerReviewAgent()
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=get_agent_tools(),
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str, history: list | None = None):
        messages = list(history) if history else []
        original_history = list(messages)

        memory_context = self.memory.retrieve(query)
        if memory_context:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "以下是从长期记忆中检索到的用户信息，仅在与当前问题相关时使用；"
                        "不要向用户暴露“长期记忆”字样。\n"
                        f"{memory_context}"
                    ),
                },
            )

        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != query:
            messages.append({"role": "user", "content": query})

        input_dict = {"messages": messages}
        response_parts = []
        trace_messages = []
        final_reply = ""

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
                trace_messages = chunk.get("messages", trace_messages)
                latest_message = chunk["messages"][-1]
                is_ai_message = isinstance(latest_message, AIMessage) or getattr(latest_message, "type", None) == "ai"
                if not is_ai_message or getattr(latest_message, "tool_calls", None):
                    continue

                content = message_content_to_text(getattr(latest_message, "content", "")).strip()
                if not content:
                    continue
                if response_parts and response_parts[-1].strip() == content:
                    continue

                content = content + "\n"
                response_parts.append(content)

            draft_reply = "".join(response_parts).strip()
            if draft_reply:
                final_reply = self.reviewer.review(
                    query=query,
                    draft_answer=draft_reply,
                    history=original_history,
                    trace_messages=trace_messages,
                )
        except Exception as e:
            logger.error(f"[ReactAgent]模型调用失败：{str(e)}", exc_info=True)
            error_message = f"模型调用失败：{str(e)}"
            final_reply = error_message
        finally:
            final_reply = (final_reply or "").strip()
            if final_reply:
                yield final_reply + "\n"
                try:
                    self.memory.save_turn(query, final_reply, original_history)
                except Exception as e:
                    logger.error(f"[ReactAgent]长期记忆保存失败：{str(e)}", exc_info=True)


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)

