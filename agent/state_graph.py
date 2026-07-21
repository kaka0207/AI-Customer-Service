"""
Optional LangGraph StateGraph orchestration for the customer-service pipeline.

This module is intentionally not imported by app.py or ReactAgent. It documents
and exposes a hand-written graph that can be wired in later without changing the
current runtime path.
"""

from typing import Any, Callable, NotRequired, TypedDict

from langgraph.graph import END, StateGraph


class CustomerServiceState(TypedDict):
    query: str
    history: list[dict]
    memory_context: NotRequired[str]
    draft_answer: NotRequired[str]
    final_answer: NotRequired[str]
    trace_messages: NotRequired[list[Any]]
    review_passed: NotRequired[bool]
    error: NotRequired[str]


RetrieveMemoryFn = Callable[[str], str]
GenerateDraftFn = Callable[[str, list[dict], str], tuple[str, list[Any]]]
ReviewAnswerFn = Callable[[str, str, list[dict], list[Any]], str]
SaveMemoryFn = Callable[[str, str, list[dict]], None]


def build_customer_service_state_graph(
    retrieve_memory: RetrieveMemoryFn,
    generate_draft: GenerateDraftFn,
    review_answer: ReviewAnswerFn,
    save_memory: SaveMemoryFn,
):
    """Build a future-use graph for explicit multi-agent orchestration.

    Expected flow:
    load_memory -> main_agent_generate -> answer_review -> save_memory -> END

    The current application does not call this function. The callables are
    injected so importing this module has no model, vector-store, or network
    side effects.
    """

    graph = StateGraph(CustomerServiceState)

    def load_memory_node(state: CustomerServiceState) -> dict:
        return {"memory_context": retrieve_memory(state["query"])}

    def main_agent_generate_node(state: CustomerServiceState) -> dict:
        draft_answer, trace_messages = generate_draft(
            state["query"],
            state.get("history", []),
            state.get("memory_context", ""),
        )
        return {
            "draft_answer": draft_answer,
            "trace_messages": trace_messages,
        }

    def answer_review_node(state: CustomerServiceState) -> dict:
        draft_answer = state.get("draft_answer", "")
        final_answer = review_answer(
            state["query"],
            draft_answer,
            state.get("history", []),
            state.get("trace_messages", []),
        )
        return {
            "final_answer": final_answer or draft_answer,
            "review_passed": bool(final_answer),
        }

    def save_memory_node(state: CustomerServiceState) -> dict:
        final_answer = state.get("final_answer", "")
        if final_answer:
            save_memory(state["query"], final_answer, state.get("history", []))
        return {}

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("main_agent_generate", main_agent_generate_node)
    graph.add_node("answer_review", answer_review_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "main_agent_generate")
    graph.add_edge("main_agent_generate", "answer_review")
    graph.add_edge("answer_review", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
