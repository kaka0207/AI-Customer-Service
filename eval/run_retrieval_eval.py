import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf


DEFAULT_CASES_PATH = PROJECT_ROOT / "eval" / "qa_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"


def load_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_mode(mode: str, cases: list[dict], top_k: int) -> dict:
    service = VectorStoreService()
    service.retrieval_mode = mode

    if mode == "bm25":
        chroma_conf["bm25_k"] = top_k
    else:
        chroma_conf["k"] = top_k

    retriever = service.get_retriever()
    case_results = []

    for case in cases:
        documents = retriever.invoke(case["question"])
        joined_context = "\n".join(doc.page_content for doc in documents)
        source_text = "\n".join(str(doc.metadata.get("source", "")) for doc in documents)

        keyword_hits = [
            keyword for keyword in case.get("expected_keywords", [])
            if keyword in joined_context
        ]
        source_hits = [
            source for source in case.get("expected_sources", [])
            if source in source_text
        ]

        case_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "keyword_hit_count": len(keyword_hits),
                "keyword_total": len(case.get("expected_keywords", [])),
                "keyword_recall": _safe_divide(len(keyword_hits), len(case.get("expected_keywords", []))),
                "source_hit": bool(source_hits),
                "keyword_hits": keyword_hits,
                "source_hits": source_hits,
                "top_documents": [
                    {
                        "rank": index,
                        "source": doc.metadata.get("source", ""),
                        "page": doc.metadata.get("page", ""),
                        "preview": doc.page_content.replace("\n", " ")[:180],
                    }
                    for index, doc in enumerate(documents, start=1)
                ],
            }
        )

    return summarize_mode(mode, case_results)


def summarize_mode(mode: str, case_results: list[dict]) -> dict:
    total = len(case_results)
    keyword_recall_avg = _safe_divide(
        sum(result["keyword_recall"] for result in case_results),
        total,
    )
    source_hit_rate = _safe_divide(
        sum(1 for result in case_results if result["source_hit"]),
        total,
    )
    full_keyword_hit_rate = _safe_divide(
        sum(1 for result in case_results if result["keyword_hit_count"] == result["keyword_total"]),
        total,
    )

    return {
        "mode": mode,
        "case_count": total,
        "keyword_recall_avg": round(keyword_recall_avg, 4),
        "source_hit_rate": round(source_hit_rate, 4),
        "full_keyword_hit_rate": round(full_keyword_hit_rate, 4),
        "category_summary": summarize_by_category(case_results),
        "cases": case_results,
    }


def summarize_by_category(case_results: list[dict]) -> dict:
    categories = {}
    for result in case_results:
        categories.setdefault(result["category"], []).append(result)

    summary = {}
    for category, results in categories.items():
        total = len(results)
        summary[category] = {
            "case_count": total,
            "keyword_recall_avg": round(
                _safe_divide(sum(item["keyword_recall"] for item in results), total),
                4,
            ),
            "source_hit_rate": round(
                _safe_divide(sum(1 for item in results if item["source_hit"]), total),
                4,
            ),
            "full_keyword_hit_rate": round(
                _safe_divide(
                    sum(1 for item in results if item["keyword_hit_count"] == item["keyword_total"]),
                    total,
                ),
                4,
            ),
        }

    return summary


def write_reports(results: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"retrieval_eval_{timestamp}.json"
    md_path = output_dir / f"retrieval_eval_{timestamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": timestamp, "results": results}, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RAG 检索评测报告\n\n")
        f.write(f"- 生成时间：{timestamp}\n")
        f.write("- 评估口径：Top-K 召回内容中的关键词命中率、来源命中率。\n\n")
        f.write("## 模式对比\n\n")
        f.write("| 检索模式 | 用例数 | 平均关键词召回 | 来源命中率 | 全关键词命中率 | 状态 |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for result in results:
            status = result.get("error", "ok")
            f.write(
                f"| {result['mode']} | {result['case_count']} | "
                f"{result['keyword_recall_avg']:.2%} | {result['source_hit_rate']:.2%} | "
                f"{result['full_keyword_hit_rate']:.2%} | {status} |\n"
            )

        f.write("\n## 分类对比\n\n")
        f.write("| 检索模式 | 分类 | 用例数 | 平均关键词召回 | 来源命中率 | 全关键词命中率 |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for result in results:
            for category, summary in result.get("category_summary", {}).items():
                f.write(
                    f"| {result['mode']} | {category} | {summary['case_count']} | "
                    f"{summary['keyword_recall_avg']:.2%} | {summary['source_hit_rate']:.2%} | "
                    f"{summary['full_keyword_hit_rate']:.2%} |\n"
                )

        for result in results:
            f.write(f"\n## {result['mode']} 明细\n\n")
            for case in result["cases"]:
                f.write(f"### {case['id']}（{case['category']}）\n\n")
                f.write(f"- 问题：{case['question']}\n")
                f.write(f"- 关键词命中：{case['keyword_hit_count']}/{case['keyword_total']}，")
                f.write(f"来源命中：{'是' if case['source_hit'] else '否'}\n")
                f.write(f"- 命中关键词：{', '.join(case['keyword_hits']) or '无'}\n")
                f.write("- Top 文档：\n")
                for doc in case["top_documents"]:
                    f.write(
                        f"  - #{doc['rank']} {doc['source']} page={doc['page']}："
                        f"{doc['preview']}\n"
                    )
                f.write("\n")

    return json_path, md_path


def _safe_divide(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to qa_cases.json.")
    parser.add_argument("--modes", nargs="+", default=["bm25"], help="Retrieval modes to evaluate.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K documents per query.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for reports.")
    args = parser.parse_args()

    original_conf = copy.deepcopy(chroma_conf)
    cases = load_cases(Path(args.cases))
    results = []
    for mode in args.modes:
        try:
            results.append(run_mode(mode, cases, args.top_k))
        except Exception as e:
            results.append(
                {
                    "mode": mode,
                    "case_count": len(cases),
                    "keyword_recall_avg": 0.0,
                    "source_hit_rate": 0.0,
                    "full_keyword_hit_rate": 0.0,
                    "error": str(e),
                    "cases": [],
                }
            )
        finally:
            chroma_conf.clear()
            chroma_conf.update(copy.deepcopy(original_conf))

    json_path, md_path = write_reports(results, Path(args.output_dir))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    for result in results:
        if "error" in result:
            print(f"{result['mode']}: ERROR {result['error']}")
        else:
            print(
                f"{result['mode']}: keyword_recall={result['keyword_recall_avg']:.2%}, "
                f"source_hit={result['source_hit_rate']:.2%}, "
                f"full_keyword_hit={result['full_keyword_hit_rate']:.2%}"
            )


if __name__ == "__main__":
    main()
