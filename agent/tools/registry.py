import json
import os
from string import Formatter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


BUILTIN_TOOLS = [
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
]

TYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
}


def get_agent_tools():
    tools = list(BUILTIN_TOOLS)
    tools.extend(load_external_http_tools())
    return tools


def load_external_http_tools(config_path: str = "config/external_tools.yml"):
    config = _load_external_tools_config(config_path)
    if not config.get("enabled", False):
        return []

    tools = []
    for spec in config.get("http_tools", []):
        try:
            tools.append(build_http_tool(spec))
        except Exception as e:
            logger.error(f"[tool registry]外部工具加载失败 spec={spec} err={str(e)}", exc_info=True)

    return tools


def build_http_tool(spec: dict):
    name = spec["name"]
    description = spec["description"]
    args_schema = _build_args_schema(name, spec.get("params", []))

    def invoke_tool(**kwargs):
        return _call_http_tool(spec, kwargs)

    return StructuredTool.from_function(
        func=invoke_tool,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def _load_external_tools_config(config_path: str):
    path = get_abs_path(config_path)
    if not os.path.exists(path):
        return {"enabled": False, "http_tools": []}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {"enabled": False, "http_tools": []}


def _build_args_schema(tool_name: str, params: list[dict]):
    fields = {}
    for param in params:
        param_type = TYPE_MAP.get(str(param.get("type", "str")).lower(), str)
        description = param.get("description", "")
        required = bool(param.get("required", True))

        if required:
            fields[param["name"]] = (param_type, Field(..., description=description))
        else:
            fields[param["name"]] = (param_type | None, Field(None, description=description))

    return create_model(f"{tool_name.title().replace('_', '')}Args", **fields)


def _call_http_tool(spec: dict, args: dict[str, Any]) -> str:
    method = spec.get("method", "GET").upper()
    if method not in {"GET", "POST"}:
        raise ValueError(f"外部工具仅支持 GET/POST，当前 method={method}")

    timeout = float(spec.get("timeout", 5))
    url = _render_template(spec["url"], args)
    headers = _render_mapping(spec.get("headers", {}), args)
    query = _build_query(spec.get("params", []), args)
    body = _build_body(spec, args, method)

    if query:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{urlencode(query)}"

    request = Request(url=url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            text = response.read().decode(spec.get("encoding", "utf-8"))
            if spec.get("response_format", "text") == "json":
                return json.dumps(json.loads(text), ensure_ascii=False)
            return text
    except HTTPError as e:
        raise RuntimeError(f"外部工具HTTP错误: {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"外部工具网络错误: {e.reason}") from e


def _build_query(params: list[dict], args: dict[str, Any]) -> dict:
    query = {}
    for param in params:
        if param.get("location", "query") != "query":
            continue

        name = param["name"]
        if args.get(name) is not None:
            query[name] = args[name]

    return query


def _build_body(spec: dict, args: dict[str, Any], method: str):
    if method != "POST":
        return None

    body = {}
    for param in spec.get("params", []):
        if param.get("location") == "body":
            name = param["name"]
            body[name] = args.get(name)

    if not body:
        return None

    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _render_mapping(values: dict, args: dict[str, Any]) -> dict:
    return {key: _render_template(str(value), args) for key, value in values.items()}


def _render_template(template: str, args: dict[str, Any]) -> str:
    rendered = template
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        if field_name in args:
            rendered = rendered.replace("{" + field_name + "}", str(args[field_name]))

    while "${ENV:" in rendered:
        start = rendered.index("${ENV:")
        end = rendered.index("}", start)
        env_name = rendered[start + len("${ENV:"):end]
        rendered = rendered[:start] + os.getenv(env_name, "") + rendered[end + 1:]

    return rendered
