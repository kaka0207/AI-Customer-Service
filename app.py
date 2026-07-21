import time
import json
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from utils.network_env import clear_invalid_dashscope_proxy

clear_invalid_dashscope_proxy()


CONVERSATION_STORE_PATH = Path(__file__).resolve().parent / "data" / "conversations.json"
UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "uploads"
QUICK_QUESTIONS = [
    "扫地机器人吸力变弱怎么办？",
    "滤网多久需要更换一次？",
    "机器人回不了基站怎么处理？",
    "帮我生成这个月使用报告",
]


st.set_page_config(
    page_title="扫地机器人智能客服",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 1.5rem;
            padding-bottom: 6rem;
        }
        [data-testid="stSidebar"] {
            background: #f7f8fa;
            border-right: 1px solid #e6e8ec;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }
        .app-header {
            border-bottom: 1px solid #eceff3;
            margin-bottom: 1rem;
            padding-bottom: 0.85rem;
        }
        .app-title {
            color: #172033;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: 0;
            line-height: 1.2;
            margin: 0;
        }
        .app-subtitle {
            color: #667085;
            font-size: 0.95rem;
            margin: 0.35rem 0 0 0;
        }
        .side-title {
            color: #172033;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .side-caption {
            color: #667085;
            font-size: 0.82rem;
            line-height: 1.45;
            margin-bottom: 0.7rem;
        }
        .conversation-meta {
            color: #98a2b3;
            font-size: 0.78rem;
            margin: -0.25rem 0 0.55rem 0;
        }
        .empty-state {
            align-items: center;
            border: 1px dashed #d0d5dd;
            border-radius: 8px;
            color: #667085;
            display: flex;
            justify-content: center;
            min-height: 280px;
            padding: 2rem;
            text-align: center;
        }
        .status-strip {
            background: #f8fafc;
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            color: #475467;
            font-size: 0.88rem;
            margin-bottom: 1rem;
            padding: 0.65rem 0.8rem;
        }
        .input-panel {
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            margin: 0.8rem 0 1rem 0;
            padding: 0.8rem;
        }
        .panel-title {
            color: #344054;
            font-size: 0.92rem;
            font-weight: 650;
            margin-bottom: 0.45rem;
        }
        .stButton > button {
            border-radius: 8px;
            min-height: 2.35rem;
            width: 100%;
        }
        [data-testid="stChatMessage"] {
            border-radius: 8px;
            padding: 0.4rem 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_agent():
    if "agent" in st.session_state:
        return

    try:
        from agent.react_agent import ReactAgent

        st.session_state["agent"] = ReactAgent()
    except Exception as exc:
        st.session_state["agent_error"] = str(exc)


def load_conversations() -> dict:
    if not CONVERSATION_STORE_PATH.exists():
        return {}

    try:
        with open(CONVERSATION_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    conversations = data.get("conversations", {})
    if not isinstance(conversations, dict):
        return {}

    return sanitize_conversations(conversations)


def sanitize_conversations(conversations: dict) -> dict:
    for conversation in conversations.values():
        title = conversation.get("title")
        if isinstance(title, str):
            clean_title = sanitize_model_text(title)
            if clean_title and clean_title != title:
                conversation["title"] = clean_title[:18] + ("..." if len(clean_title) > 18 else "")

        cleaned_messages = []
        for message in conversation.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = sanitize_model_text(content)
            if not message.get("content"):
                continue
            if cleaned_messages and cleaned_messages[-1].get("role") == message.get("role") and cleaned_messages[-1].get("content") == message.get("content"):
                continue
            cleaned_messages.append(message)
        conversation["messages"] = cleaned_messages

    return conversations


def dedupe_texts(texts: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for text in texts:
        item = str(text).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def extract_embedded_asr_text(text: str) -> str:
    if "sentence_id" not in text or "begin_time" not in text:
        return ""

    matches = re.findall(r"['\"]text['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    if not matches:
        return ""

    sentence_like = [
        item for item in matches
        if len(item.strip()) > 2 or any(mark in item for mark in "，,。.!！?？")
    ]
    candidates = dedupe_texts(sentence_like or matches)
    return "\n".join(candidates[:3]).strip()


def sanitize_model_text(text: str) -> str:
    embedded_asr_text = extract_embedded_asr_text(text)
    if embedded_asr_text:
        return embedded_asr_text

    lines = []
    skip_attachment_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "【用户上传附件】":
            skip_attachment_block = True
            continue

        if skip_attachment_block:
            if stripped.startswith("语音转写："):
                lines.append(stripped.replace("语音转写：", ""))
            continue

        if (
                stripped.startswith("保存路径：")
                or stripped.startswith("保存路径:")
                or "file:///" in stripped
                or "E:\\Github" in stripped
                or "ASR 返回为空" in stripped
                or "url error" in stripped.lower()
                or "provided URL" in stripped
        ):
            continue
        else:
            lines.append(line)

    cleaned = "\n".join(line for line in lines if line.strip()).strip()
    return cleaned or "用户上传了附件，但附件内容未能解析。"


def save_conversations():
    CONVERSATION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conversations = {
        conversation_id: conversation
        for conversation_id, conversation in st.session_state.get("conversations", {}).items()
        if conversation.get("messages")
    }
    active_conversation_id = st.session_state.get("active_conversation_id")
    if active_conversation_id not in conversations:
        active_conversation_id = None

    payload = {
        "active_conversation_id": active_conversation_id,
        "conversations": conversations,
    }
    temp_path = CONVERSATION_STORE_PATH.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    try:
        temp_path.replace(CONVERSATION_STORE_PATH)
    except PermissionError:
        with open(CONVERSATION_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        try:
            temp_path.unlink()
        except OSError:
            pass


def load_active_conversation_id(conversations: dict) -> str | None:
    if not CONVERSATION_STORE_PATH.exists():
        return None

    try:
        with open(CONVERSATION_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    active_id = data.get("active_conversation_id")
    if active_id in conversations:
        return active_id

    return None


def create_conversation(title: str = "新的对话") -> str:
    active_id = st.session_state.get("active_conversation_id")
    if active_id in st.session_state["conversations"]:
        active_conversation = st.session_state["conversations"][active_id]
        if not active_conversation.get("messages"):
            st.session_state["conversations"].pop(active_id, None)

    conversation_id = str(uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state["conversations"][conversation_id] = {
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    st.session_state["active_conversation_id"] = conversation_id
    save_conversations()
    return conversation_id


def ensure_conversations():
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = load_conversations()
    else:
        st.session_state["conversations"] = sanitize_conversations(st.session_state["conversations"])
    save_conversations()

    if "active_conversation_id" not in st.session_state:
        active_id = load_active_conversation_id(st.session_state["conversations"])
        if active_id:
            st.session_state["active_conversation_id"] = active_id
        else:
            create_conversation()

    active_id = st.session_state["active_conversation_id"]
    if active_id not in st.session_state["conversations"]:
        create_conversation()


def get_active_conversation():
    return st.session_state["conversations"][st.session_state["active_conversation_id"]]


def update_conversation_title(conversation: dict, prompt: str):
    if conversation["title"] != "新的对话":
        return

    title = prompt.strip().replace("\n", " ")
    conversation["title"] = title[:18] + ("..." if len(title) > 18 else "")


def save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def save_audio_file(audio_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = UPLOAD_DIR / f"{timestamp}_voice_input.wav"
    with open(target_path, "wb") as f:
        f.write(audio_file.getbuffer())
    return target_path


def transcribe_audio(audio_path: Path) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        st.session_state["last_asr_error"] = "DASHSCOPE_API_KEY 未设置"
        return ""

    try:
        from utils.config_handler import rag_conf

        model_name = rag_conf.get("asr_local_model_name", "paraformer-realtime-v2")
        response = call_local_asr(audio_path, model_name)
        transcript = extract_asr_text(response)

        if not transcript:
            st.session_state["last_asr_error"] = f"ASR 返回为空：{response}"
        else:
            st.session_state.pop("last_asr_error", None)
        return transcript
    except Exception as exc:
        st.session_state["last_asr_error"] = str(exc)
        st.warning("语音转写失败，已保留录音文件，请补充文字描述。")
        return ""


def call_local_asr(audio_path: Path, model_name: str):
    from dashscope.audio.asr import Recognition

    audio_format, sample_rate = get_audio_format(audio_path)
    recognition = Recognition(
        model=model_name,
        callback=None,
        format=audio_format,
        sample_rate=sample_rate,
    )
    return recognition.call(file=str(audio_path.resolve()))


def get_audio_format(audio_path: Path) -> tuple[str, int]:
    if audio_path.suffix.lower() == ".wav":
        try:
            import wave

            with wave.open(str(audio_path), "rb") as wav_file:
                return "wav", wav_file.getframerate()
        except Exception:
            return "wav", 16000

    return audio_path.suffix.lower().lstrip(".") or "wav", 16000


def collect_asr_text(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        for key in ("text", "transcript", "transcription"):
            if value.get(key):
                return collect_asr_text(value.get(key))

        texts = []
        for key in ("results", "sentences", "sentence", "words"):
            if value.get(key):
                texts.extend(collect_asr_text(value.get(key)))
        return texts
    if isinstance(value, list):
        texts = []
        for item in value:
            texts.extend(collect_asr_text(item))
        return texts

    return []


def extract_asr_text(response) -> str:
    if hasattr(response, "get_sentence"):
        transcript = "\n".join(dedupe_texts(collect_asr_text(response.get_sentence()))).strip()
        if transcript:
            return transcript

    if hasattr(response, "output"):
        output = response.output
    elif isinstance(response, dict):
        output = response.get("output", {})
    else:
        return ""

    if hasattr(output, "to_dict"):
        output = output.to_dict()

    if isinstance(output, dict):
        for key in ("text", "transcript", "transcription"):
            if output.get(key):
                return str(output[key]).strip()

        results = output.get("results") or output.get("sentences") or output.get("sentence")
        texts = dedupe_texts(collect_asr_text(results))
        if texts:
            return "\n".join(texts).strip()

    choices = output.get("choices", []) if isinstance(output, dict) else getattr(output, "choices", [])
    if not choices:
        return ""

    choice = choices[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else getattr(choice, "message", {})
    content = message.get("content", []) if isinstance(message, dict) else getattr(message, "content", [])
    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("text"):
            texts.append(str(item["text"]))
        elif hasattr(item, "text"):
            texts.append(str(item.text))

    return "\n".join(texts).strip()


def file_fingerprint(file_obj) -> str:
    name = getattr(file_obj, "name", "unnamed")
    file_type = getattr(file_obj, "type", "")
    try:
        content = bytes(file_obj.getbuffer())
    except Exception:
        content = str(file_obj).encode("utf-8", errors="ignore")

    digest = hashlib.sha1(content).hexdigest()
    return f"{name}:{file_type}:{len(content)}:{digest}"


def build_attachment_signature(uploaded_files, audio_file=None) -> str:
    parts = []
    for uploaded_file in uploaded_files or []:
        parts.append(f"file:{file_fingerprint(uploaded_file)}")
    if audio_file is not None:
        parts.append(f"audio:{file_fingerprint(audio_file)}")

    if not parts:
        return ""

    joined = "|".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def build_attachment_context(uploaded_files, audio_file=None) -> tuple[str, str]:
    if not uploaded_files and audio_file is None:
        return "", ""

    lines = ["\n\n【用户上传附件】"]
    audio_transcript = ""
    for uploaded_file in uploaded_files:
        saved_path = save_uploaded_file(uploaded_file)
        file_type = uploaded_file.type or "unknown"
        lines.append(f"- 文件名：{uploaded_file.name}")
        lines.append(f"  类型：{file_type}")
        lines.append("  状态：附件已保存到本地 uploads 目录。")

        if file_type.startswith("text/") or uploaded_file.name.lower().endswith((".txt", ".md", ".csv", ".json")):
            try:
                text = saved_path.read_text(encoding="utf-8")[:1200]
                lines.append(f"  文本预览：{text}")
            except UnicodeDecodeError:
                lines.append("  文本预览：文件编码不是 UTF-8，暂无法读取。")
        else:
            lines.append("  说明：该附件已保存，当前版本仅将文件元信息提供给客服。")

    if audio_file is not None:
        audio_path = save_audio_file(audio_file)
        transcript = transcribe_audio(audio_path)
        audio_transcript = transcript
        lines.append("- 语音输入")
        lines.append("  状态：录音已保存到本地 uploads 目录。")
        if transcript:
            lines.append(f"  语音转写：{transcript}")
        else:
            lines.append("  说明：用户录入了一段语音，但 ASR 未能完成转写。请用户补充文字描述。")

    return "\n".join(lines), audio_transcript


def render_assist_inputs():
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">快捷输入</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    for index, question in enumerate(QUICK_QUESTIONS):
        if columns[index].button(question, use_container_width=True):
            st.session_state["pending_prompt"] = question
            st.rerun()

    st.markdown('<div class="panel-title">附件</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "上传说明书片段、日志、维修单或数据文件",
        type=["txt", "md", "csv", "json", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="panel-title">语音</div>', unsafe_allow_html=True)
    audio_file = st.audio_input("录音描述问题", label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_files, audio_file


def resolve_prompt(user_prompt: str | None, uploaded_files, audio_file=None) -> tuple[str, str, str] | None:
    prompt = user_prompt or st.session_state.pop("pending_prompt", None)
    attachment_signature = build_attachment_signature(uploaded_files, audio_file)

    if (
        not prompt
        and attachment_signature
        and attachment_signature == st.session_state.get("last_consumed_attachment_signature")
    ):
        return None

    attachment_context, audio_transcript = build_attachment_context(uploaded_files, audio_file)

    if not prompt and audio_transcript:
        prompt = audio_transcript
    elif not prompt and audio_file is not None and not audio_transcript and not uploaded_files:
        st.session_state["last_consumed_attachment_signature"] = attachment_signature
        return None
    elif not prompt and attachment_context:
        prompt = "请根据我上传的附件帮我分析扫地机器人相关问题。"

    if not prompt:
        return None

    return prompt + attachment_context, prompt, attachment_signature


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="side-title">历史对话</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="side-caption">聊天记录会保存到本地，可在重启服务后继续切换上下文。</div>',
            unsafe_allow_html=True,
        )

        if st.button("新建对话", use_container_width=True):
            create_conversation()
            st.rerun()

        if st.button("清空当前对话", use_container_width=True):
            conversation = get_active_conversation()
            conversation["messages"] = []
            conversation["title"] = "新的对话"
            conversation["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            save_conversations()
            st.rerun()

        st.divider()

        conversations = sorted(
            (
                (conversation_id, conversation)
                for conversation_id, conversation in st.session_state["conversations"].items()
                if conversation.get("messages")
            ),
            key=lambda item: item[1]["updated_at"],
            reverse=True,
        )

        for conversation_id, conversation in conversations:
            is_active = conversation_id == st.session_state["active_conversation_id"]
            label = conversation["title"]
            if is_active:
                label = f"当前：{label}"

            if st.button(label, key=f"conversation_{conversation_id}", use_container_width=True):
                st.session_state["active_conversation_id"] = conversation_id
                save_conversations()
                st.rerun()

            message_count = len(conversation["messages"])
            st.markdown(
                f'<div class="conversation-meta">{conversation["updated_at"]} · {message_count} 条消息</div>',
                unsafe_allow_html=True,
            )


def render_header(conversation: dict):
    st.markdown(
        f"""
        <div class="app-header">
            <p class="app-title">扫地机器人智能客服</p>
            <p class="app-subtitle">当前对话：{conversation["title"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="status-strip">支持知识库检索、天气环境判断、使用报告生成和跨会话记忆。</div>',
        unsafe_allow_html=True,
    )


def render_messages(messages: list[dict]):
    if not messages:
        st.markdown(
            """
            <div class="empty-state">
                可以直接询问故障排查、保养周期、选购建议，或让客服生成你的使用报告。
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in messages:
        st.chat_message(message["role"]).write(message["content"])


apply_styles()
ensure_conversations()
init_agent()
render_sidebar()

conversation = get_active_conversation()
render_header(conversation)

if "agent_error" in st.session_state and "agent" not in st.session_state:
    st.error("智能客服初始化失败，请先完成模型配置后重启应用。")
    st.code('$env:DASHSCOPE_API_KEY="你的阿里云 DashScope Key"', language="powershell")
    st.info("同时请在 config/agent.yml 中填写有效的 gaodekey。")
    st.caption(f"当前错误：{st.session_state['agent_error']}")
    st.stop()

render_messages(conversation["messages"])

uploaded_files, audio_file = render_assist_inputs()
chat_prompt = st.chat_input("输入你的问题，例如：滤网多久更换一次？")
resolved_prompt = resolve_prompt(chat_prompt, uploaded_files, audio_file)

if resolved_prompt:
    prompt, visible_prompt, attachment_signature = resolved_prompt
    update_conversation_title(conversation, visible_prompt)
    conversation["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    conversation["messages"].append({"role": "user", "content": visible_prompt})
    save_conversations()
    st.chat_message("user").write(visible_prompt)

    response_messages = []
    with st.spinner("智能客服思考中..."):
        history_for_agent = conversation["messages"][:-1]
        res_stream = st.session_state["agent"].execute_stream(prompt, history_for_agent)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        assistant_reply = "".join(response_messages).strip()
        conversation["messages"].append({"role": "assistant", "content": assistant_reply})
        conversation["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        if attachment_signature:
            st.session_state["last_consumed_attachment_signature"] = attachment_signature
        save_conversations()
