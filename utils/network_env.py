import os


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def clear_invalid_dashscope_proxy():
    for key in PROXY_ENV_KEYS:
        value = os.getenv(key, "")
        normalized = value.lower()
        if "127.0.0.1:9" in normalized or "localhost:9" in normalized:
            os.environ.pop(key, None)
