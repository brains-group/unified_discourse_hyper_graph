import os
import dspy
import openai
from typing import Optional
from dotenv import load_dotenv



def configure_dspy(
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        cache: Optional[bool] = None,
        num_retries: int = 1,
) -> None:
    """
    Unified DSPy configuration function supporting both explicit parameter passing
    and environment-variable-backed defaults.
    """
    # 1. Load environment variables for fallback behavior
    load_dotenv()

    # 2. Resolve Model Name
    if model is None:
        # Fall back to Function 2's environment variable logic
        env_model = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
        model = f"openai/{env_model}"

    # 3. Resolve API Base and API Key
    if api_base is None:
        api_base = os.getenv("API_BASE", "http://localhost:8000/v1")

    if api_key is None:
        # Use Function 2's dummy key if no explicit key is provided
        api_key = "sk-no-key-needed"

    # 4. Resolve Max Tokens & Cache defaults
    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_TOKENS", "4096"))

    if cache is None:
        # If model is explicitly passed (Func 1), disable cache. Otherwise, enable it (Func 2).
        # Checking if model looks like the default helps preserve original intents.
        cache = False if not model.startswith("openai/") else True

    # 5. Global Side-Effects (From Function 2)
    openai.api_base = api_base
    openai.api_key = api_key

    # 6. Build DSPy LM Arguments
    kwargs: dict = dict(
        model=model,
        api_key=api_key,
        api_base=api_base,
        cache=cache,
        temperature=temperature,
        max_tokens=max_tokens,
        model_type="chat",
        num_retries=num_retries,
    )

    # Function 1 DeepSeek rule: Suppress extended thinking tokens
    if model.startswith("deepseek/"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    # 7. HTTP connection pool for litellm's synchronous calls. litellm builds its httpx clients (HTTPHandler for
    #    providers such as deepseek, and the OpenAI SDK path via client_session) without limits, i.e. httpx's
    #    default of 100 connections, so any number of worker threads above ~100 just queues for a connection.
    #    Size the pool with LLM_MAX_CONNECTIONS (default 2500); set before the first call creates a cached client.
    import httpx
    import litellm
    max_conn = int(os.getenv("LLM_MAX_CONNECTIONS") or 2500)
    if getattr(litellm, "_nkg_max_connections", 0) < max_conn:
        limits = httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn)
        litellm.sync_transport = httpx.HTTPTransport(limits=limits)
        litellm.client_session = httpx.Client(limits=limits, follow_redirects=True)
        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache is not None and hasattr(cache, "flush_cache"):
            cache.flush_cache()  # drop clients created with the old pool
        litellm._nkg_max_connections = max_conn

    # 8. Initialize and Configure DSPy
    lm = dspy.LM(**kwargs)

    # Merged configurations from both functions
    dspy.configure(lm=lm)
    dspy.settings.configure(lm=lm, temperature=temperature)