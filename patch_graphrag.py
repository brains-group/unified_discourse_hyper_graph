import litellm
import inspect
from pydantic import BaseModel

print("[PATCH] patch_graphrag.py loaded successfully")

original_acompletion = litellm.acompletion
original_completion = litellm.completion

def patch_response_format(kwargs):
    if "response_format" in kwargs:
        val = kwargs["response_format"]
        # If it is a Pydantic model class, downgrade to standard JSON Mode
        if inspect.isclass(val) and issubclass(val, BaseModel):
            print(f"[PATCH] Downgrading response_format from Pydantic model {val.__name__} to standard JSON object")
            kwargs["response_format"] = {"type": "json_object"}
    return kwargs

async def patched_acompletion(*args, **kwargs):
    kwargs = patch_response_format(kwargs)
    return await original_acompletion(*args, **kwargs)

def patched_completion(*args, **kwargs):
    kwargs = patch_response_format(kwargs)
    return original_completion(*args, **kwargs)

litellm.acompletion = patched_acompletion
litellm.completion = patched_completion
print("[PATCH] litellm.acompletion and litellm.completion patched")
