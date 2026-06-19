from pathlib import Path

from fastapi import APIRouter, HTTPException

import loaders as ld
from conv_helpers import _PROMPTS_DIR
from llm import backend
from message_types import LLMMessage

router = APIRouter()

_DUMMY_USER: list[LLMMessage] = [{"role": "user", "content": "."}]


def _load_prompt_file(path: Path) -> dict:
    """Parse a prompt YAML file and return its data dict."""
    import yaml as _yaml
    data = _yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "id": path.stem,
        "name": data.get("name") or path.stem,
        "category": data.get("category") or "general",
        "content": data.get("content") or "",
        "is_default": bool(data.get("is_default")),
        "token_count": data.get("token_count") or None,
    }


def _write_prompt_file(path: Path, name: str, category: str, content: str, is_default: bool, token_count: int | None) -> None:
    """Write a prompt YAML file."""
    import yaml as _yaml
    data = {
        "name": name,
        "category": category,
        "is_default": is_default,
        "token_count": token_count,
        "content": content,
    }
    path.write_text(_yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False), encoding="utf-8")


async def _compute_prompt_token_count(content: str) -> int:
    """Count tokens contributed by this system prompt (with vs without)."""
    tools: list = []
    system_msg: LLMMessage = {"role": "system", "content": content}
    with_prompt = await backend.count_tokens([system_msg] + _DUMMY_USER, tools)
    baseline = await backend.count_tokens(_DUMMY_USER, tools)
    return with_prompt - baseline


@router.get("/api/system-prompts")
async def list_system_prompts():
    """List all system prompt YAML files from backend/prompts/."""
    _PROMPTS_DIR.mkdir(exist_ok=True)
    return [_load_prompt_file(f) for f in sorted(_PROMPTS_DIR.glob("*.yaml"))]


@router.post("/api/system-prompts")
async def create_system_prompt(body: ld.NewSystemPrompt):
    """Create a new system prompt YAML file. Slug is derived from the name."""
    from conv_helpers import _slugify
    _PROMPTS_DIR.mkdir(exist_ok=True)
    slug = _slugify(body.name)
    path = _PROMPTS_DIR / f"{slug}.yaml"
    if path.exists():
        suffix = 2
        while (_PROMPTS_DIR / f"{slug}-{suffix}.yaml").exists():
            suffix += 1
        slug = f"{slug}-{suffix}"
        path = _PROMPTS_DIR / f"{slug}.yaml"
    token_count_value = await _compute_prompt_token_count(body.content)
    _write_prompt_file(path, body.name, body.category, body.content, body.is_default, token_count_value)
    return _load_prompt_file(path)


@router.put("/api/system-prompts/{slug}")
async def update_system_prompt(slug: str, body: ld.UpdateSystemPrompt):
    """Update an existing system prompt YAML file by slug."""
    path = _PROMPTS_DIR / f"{slug}.yaml"
    if not path.exists():
        raise HTTPException(404)
    current = _load_prompt_file(path)
    name = body.name if body.name is not None else current["name"]
    category = body.category if body.category is not None else current["category"]
    is_default = body.is_default if body.is_default is not None else current["is_default"]
    content = body.content if body.content is not None else current["content"]
    token_count = current["token_count"]
    if body.content is not None:
        token_count = await _compute_prompt_token_count(content)
    _write_prompt_file(path, name, category, content, is_default, token_count)
    return _load_prompt_file(path)


@router.delete("/api/system-prompts/{slug}")
async def delete_system_prompt(slug: str):
    """Delete a system prompt YAML file by slug."""
    path = _PROMPTS_DIR / f"{slug}.yaml"
    if path.exists():
        path.unlink()
    return ""
