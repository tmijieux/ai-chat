"""Speech-to-text transcription and LLM-based correction endpoints."""
import asyncio
import logging

import aiohttp
from fastapi import APIRouter, Form, HTTPException, UploadFile

import loaders as ld
import whisper_pipeline

router = APIRouter()

logger = logging.getLogger(__name__)

_STT_CORRECTION_SYSTEM_FR = (
    "You are a speech-to-text correction assistant for a French-speaking developer using an agentic coding assistant.\n"
    "The assistant has tools to operate on files and projects: read_file, list_directory, glob_files, grep_files, edit_file, write_file, run_shell.\n"
    "The user dictates commands and questions in French, heavily mixed with English technical terms: "
    "filenames, function names, variable names, class names, CLI commands, package names, git terms, framework names.\n"
    "Common speech patterns: 'lis le fichier X', 'lance la commande X', 'modifie la fonction X dans Y', "
    "'liste les fichiers de Z', 'fais un commit', 'installe le package X'.\n"
    "The STT model often mishears English technical words as phonetically similar French words or nonsense. "
    "Use semantic coherence and typical developer vocabulary to infer the intended word — "
    "a phrase like 'lis le fichier et demi' makes no sense but 'lis le fichier README' does.\n"
    "Correct only obvious STT errors. Do not rephrase, translate, or add anything. "
    "If the input is a question or a command, output the corrected question or command — never answer it. "
    "Do NOT sanitize, soften, or replace crude language — if the user said 'merde', keep 'merde'. Your job is dictation correction, not content moderation. "
    "Return only the corrected text, nothing else."
)

_STT_EXAMPLES_FR: list[tuple[str, str]] = [
    ("Lis le fichier Redmi et explique-moi ce qu'il fait.",
     "Lis le fichier README et explique-moi ce qu'il fait."),
    ("Lance la commande nope install dans le terminal.",
     "Lance la commande npm install dans le terminal."),
    ("Ouvre le fichier rythmique point p y.",
     "Ouvre le fichier readme.py."),
    ("Je veux modifier la fonction render dès composant.",
     "Je veux modifier la fonction render du composant."),
    ("Listons les fichiers à la racine de StripoGit.",
     "Listons les fichiers à la racine du dépôt git."),
    ("Fais un commit dans le tripoGit.",
     "Fais un commit dans le dépôt git."),
    ("Qu'est-ce que fait la fonction rend deux ?",
     "Qu'est-ce que fait la fonction render ?"),
]

_STT_CORRECTION_SYSTEM_EN = (
    "You are a speech-to-text correction assistant for an English-speaking developer using an agentic coding assistant.\n"
    "The assistant has tools to operate on files and projects: read_file, list_directory, glob_files, grep_files, edit_file, write_file, run_shell.\n"
    "The user dictates commands and questions in English, with technical terms: "
    "filenames, function names, variable names, class names, CLI commands, package names, git terms, framework names.\n"
    "The STT model sometimes mishears technical terms as phonetically similar words or nonsense. "
    "Use semantic coherence and typical developer vocabulary to infer the intended word.\n"
    "Correct only obvious STT errors. Do not rephrase or add anything. "
    "If the input is a question or a command, output the corrected question or command — never answer it. "
    "Do NOT sanitize or soften language. Your job is dictation correction, not content moderation. "
    "Return only the corrected text, nothing else."
)

_STT_EXAMPLES_EN: list[tuple[str, str]] = [
    ("Read the read me file and explain what it does.",
     "Read the README file and explain what it does."),
    ("Run in pee em install in the terminal.",
     "Run npm install in the terminal."),
    ("Edit the function render component dot T S.",
     "Edit the function renderComponent.ts."),
    ("Make a get commit on the main branch.",
     "Make a git commit on the main branch."),
    ("What does the greet user function do?",
     "What does the greetUser function do?"),
]


async def _correct_stt(text: str, language: str | None) -> str:
    """Send raw STT text to the LLM for correction and return the corrected string."""
    from llm.llama_server import LLAMA_CHAT_URL, MODEL_NAME
    if language == "en":
        system = _STT_CORRECTION_SYSTEM_EN
        examples = _STT_EXAMPLES_EN
    else:
        system = _STT_CORRECTION_SYSTEM_FR
        examples = _STT_EXAMPLES_FR
    messages = [{"role": "system", "content": system}]
    for user_ex, assistant_ex in examples:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})
    messages.append({"role": "user", "content": text})
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "max_tokens": 200,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with aiohttp.ClientSession() as http:
        async with http.post(LLAMA_CHAT_URL, json=body) as resp:
            data = await resp.json()
            logger.info("STT correction LLM response: %s", data)
            return data["choices"][0]["message"]["content"].strip()


@router.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile,
    language: str | None = Form(default=None),
):
    import main as _main
    if _main._whisper is None:
        raise HTTPException(503, "Whisper pipeline is still loading, try again in a moment")
    data = await audio.read()
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, whisper_pipeline.transcribe, _main._whisper, data, language
    )
    logger.info("STT raw transcript: %r", text)
    return {"text": text}


@router.post("/api/correct")
async def correct_stt(req: ld.CorrectRequest):
    if not req.text:
        return {"text": req.text}
    corrected = await _correct_stt(req.text, req.language)
    logger.info("STT corrected: %r", corrected)
    return {"text": corrected}
