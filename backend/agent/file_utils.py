from pathlib import Path
import pathspec

# Directories that are always ignored regardless of .gitignore
_HARDCODED_IGNORE_DIRS = {
    "venv", ".venv", "node_modules", ".git", "__pycache__",
    "dist", "build", ".tox", ".cache", ".angular"
}

# Extensions treated as non-text — no useful chunk/summary comes from reading these as source
# text, and several (media, archives) can be large enough to matter.
_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".mov", ".avi", ".webm",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".pdf", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pyc", ".pyo", ".o", ".a", ".so", ".dll", ".exe", ".bin", ".wasm",
    ".gguf", ".safetensors", ".onnx", ".pt", ".pth",
    ".sqlite", ".sqlite3", ".db",
}

# Generated lockfiles: authored by tooling, not humans, and disproportionately large relative to
# their information content — never worth a per-file summary or index entry.
_LOCKFILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "uv.lock",
    "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum",
}


def load_ignore_spec(workspace: str) -> pathspec.PathSpec:
    """Build a PathSpec from .gitignore (if present) plus hardcoded defaults."""
    patterns = [f"{d}/" for d in _HARDCODED_IGNORE_DIRS]
    gitignore = Path(workspace) / ".gitignore"
    if gitignore.is_file():
        try:
            patterns.extend(gitignore.read_text(encoding="utf-8").splitlines())
        except Exception:
            pass
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def is_path_ignored(path: Path, workspace: str, spec: pathspec.PathSpec) -> bool:
    """Return True if path should be excluded: hardcoded dirs, gitignore, non-text extensions, or
    generated lockfiles."""
    try:
        rel = path.relative_to(workspace)
    except ValueError:
        return False
    # Check if any directory component is in the hardcoded set
    for part in rel.parts:
        if part in _HARDCODED_IGNORE_DIRS:
            return True
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    if path.name in _LOCKFILE_NAMES:
        return True
    # Check gitignore spec (use forward slashes for cross-platform consistency)
    return spec.match_file(rel.as_posix())


def resolve_workspace_path(path: str, working_directory: str) -> Path:
    """Resolve path relative to working_directory if not absolute, then normalise."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(working_directory) / p
    return p.resolve()


def file_in_directory(file_path_str: str, directory_path_str: str) -> bool:
    """
    Check if a file is within a given directory.

    Args:
        file_path: Path to the file (relative or absolute)
        directory_path: Path to the directory to check against

    Returns:
        True if file exists inside the directory, False otherwise
    """
    dir_path = Path(directory_path_str)
    file_path = Path(file_path_str)

    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory '{directory_path_str}' does not exist.")

    return file_path.is_relative_to(dir_path)
    