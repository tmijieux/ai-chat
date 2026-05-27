from pathlib import Path
import pathspec

# Directories that are always ignored regardless of .gitignore
_HARDCODED_IGNORE_DIRS = {
    "venv", ".venv", "node_modules", ".git", "__pycache__",
    "dist", "build", ".tox",
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
    """Return True if path should be excluded by the ignore spec."""
    try:
        rel = path.relative_to(workspace)
    except ValueError:
        return False
    # Check if any directory component is in the hardcoded set
    for part in rel.parts:
        if part in _HARDCODED_IGNORE_DIRS:
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
    