from pathlib import Path

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
    