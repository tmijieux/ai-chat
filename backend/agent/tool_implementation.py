import os
import json
from typing import Any
from pathlib import Path
import subprocess
from .file_utils import file_in_directory

def _tool_error(call_name: str, call_id: str, error_message: str, user_message: str|None=None):
    response = {
        "tool":call_name,
        "tool_call_id":call_id,
        "status":"error",
        "error":{
            "message":error_message
        }
    }
    if user_message is not None:
        response["error"]["user_message"] = user_message
    return json.dumps(response)



NO = ("n","N", "no","No","NO")
YES = ("y","Y","yes", "Yes","YES")
POSSIBLE_ANSWERS = YES + NO

def user_confirmation(ask: str) -> tuple[bool, str|None]:
    user_choice = ""
    print(ask)
    while user_choice not in POSSIBLE_ANSWERS:
        user_choice = input("Would you like to overwrite with this content ? (y/n)")
        user_choice = user_choice.strip()
        
        if user_choice not in POSSIBLE_ANSWERS:
            print(f"Invalid choice `{user_choice}`. Please answer again")    

    if user_choice not in YES:
        user_message = input("Reason for aborting (empty for nothing): ")
        user_message = user_message.strip()
        if user_message == "":
            user_message = None
        return False, user_message

    return True, None

def tool_write_file(call_name: str, call_id: str, arguments: dict[str, Any]):
    path = arguments.get("file_path", "")
    content = arguments.get("content", "")
    append = arguments.get("append", False)
    try:
        path = Path(path)
        if not path.is_file():
            return _tool_error(call_name, call_id, "The file does not exists or is not a regular file")
        

        confirmed, user_msg = user_confirmation(
            f"The agent would like to overwrite the file `{path}` with the following content:\n `{content}`\n\n",
        )
        if not confirmed:
            return _tool_error(call_name, call_id, "The user aborted the modification", user_message=user_msg)
            
        
        current_working_directory = os.path.realpath(os.getcwd())
        file_real_path = os.path.realpath(path)
        if not file_in_directory(file_real_path, current_working_directory):
            return _tool_error(
                call_name, 
                call_id, 
                f"Writing to file outside of your working directory is forbidden. Your working directory is {current_working_directory}",
            )


        mode = "ab" if append else "wb"
        with open(path, mode=mode) as f:
            f.write(content.encode())

        response = {
            "tool":call_name,
            "tool_call_id": call_id,
            "status":"success",
        }
    except Exception as e:
        response = {
            "tool": call_name,
            "tool_call_id": call_id,
            "status":"error",
            "error":{
                "message":f"Unexpected error: {str(e)}"
            }
        }
    return json.dumps(response)

def tool_read_file(call_name: str, call_id: str, arguments: dict[str, Any]):
    path = arguments.get("file_path", "")
    try:
        #file_content = Path(path).read_text(encoding="utf-8")[:10000]
        file_content = Path(path).read_text(encoding="utf-8")

        response = {
            "tool":call_name,
            "tool_call_id": call_id,
            "status":"success",
            "content": file_content
        }
    except Exception as e:
        response = {
            "tool": call_name,
            "tool_call_id": call_id,
            "status":"error",
            "error":{
                "message":f"Error reading file: {str(e)}"
            }
        }
    return json.dumps(response)

def tool_edit_file(call_name: str, call_id: str, arguments: dict[str, Any]):
    """
    Edit an existing file by replacing old_string with new_string.
    Always reads the file first before editing to ensure correct content.
    """
    path = arguments.get("file_path", "")
    old_string = arguments.get("old_string", "")
    new_string = arguments.get("new_string", "")
    replace_all = arguments.get("replace_all", False)

    # Validate inputs
    if not path:
        return _tool_error(call_name, call_id, "file_path is required")
    if not old_string:
        return _tool_error(call_name, call_id, "old_string is required")
    if not new_string:
        return _tool_error(call_name, call_id, "new_string is required")
    if old_string == new_string:
        return _tool_error(call_name, call_id, "old_string and new_string must be different")

    try:
        path_obj = Path(path)

        # Check if file exists
        if not path_obj.is_file():
            return _tool_error(call_name, call_id, f"The file '{path}' does not exist or is not a regular file")

        # Read file content first (required by tool design)
        current_content = path_obj.read_text(encoding="utf-8")

        # Security check: ensure file is within working directory
        current_working_directory = os.path.realpath(os.getcwd())
        file_real_path = os.path.realpath(path)
        if not file_in_directory(file_real_path, current_working_directory):
            return _tool_error(
                call_name,
                call_id,
                f"Editing file outside of your working directory is forbidden. Your working directory is {current_working_directory}",
            )

        if old_string not in current_content:
            return _tool_error(call_name, call_id, f"old_string '{old_string}' not found in file")

        # Perform replacement        
        if replace_all:
            # Replace all occurrences
            new_content = current_content.replace(old_string, new_string)
        else:
            # Replace only first occurrence
            # Find the first occurrence and replace only that one
            first_index = current_content.find(old_string)
            if first_index == -1:
                return _tool_error(call_name, call_id, f"old_string '{old_string}' not found in file")
            new_content = current_content[:first_index] + new_string + current_content[first_index + len(old_string):]

        confirmed, user_msg = user_confirmation(
            f"The agent will perfom the following replacement in the file `{path}` :\n`{old_string}`\n==>\n `{new_string}`\n\n",
        )
        if not confirmed:
            return _tool_error(call_name, call_id, "The user aborted the modification", user_message=user_msg)

        # Write back the modified content
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(new_content)

        response = {
            "tool": call_name,
            "tool_call_id": call_id,
            "status": "success",
            "message": f"Successfully edited {path}",
        }
    except Exception as e:
        response = {
            "tool": call_name,
            "tool_call_id": call_id,
            "status": "error",
            "error": {
                "message": f"Unexpected error: {str(e)}"
            }
        }
    return json.dumps(response)


def tool_list_directory(call_name: str, call_id: str, arguments: dict[str, Any]):
    path = arguments.get("path", "")


    is_recursive = arguments.get("recursive", False)
    maximum_depth = arguments.get("maximum_depth", 3 if is_recursive else 1)

    print("is_recursive=",is_recursive)
    print("maximum_depth=",maximum_depth)

    exe = "c:\\Program Files\\Git\\usr\\bin\\find.exe"
    cmd = [exe, path, "-maxdepth", str(maximum_depth)]
    
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 0:
        response = {
            "tool": call_name,
            "tool_call_id": call_id,
            "status": "success",
            "content": proc.stdout.decode()
        }
    else:
        response = {
            "tool": call_name,
            "tool_call_id": call_id,
            "status": "error",
            "error": {
                "message": proc.stderr.decode()
            }
        }
    return json.dumps(response)


def tool_run_shell(call_name: str, call_id: str, arguments: dict[str, Any]):
    """
    Execute a system shell command (bash).
    This is for running npm, git, docker, or complex scripts.
    """
    command = arguments.get("command", "")
    
    if not command:
        return _tool_error(call_name, call_id, "command is required")
    
    try:
        confirm, user_msg = user_confirmation(f"The agent want to invoke the following command `{command}`")
        if not confirm:
            return _tool_error(call_name, call_id, "the user aborted the call", user_message=user_msg)
        
        # Execute the shell command
        proc = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if proc.returncode == 0:
            response = {
                "tool": call_name,
                "tool_call_id": call_id,
                "status": "success",
                "output": proc.stdout
            }
        else:
            response = {
                "tool": call_name,
                "tool_call_id": call_id,
                "status": "error",
                "error": {
                    "message": proc.stderr
                }
            }
    except Exception as e:
        response = {
            "tool": call_name,
            "tool_call_id": call_id,
            "status": "error",
            "error": {
                "message": f"Unexpected error: {str(e)}"
            }
        }
    return json.dumps(response)


def tool_grep(call_name: str, call_id: str, arguments: dict[str, Any]):
    """
    A powerful search tool built on ripgrep (rg).
    Supports full regex syntax, file filtering, and multiple output modes.
    """
    pattern = arguments.get("pattern", "")
    path = arguments.get("path", ".")
    glob = arguments.get("glob", None)
    output_mode = arguments.get("output_mode", "files_with_matches")
    context_before = arguments.get("-B", 0)
    context_after = arguments.get("-A", 0)
    context = arguments.get("context", 0)
    show_line_numbers = arguments.get("-n", True)
    case_insensitive = arguments.get("-i", False)
    only_matching = arguments.get("-o", False)
    file_type = arguments.get("type", None)
    head_limit = arguments.get("head_limit", 250)
    offset = arguments.get("offset", 0)
    multiline = arguments.get("multiline", False)
    
    if not pattern:
        return _tool_error(call_name, call_id, "pattern is required")
    
    try:
        # Build ripgrep command
        cmd = ["rg", pattern]
        
        # Add path if specified
        if path and path != ".":
            cmd.append(path)
        
        # Add glob filter
        if glob:
            cmd.extend(["--glob", glob])
        
        # Add output mode
        if output_mode == "content":
            cmd.append("--no-heading")
        elif output_mode == "count":
            cmd.append("--count")
        
        # Add context
        if context > 0:
            cmd.extend(["-C", str(context)])
        elif context_before > 0 or context_after > 0:
            if context_before > 0:
                cmd.extend(["-B", str(context_before)])
            if context_after > 0:
                cmd.extend(["-A", str(context_after)])
        
        # Add line numbers
        if show_line_numbers:
            cmd.append("--line-number")
        
        # Add case insensitive flag
        if case_insensitive:
            cmd.append("-i")
        
        # Add only matching flag
        if only_matching:
            cmd.append("--only-matching")
        
        # Add file type filter
        if file_type:
            cmd.extend(["--type", file_type])
        
        # Add head limit
        if head_limit and head_limit != 0:
            cmd.extend(["--max-columns", str(head_limit)])
        
        # Add offset
        if offset and offset != 0:
            cmd.extend(["--with-filename"])
        
        # Add multiline mode
        if multiline:
            cmd.extend(["-U", "--multiline-dotall"])
        
        # Execute command
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False
        )
        
        if proc.returncode == 0:
            response = {
                "tool": call_name,
                "tool_call_id": call_id,
                "status": "success",
                "content": proc.stdout.strip()
            }
        else:
            # Check if it's a "no matches" error vs actual error
            stderr = proc.stderr.strip()
            if "no matches" in stderr.lower():
                response = {
                    "tool": call_name,
                    "tool_call_id": call_id,
                    "status": "success",
                    "content": ""
                }
            else:
                response = {
                    "tool": call_name,
                    "tool_call_id": call_id,
                    "status": "error",
                    "error": {
                        "message": proc.stderr
                    }
                }
    except Exception as e:
        response = {
            "tool": call_name,
            "tool_call_id": call_id,
            "status": "error",
            "error": {
                "message": f"Unexpected error: {str(e)}"
            }
        }
    return json.dumps(response)


# Your tool execution functions
def execute_tool(call_name: str, call_id: str, arguments: dict[str, Any]) -> str:
    """
    Execute the tool that was called.
    Match the tool name from Ollama schema to your Python functions.
    """
    print(f"\n=== AGENT CALLING TOOL `{call_name}` ===")

    if call_name == "read_file":
        return tool_read_file(call_name, call_id, arguments)
    elif call_name == "list_directory":
        return tool_list_directory(call_name, call_id, arguments)
    elif call_name == "write_file":
        return tool_write_file(call_name, call_id, arguments)
    elif call_name == "edit_file":
        return tool_edit_file(call_name, call_id, arguments)
    elif call_name == "run_shell":
        return tool_run_shell(call_name, call_id, arguments)
    elif call_name == "search_files":
        return _not_implemented_error(call_name, call_id)
    elif call_name == "search_web":
        return _not_implemented_error(call_name, call_id)
    elif call_name == "grep":
        return tool_grep(call_name, call_id, arguments)
    else:
        return _not_implemented_error(call_name, call_id)


def _not_implemented_error(call_name: str, call_id: str) -> str:
    """
    Helper function to return a consistent 'not implemented' error for missing tools.
    """
    error = {
        "tool": call_name,
        "tool_call_id": call_id,
        "status": "error",
        "error":{
            "message": f"Tool `{call_name}` is not yet implemented. This agent is under development.",
        }
    }
    return json.dumps(error)
