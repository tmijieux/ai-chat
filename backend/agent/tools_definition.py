TOOLS = [
  {
    "name": "list_directory",
    "description": "List files and directories within a specific path. Use this to understand the project structure, file hierarchy, and permissions.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "The directory path to list (e.g. '.', '/src', 'project_name')."
        },
        "recursive": {
          "type": "boolean",
          "description": "If true, lists files in subdirectories. Keep this false if you are just looking at a root folder to save tokens."
        },
        "max_depth": {
          "type": "integer",
          "description": "Maximum depth for recursive listing (default 3). Keep low to prevent context explosion."
        }
      },
      "required": ["path"]
    }
  },
  {
    "name": "search_files",
    "description": "NOT IMPLEMENTED Find files matching a specific name or extension using a file system search.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The filename or pattern to search for (e.g., '*.log', 'config.py')."
        },
        "path": {
          "type": "string",
          "description": "The directory path to start searching from."
        }
      },
      "required": ["query", "path"]
    }
  },
  {
    "name": "read_file",
    "description": "Read the content of a file.",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {
          "type": "string",
          "description": "The absolute or relative path to the file."
        },
        "limit": {
          "type": "integer",
          "description": "Number of lines to read from end (for debugging). If 0, reads entire file. Keep this low for large files."
        }
      },
      "required": ["file_path"]
    }
  },
  {
    "name": "write_file",
    "description": "Create a new file or overwrite an existing file with the provided content. Read file at least once before overwriting. Only use when creating new file or starting from scratch. Prefer using the edit_file tool to edit existing files",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {
          "type": "string",
          "description": "The path where the file will be written."
        },
        "content": {
          "type": "string",
          "description": "The content to write. Use code blocks or plain text. Do not include markdown backticks unless intended."
        },
        "append": {
          "type": "boolean",
          "description": "If true, append content instead of overwriting."
        }
      },
      "required": ["file_path", "content"]
    }
  },
  {
    "name": "edit_file",
    "description": "Edit an existing file with the provided content. Read file at least once before using edit_file. This tool may fail because old_string is not unique in file. When this happens, you can re-try with a larger chunk of the file to fix the error or use the replace_all parameter if you actually intended to replace multiple occurences.",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {
          "type": "string",
          "description": "The path where the file will be written."
        },
        "old_string": {
          "type": "string",
          "description": "The text to replace"
        },
        "new_string": {
          "type": "string",
          "description": "The text to replace it with (must be different from old_string)"
        },
        "replace_all": {
          "description": "Replace all occurrences of old_string (default false)",
          "type": "boolean"
        },
      },
      "required": ["file_path", "old_string", "new_string"]
    }
  },
  {
    "name": "run_shell",
    "description": "Execute a system shell command (bash). This is for running npm, git, docker, or complex scripts.",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string",
          "description": "The full command string to execute. Example: 'npm install express', 'git log -10'."
        }
      },
      "required": ["command"]
    }
  },
  {
    "name": "search_web",
    "description": "NOT IMPLEMENTED YET. Search the internet for documentation, errors, or API reference info. Use this when the user asks for external knowledge.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query (e.g., 'React useReducer example', 'git push error solution')."
        },
        "engine": {
          "type": "string",
          "description": "Search engine to use (duckduckgo, google, Bing). Default duckduckgo."
        }
      },
      "required": ["query"]
    }
  },
  # {
  #   "name": "grep",
  #   "description": "A powerful search tool built on grep\n\n  Usage:\n  - ALWAYS use grep tool for search tasks. NEVER invoke `grep` as a Bash command. The Grep tool has been optimized for correct permissions and access.\n  - Supports full regex syntax (e.g., \"log.*Error\", \"function\\s+\\w+\")\n  - Filter files with glob parameter (e.g., \"*.js\", \"**/*.tsx\") or type parameter (e.g., \"js\", \"py\", \"rust\")\n  - Output modes: \"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows match counts\n- Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)\n  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`\n",
  #   "parameters": {
  #     "type": "object",
  #     "properties": {
  #       "pattern": {
  #         "description": "The regular expression pattern to search for in file contents",
  #         "type": "string"
  #       },
  #       "path": {
  #         "description": "File or directory to search in (rg PATH). Defaults to current working directory.",
  #         "type": "string"
  #       },
  #       "glob": {
  #         "description": "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob",
  #         "type": "string"
  #       },
  #       "output_mode": {
  #         "description": "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\".",
  #         "type": "string",
  #         "enum": [
  #           "content",
  #           "files_with_matches",
  #           "count"
  #         ]
  #       },
  #       "-B": {
  #         "description": "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise.",
  #         "type": "number"
  #       },
  #       "-A": {
  #         "description": "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise.",
  #         "type": "number"
  #       },
  #       "-C": {
  #         "description": "Alias for context.",
  #         "type": "number"
  #       },
  #       "context": {
  #         "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise.",
  #         "type": "number"
  #       },
  #       "-n": {
  #         "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true.",
  #         "type": "boolean"
  #       },
  #       "-i": {
  #         "description": "Case insensitive search (rg -i)",
  #         "type": "boolean"
  #       },
  #       "-o": {
  #         "description": "Print only the matched (non-empty) parts of each matching line, one match per output line (rg -o / --only-matching). Requires output_mode: \"content\", ignored otherwise. Defaults to false.",
  #         "type": "boolean"
  #       },
  #       "type": {
  #         "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types.",
  #         "type": "string"
  #       },
  #       "head_limit": {
  #         "description": "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 250 when unspecified. Pass 0 for unlimited (use sparingly \u2014 large result sets waste context).",
  #         "type": "number"
  #       },
  #       "offset": {
  #         "description": "Skip first N lines/entries before applying head_limit, equivalent to \"| tail -n +N | head -N\". Works across all output modes. Defaults to 0.",
  #         "type": "number"
  #       },
  #       "multiline": {
  #         "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: False.",
  #         "type": "boolean"
  #       }
  #     },
  #     "required": [
  #       "pattern"
  #     ],
  #   }
  # },    
  # {
  #   "name": "glob",
  #   "description": "Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n- Returns matching file paths sorted by modification time\n- Use this tool when you need to find files by name patterns\n",
  #   "input_schema": {
  #     "type": "object",
  #     "properties": {
  #       "pattern": {
  #         "description": "The glob pattern to match files against",
  #         "type": "string"
  #       },
  #       "path": {
  #         "description": "The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided.",
  #         "type": "string"
  #       }
  #     },
  #     "required": [
  #       "pattern"
  #     ],
  #     "additionalProperties": False
  #   }
  # }
]
