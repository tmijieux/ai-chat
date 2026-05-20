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
          "description": "If true, lists files in subdirectories. Keep false when just looking at a root folder to save tokens."
        },
        "maximum_depth": {
          "type": "integer",
          "description": "Maximum depth for recursive listing (default 3). Keep low to prevent context explosion."
        }
      },
      "required": ["path"]
    }
  },
  {
    "name": "glob_files",
    "description": "Find files matching a glob pattern (e.g. '**/*.ts', 'src/**/*.py'). Use this to locate files by name or extension before reading them.",
    "parameters": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "description": "Glob pattern (e.g. '**/*.ts', 'src/**/*.py', 'tests/test_*.py')."
        },
        "path": {
          "type": "string",
          "description": "Root directory to search from (default '.')."
        }
      },
      "required": ["pattern"]
    }
  },
  {
    "name": "grep_files",
    "description": "Search file contents with a regex pattern. Returns matching lines with file path and line numbers. Prefer this over reading whole files when looking for a symbol, function, or string.",
    "parameters": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "description": "Regex pattern to search for (e.g. 'def my_func', 'import.*react', 'TODO')."
        },
        "path": {
          "type": "string",
          "description": "Root directory to search from (default '.')."
        },
        "glob": {
          "type": "string",
          "description": "Glob filter for which files to search (e.g. '**/*.py', '**/*.ts'). Default '**/*'."
        },
        "case_insensitive": {
          "type": "boolean",
          "description": "If true, match case-insensitively (default false)."
        },
        "max_matches": {
          "type": "integer",
          "description": "Maximum matches to return (default 50). Increase only if you need more."
        }
      },
      "required": ["pattern"]
    }
  },
  {
    "name": "read_file",
    "description": "Read the full content of a file. For large files, use the limit parameter to read only the last N lines.",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {
          "type": "string",
          "description": "The absolute or relative path to the file."
        },
        "limit": {
          "type": "integer",
          "description": "Number of lines to read from the END of the file (useful for logs). 0 or omit to read the entire file."
        }
      },
      "required": ["file_path"]
    }
  },
  {
    "name": "write_file",
    "description": "Create a new file or overwrite an existing file. Requires user confirmation. Only use when creating a brand-new file or doing a full rewrite. Prefer edit_file to make targeted edits.",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {
          "type": "string",
          "description": "The path where the file will be written."
        },
        "content": {
          "type": "string",
          "description": "The content to write."
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
    "description": "Replace a specific string in a file with a new string. Requires user confirmation. Read the file first. Will fail if old_string is not found or not unique — use replace_all for multiple occurrences or widen the context string to make it unique.",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {
          "type": "string",
          "description": "Path to the file to edit."
        },
        "old_string": {
          "type": "string",
          "description": "The exact text to replace. Must be unique in the file unless replace_all is true."
        },
        "new_string": {
          "type": "string",
          "description": "The text to replace it with."
        },
        "replace_all": {
          "type": "boolean",
          "description": "Replace all occurrences of old_string (default false)."
        }
      },
      "required": ["file_path", "old_string", "new_string"]
    }
  },
  {
    "name": "run_shell",
    "description": "Execute a shell command (bash). For running npm, git, python scripts, etc. Requires user confirmation.",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string",
          "description": "The full command string to execute (e.g. 'npm install', 'git log -10')."
        }
      },
      "required": ["command"]
    }
  },
  {
    "name": "search_web",
    "description": "Search DuckDuckGo and extract page content. Use for external documentation, error messages, or API references.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query."
        }
      },
      "required": ["query"]
    }
  },
  {
    "name": "summarize_subtask",
    "description": "Summarize a large piece of content relative to a specific task using a fresh LLM call. Use this to compress large tool outputs (file reads, search results) that would overflow context.",
    "parameters": {
      "type": "object",
      "properties": {
        "task": {
          "type": "string",
          "description": "What you need to know from the content (guides what to keep)."
        },
        "content": {
          "type": "string",
          "description": "The large content to summarize."
        }
      },
      "required": ["task", "content"]
    }
  }
]
