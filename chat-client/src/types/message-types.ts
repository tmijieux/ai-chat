export type Role = 'user' | 'assistant' | 'system' | 'tool'

export type Message = {
  id: string
  conversation_id?: string
  parent_id?: string | null
  role: Role
  content: string
  thinking?: string
  thinking_visible?: boolean
  thinking_included_in_context?: boolean
  loading?: boolean
  token_count?: number | null
  created_at?: string
}

export type ApiDone =
  | { done: false }
  | {
      done: true
      done_reason: string
      total_duration: number
      load_duration: number
      prompt_eval_count: number
      prompt_eval_duration: number
      eval_count: number
      eval_duration: number
    }

export type ApiResponse = {
  model: string
  created_at: string
  message: Message
} & ApiDone

export type ConversationHistory = Message[]

export type ConversationSettings = {
  active_prompt_ids: string[]
  active_tool_names: string[]
  tools_enabled: boolean
  agentic_mode: boolean
}

export type Conversation = {
  id: string
  title: string
  created_at: string
  active_message_id: string | null
  settings: string | null  // JSON-encoded ConversationSettings
  history?: ConversationHistory
}

export type SystemPromptCategory =
  | 'general'
  | 'code'
  | 'summarization'
  | 'context_compaction'
  | 'state_storage'

export type SystemPromptTemplate = {
  id: string
  name: string
  category: SystemPromptCategory
  content: string
  is_global: number  // 1 | 0
  created_at: string
}

/** Flat node returned by GET /api/conversations/{id}/tree */
export type MessageTreeNode = {
  id: string
  parent_id: string | null
  role: Role
  content_preview: string
  created_at: string
  sibling_count: number
}

export type ConversationTree = {
  active_message_id: string | null
  nodes: MessageTreeNode[]
}

// ---------------------------------------------------------------------------
// Agent WebSocket event types
// ---------------------------------------------------------------------------

export type AgentEventType =
  | 'thinking'
  | 'content'
  | 'tool_call'
  | 'tool_confirm'
  | 'tool_result'
  | 'iteration_end'
  | 'done'
  | 'error'

export type AgentEvent = {
  type: AgentEventType
  // content/thinking chunks
  content?: string
  // tool events
  tool_id?: string
  tool_name?: string
  arguments?: Record<string, unknown>
  preview?: string
  // tool_result
  // iteration_end
  prompt_tokens?: number
  response_tokens?: number
  // error
  message?: string
}

/** A message-like object used only in the UI for agent tool interactions. */
export type AgentUiMessage = {
  id: string
  ui_type: 'user' | 'thinking' | 'content' | 'tool_confirm' | 'tool_result' | 'iteration_end'
  done?: boolean
  tool_id?: string
  tool_name?: string
  tool_args?: Record<string, unknown>
  preview?: string
  content: string
  confirmed?: boolean | null
  prompt_tokens?: number
}
