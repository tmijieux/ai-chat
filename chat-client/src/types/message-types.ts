export type Role = 'user' | 'assistant' | 'system' | 'tool'

export type MessageForQuery = {
  role: Role
  content: string
}

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
  sibling_count?: number
  sibling_index?: number
  prev_sibling_id?: string | null
  next_sibling_id?: string | null
  has_children?: boolean
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
  active_prompt_id: string | null
  active_tool_names: string[]
  agentic_mode: boolean
  working_directory: string | null
}

export type Conversation = {
  id: string
  title: string
  created_at: string
  active_message_id: string | null
  settings: string | null // JSON-encoded ConversationSettings
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
  is_default: number // 1 | 0
  token_count: number | null
  created_at: string
}

export type AgentToolMeta = {
  name: string
  description: string
  requires_confirmation: boolean
  token_count: number
}

export type AgentToolsResponse = {
  framework_overhead: number
  tools: AgentToolMeta[]
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
  content?: string
  tool_id?: string
  tool_name?: string
  arguments?: Record<string, unknown>
  preview?: string
  prompt_tokens?: number
  response_tokens?: number
  message?: string
}

// ---------------------------------------------------------------------------
// Unified display message — single type rendered in the template.
// Sources: DB reload (via selectConversation) and live agent event stream.
// ---------------------------------------------------------------------------

/** Token metadata added to messages that carry a cumulative context size. */
export type TokenMeta = {
  token_count: number
  /** Tokens added by this message relative to the previous counted message. */
  token_contribution: number | null
  /** token_count as a percentage of the 16 384 context window. */
  token_pct: number
}

type SiblingMeta = {
  sibling_count?: number
  sibling_index?: number
  prev_sibling_id?: string | null
  next_sibling_id?: string | null
  has_children?: boolean
}

export type DisplayMessage =
  | ({
      kind: 'user'
      id: string
      content: string
      token_count?: number | null
    } & SiblingMeta)
  | ({
      kind: 'assistant'
      id: string
      content: string
      thinking?: string
      /** True while the HTTP stream is still open (non-agentic mode). */
      streaming?: boolean
      token_count?: number | null
    } & SiblingMeta)
  | ({
      kind: 'thinking'
      id: string
      content: string
      /** False while still streaming; true once the block is complete. */
      done: boolean
    } & SiblingMeta)
  | ({
      kind: 'tool_confirm'
      id: string
      tool_id: string
      tool_name: string
      args: Record<string, unknown>
      preview: string
      /** null = awaiting response, true/false = confirmed/rejected */
      confirmed: boolean | null
    } & SiblingMeta)
  | ({
      kind: 'tool_result'
      id: string
      tool_name: string
      content: string
      token_count?: number | null
    } & SiblingMeta)

/**
 * DisplayMessage enriched with token contribution metadata and sibling navigation.
 * Computed in the component from the raw DisplayMessage array — not stored in the signal.
 */
export type DisplayMessageWithMeta = DisplayMessage & {
  token_meta?: TokenMeta
  sibling_count?: number
  sibling_index?: number
  prev_sibling_id?: string | null
  next_sibling_id?: string | null
  has_children?: boolean
}
