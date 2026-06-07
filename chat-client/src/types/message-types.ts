export type Role = 'user' | 'assistant' | 'system' | 'tool'

export type AppStatus = { llm: boolean; whisper: boolean }

export type ImageAttachment = { id: string; mime_type: string }

export type PendingImage = {
  localUrl: string
  uploading: boolean
  id?: string
  mime_type?: string
}

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
  token_delta?: number | null
  context_excluded?: boolean
  exclusion_reason?: string | null
  compressed_summary?: string | null
  log_message?: string | null
  created_at?: string
  sibling_count?: number
  sibling_index?: number
  prev_sibling_id?: string | null
  next_sibling_id?: string | null
  has_children?: boolean
  images?: ImageAttachment[]
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
  is_default: boolean
  token_count: number | null
  created_at: string
}

export type AppSetting = {
  key: string
  value: string | null
}

export type AgentToolMeta = {
  name: string
  description: string
  requires_confirmation: boolean
  token_count: number
}

export type AgentToolsResponse = {
  framework_overhead: number
  stacking_overhead_per_additional_tool: number
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

export type DiffLine = {
  type: 'added' | 'removed' | 'context' | 'header'
  text: string
  line?: number | null
}

export type AgentEvent = {
  type: AgentEventType
  content?: string
  tool_id?: string
  tool_name?: string
  arguments?: Record<string, unknown>
  preview?: string
  diff_lines?: DiffLine[]
  prompt_tokens?: number
  response_tokens?: number
  message?: string
  log_message?: string
}

// ---------------------------------------------------------------------------
// Unified display message — single type rendered in the template.
// Sources: DB reload (via selectConversation) and live agent event stream.
// ---------------------------------------------------------------------------

/** Token metadata added to messages that carry token information. */
export type TokenMeta = {
  token_count: number | null
  /** Estimated tokens for this message alone. */
  token_delta: number | null
  /** token_count as a percentage of the 16 384 context window. */
  token_pct: number | null
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
      images?: ImageAttachment[]
      token_count?: number | null
      token_delta?: number | null
      context_excluded?: boolean
    } & SiblingMeta)
  | ({
      kind: 'assistant'
      id: string
      content: string
      thinking?: string
      /** True while the HTTP stream is still open (non-agentic mode). */
      streaming?: boolean
      token_count?: number | null
      token_delta?: number | null
      context_excluded?: boolean
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
      diff_lines?: DiffLine[]
      /** null = awaiting response, true/false = confirmed/rejected */
      confirmed: boolean | null
    } & SiblingMeta)
  | ({
      kind: 'tool_result'
      id: string
      tool_name: string
      log_message?: string | null
      content: string
      compressed_summary?: string | null
      token_count?: number | null
      token_delta?: number | null
      context_excluded?: boolean
    } & SiblingMeta)
  | { kind: 'error'; id: string; message: string }

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
