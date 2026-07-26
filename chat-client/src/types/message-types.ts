export type Role = 'user' | 'assistant' | 'system' | 'tool' | 'context_summary'

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
  compression_label?: string | null
  compressed_token_count?: number | null
  log_message?: string | null
  tool_calls?: ToolCallEntry[] | null
  created_at?: string
  sibling_count?: number
  sibling_index?: number
  prev_sibling_id?: string | null
  next_sibling_id?: string | null
  has_children?: boolean
  images?: ImageAttachment[]
  is_degenerate?: boolean
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

export type ConversationMode = 'standard' | 'plan' | 'auto' | 'yolo'

export type ConversationSettings = {
  active_prompt_id: string | null
  active_tool_names: string[]
  working_directory: string | null
  mode: ConversationMode
}

export type Workflow = {
  name: string
  description: string
}

export type FileSearchResult = {
  name: string
  path: string
  relative_path: string
}

export type SlashCommand =
  | { type: 'mode'; value: ConversationMode; label: string; description: string }
  | { type: 'workflow'; value: string; label: string; description: string }

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
}

export type AgentDefinition = {
  name: string
  description: string
  system_prompt: string
  tools: string[]
  finish_tool: string
  max_iterations: number | null
  inject_turn_reminders: boolean
}

export type AppSetting = {
  key: string
  value: string | null
}

export type ContextEntry = {
  role: string
  token_count: number
  content: string
  tool_name?: string | null
  status?: string | null
  image_count: number
}

export type AgentToolMeta = {
  name: string
  description: string
  requires_confirmation: boolean
  token_count: number
}

export type AlwaysActiveToolMeta = {
  name: string
  description: string
  token_count: number
  mode_context: string
}

export type AgentToolsResponse = {
  framework_overhead: number
  stacking_overhead_per_additional_tool: number
  tools: AgentToolMeta[]
  always_active_tools: AlwaysActiveToolMeta[]
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
// Workflow run view
// ---------------------------------------------------------------------------

export type WorkflowStageType = 'llm' | 'coordinator' | 'branch' | 'loop' | 'respond' | 'agent'

/** Static structure of one workflow stage, sent once up front so the plan can be drawn before it runs. */
export type WorkflowNode = {
  /** Stable dotted identity, e.g. 'translate_loop.translate_chunk'. Matches stage_enter.path. */
  path: string
  name: string
  type: WorkflowStageType
  finish_tool: string | null
  tools: string[]
  over: string | null
  attempt_total: number | null
  children: WorkflowNode[]
}

export type WorkflowStageStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped'

/** One piece of activity produced inside a stage, in arrival order. */
export type WorkflowActivityEntry =
  | { kind: 'thinking'; text: string }
  | { kind: 'content'; text: string }
  | { kind: 'tool_call'; tool_id: string; tool_name: string; args_text: string }
  | { kind: 'tool_result'; tool_id: string; tool_name: string; log_message: string | null; content: string }
  | { kind: 'error'; message: string }

/** Runtime state of a single stage invocation. Keyed by execution_id. */
export type WorkflowStageState = {
  path: string
  execution_id: string
  status: WorkflowStageStatus
  stage_type: WorkflowStageType
  invocation_number: number
  item_number: number | null
  item_total: number | null
  attempt_number: number | null
  attempt_total: number | null
  iteration_count: number
  /** Last measured prompt size for this stage's isolated sub-context. Never estimated. */
  context_tokens: number | null
  /** Sum of generated tokens across this stage's iterations. */
  response_tokens: number
  result: unknown
  duration_ms: number | null
  activity: WorkflowActivityEntry[]
  /** True once activity was released by the ring buffer; status and result are still valid. */
  detail_dropped: boolean
}

export type WorkflowLoopItemState = {
  item_number: number
  status: 'pending' | 'running' | 'done' | 'failed'
  attempts_used: number
}

export type WorkflowRun = {
  workflow_name: string
  status: 'running' | 'done' | 'error'
  started_at: number
  /** Set when the run ends, so the elapsed display freezes instead of counting forever. */
  finished_at: number | null
  nodes: WorkflowNode[]
  /** Every invocation, by execution_id — the single source of truth for stage state. */
  execution_by_id: Record<string, WorkflowStageState>
  /** Which invocation a stage row should display: static path → latest execution_id. */
  latest_execution_by_path: Record<string, string>
  /** Per-loop item outcomes, by the loop stage's static path. Drives the dot grid. */
  loop_items: Record<string, WorkflowLoopItemState[]>
  current_execution_id: string | null
  /** Tokens the whole run has put through the model — prompt + generated, summed over every iteration. */
  total_tokens_processed: number
}

// ---------------------------------------------------------------------------
// Agent WebSocket event types
// ---------------------------------------------------------------------------

export type DiffLine = {
  type: 'added' | 'removed' | 'context' | 'header'
  text: string
  line?: number | null
}

export type ToolCallEntry = { id: string; name: string; args: Record<string, unknown> }

export type AgentEvent = (
  | { type: 'thinking' | 'content'; content: string }
  | { type: 'tool_call_start'; tool_id: string; tool_name: string }
  | { type: 'tool_call_chunk'; tool_id: string; chunk: string }
  | { type: 'tool_call'; tool_id: string; tool_name: string; arguments: Record<string, unknown> }
  | { type: 'tool_confirm'; tool_id: string; tool_name: string; arguments: Record<string, unknown>; preview: string; diff_lines?: DiffLine[]; evaluator_reason?: string }
  | { type: 'tool_evaluating'; tool_id: string; tool_name: string }
  | { type: 'tool_auto_approved'; tool_id: string; reason?: string }
  | { type: 'tool_result'; tool_id: string; tool_name: string; content: string; log_message?: string; ctx_tokens?: number }
  | { type: 'generation_end'; ctx_tokens: number }
  | { type: 'iteration_end'; prompt_tokens: number; response_tokens: number }
  | { type: 'ctx_update' | 'compressing'; ctx_tokens: number }
  | { type: 'context'; ctx_tokens: number; messages: unknown[] }
  | { type: 'plan_proposal'; plan_id: string; plan: string }
  | { type: 'agent_question'; question_id: string; question: string; options?: string[] }
  | { type: 'mode_changed'; mode: ConversationMode }
  | { type: 'workflow_start'; workflow_name: string; nodes: WorkflowNode[] }
  | {
      type: 'stage_enter'
      path: string
      execution_id: string
      stage_type: WorkflowStageType
      invocation_number: number
      item_number: number | null
      item_total: number | null
      attempt_number: number | null
      attempt_total: number | null
    }
  | {
      type: 'stage_exit'
      path: string
      execution_id: string
      status: 'done' | 'failed' | 'skipped'
      result: unknown
      duration_ms: number
    }
  | {
      type: 'loop_item_exit'
      path: string
      item_number: number
      item_total: number
      success: boolean
      attempts_used: number
    }
  | { type: 'done'; finished_without_response?: boolean }
  | { type: 'error'; message: string }
) & { _pipeline_stage?: string; _workflow_execution?: string }

export type AgentEventType = AgentEvent['type']

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
      tool_calls?: ToolCallEntry[] | null
      /** True while the HTTP stream is still open (non-agentic or agentic mode). */
      streaming?: boolean
      /** True when the agent stopped without producing content or tool calls. */
      is_degenerate?: boolean
      token_count?: number | null
      token_delta?: number | null
      context_excluded?: boolean
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
      compression_label?: string | null
      compressed_token_count?: number | null
      token_count?: number | null
      token_delta?: number | null
      context_excluded?: boolean
    } & SiblingMeta)
  | { kind: 'tool_evaluating'; id: string; tool_id: string; tool_name: string; verdict?: 'safe' | 'dangerous'; reason?: string }
  | { kind: 'plan_proposal'; id: string; plan_id: string; plan: string; resolved: boolean; resolution?: string }
  | { kind: 'agent_question'; id: string; question_id: string; question: string; options?: string[]; resolved: boolean }
  | { kind: 'error'; id: string; message: string }
  | ({
      kind: 'context_summary'
      id: string
      content: string
      context_excluded?: boolean
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
