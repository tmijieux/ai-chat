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
  /** Set on the user message that started a workflow run, once the engine's run_id is known
   * (see ChatService's 'workflow_start' handling) — lets the run view be reopened after a
   * reload. Both null/undefined for a message that isn't a workflow invocation. */
  workflow_name?: string | null
  workflow_run_id?: string | null
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

/** 'script' is a coordinator stage whose action is run_script, relabeled at the backend's emission
 * boundary (see custom_workflow.py's _display_stage_type) — not a distinct dispatch type. */
export type WorkflowStageType = 'llm' | 'coordinator' | 'script' | 'branch' | 'loop' | 'respond' | 'agent'

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

export type WorkflowStageStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped' | 'stopped'

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
  status: 'pending' | 'running' | 'done' | 'failed' | 'stopped'
  attempts_used: number
  /**
   * What this item actually produced: {item: <the loop variable>, success, <inner stage name>:
   * <its result>, ...} for every inner stage that ran. Null while the item is still pending/running
   * — filled in atomically on loop_item_exit, once, unambiguous by construction (no reconstruction
   * from the separate stage_enter/stage_exit stream needed).
   */
  result: unknown | null
}

/**
 * What the detail pane is currently showing — either one stage invocation's live/frozen activity
 * feed (row click, or following the running stage), or one loop item's own frozen result (dot
 * click). Kept as a discriminated union rather than shoehorning both into WorkflowStageState,
 * since a loop item spans several stage invocations and isn't itself one.
 */
export type WorkflowSelectedDetail =
  | { kind: 'execution'; state: WorkflowStageState }
  | { kind: 'loop_item'; path: string; item: WorkflowLoopItemState }

export type WorkflowRun = {
  workflow_name: string
  /** Identifies this run's persisted directory on disk (see GET /api/workflow-runs) — ADR-0011. */
  run_id: string
  status: 'running' | 'done' | 'error' | 'stopped'
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

/** One child in a fetched node's shallow children list — status only, not its own content. */
export type WorkflowRunNodeChild = {
  segment: string
  /** 'loop_item' for a loop-item-number directory (e.g. "11"), which has no stage type of its own. */
  stage_type: WorkflowStageType | 'loop_item' | null
  status: WorkflowStageStatus | null
  /** Set for a 'loop_item' child — the loop's real item count, known from the moment any item
   * dispatches. Lets a run reopened from disk draw every pending dot up front (see
   * workflow-run-panel.component.ts's loopItemsFor), not just the ones that ran before a stop or
   * failure cut the loop short. Undefined/null for a plain stage child. */
  item_total?: number | null
}

/** One persisted attempt's full transcript, as returned by GET /api/workflow-runs/.../node. */
export type WorkflowRunNodeAttempt = {
  execution_id: string
  invocation_number: number
  status: WorkflowStageStatus
  stage_type: WorkflowStageType
  item_number: number | null
  item_total: number | null
  attempt_number: number | null
  attempt_total: number | null
  iteration_count: number
  context_tokens: number | null
  response_tokens: number
  result: unknown
  duration_ms: number | null
  activity: WorkflowActivityEntry[]
}

/**
 * Response from GET /api/workflow-runs/{workflow_name}/{run_id}/node?path=... — one node's own
 * persisted content plus a shallow (one level) list of its direct children, for the run view's
 * click-to-drill-down navigation (ADR-0011). `path` is the on-disk address used to fetch it
 * (segments joined by "/") — distinct from the engine's dotted stage path, since it interleaves
 * loop item numbers.
 */
export type WorkflowRunNode = {
  path: string
  meta: { path: string; stage_type: WorkflowStageType; status: WorkflowStageStatus; attempt_count: number; attempt_total: number | null } | null
  attempts: WorkflowRunNodeAttempt[]
  item_result: { item_number: number; item_total: number; success: boolean; attempts_used: number; item_result: unknown } | null
  children: WorkflowRunNodeChild[]
}

/** Response from GET /api/workflow-runs/{workflow_name}/{run_id} — the run's own top-level
 * status, as WorkflowRunRecorder._write_run_meta persists it into run.json. */
export type WorkflowRunStatusResponse = {
  run_id: string
  status: 'running' | 'done' | 'failed' | 'stopped'
  user_message: string
  failed_path?: string
  /** The declared stage tree, persisted alongside status (see WorkflowRunRecorder._nodes) so a
   * run reopened from disk can draw the whole plan up front, same as the live view does. */
  nodes: WorkflowNode[]
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
  | { type: 'tool_call_raw'; fragment: string }
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
  | { type: 'workflow_start'; workflow_name: string; run_id: string; nodes: WorkflowNode[] }
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
      status: 'done' | 'failed' | 'skipped' | 'stopped'
      result: unknown
      duration_ms: number
    }
  | {
      type: 'loop_item_exit'
      path: string
      item_number: number
      item_total: number
      success: boolean
      status: 'done' | 'failed' | 'stopped'
      attempts_used: number
      item_result: unknown
    }
  | { type: 'done'; finished_without_response?: boolean }
  | { type: 'error'; message: string }
  | { type: 'stopped' }
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
      /** Set once the workflow this message started has an engine run_id — drives the "Reopen
       * workflow run" message action, which stays hidden until both are non-null. */
      workflow_name?: string | null
      workflow_run_id?: string | null
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

export type TokenVisualizerPiece = {
  id: number
  piece: string
  special: boolean
}

export type TokenVisualizerStreamEvent =
  | { type: 'system_tokens'; tokens: TokenVisualizerPiece[] }
  | { type: 'user_tokens'; tokens: TokenVisualizerPiece[] }
  | { type: 'assistant_preamble_tokens'; tokens: TokenVisualizerPiece[] }
  | { type: 'assistant_token'; id: number; piece: string; special: boolean }
  | { type: 'done' }

export type TokenVisualizerToolCall = {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

/** Raw OpenAI-shaped message sent to the backend to be rendered through llama.cpp's own
 * chat template — never interpreted locally, just passed straight through. */
export type TokenVisualizerHistoryMessage =
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string | null; tool_calls?: TokenVisualizerToolCall[] }
  | { role: 'tool'; tool_call_id: string; name?: string; content: string }

export type TokenVisualizerMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  displayLabel: string
  /** The exact message sent/to-send as history for later turns. Null for 'system', since the
   * system prompt is carried on every request via its own field, not as a history entry. */
  historyMessage: TokenVisualizerHistoryMessage | null
  /** Accumulated generated text, used only for 'assistant' messages produced by real generation. */
  content: string
  tokens: TokenVisualizerPiece[]
}
