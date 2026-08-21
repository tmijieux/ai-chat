import { Injectable, computed, inject, signal } from '@angular/core'
import { firstValueFrom } from 'rxjs'
import { AgentService } from './agent.service'
import { ApiService } from './api.service'
import {
  AgentEvent,
  WorkflowActivityEntry,
  WorkflowLoopItemState,
  WorkflowRun,
  WorkflowRunNode,
  WorkflowSelectedDetail,
  WorkflowStageState,
} from '../types/message-types'

/**
 * Number of stage invocations that keep their full activity log. Older ones are released and
 * marked detail_dropped, so a loop over hundreds of items cannot grow memory without bound.
 */
const ACTIVITY_RETAINED_EXECUTIONS = 11

/**
 * Live view of the running workflow, assembled from the agent event stream.
 *
 * Subscribes to AgentService.events$ once and keeps that subscription for the app's lifetime —
 * events$ is a long-lived root Subject, so this catches every run without the per-run
 * subscribe/unsubscribe dance ChatService needs for its message accumulation.
 *
 * State is live only: nothing is persisted, and a page refresh loses the view while the run
 * continues headless on the backend.
 */
@Injectable({ providedIn: 'root' })
export class WorkflowRunService {
  private agentSvc = inject(AgentService)
  private apiSvc = inject(ApiService)

  private readonly _run = signal<WorkflowRun | null>(null)
  readonly run = this._run.asReadonly()

  /** True when the run view replaces the message list. Set automatically, toggleable by the user. */
  readonly viewOpen = signal(false)

  /** Which invocation the detail pane shows when no loop item is selected. Null follows the running stage. */
  readonly selectedExecutionId = signal<string | null>(null)

  /** Which loop item the detail pane shows — a dot click. Takes priority over selectedExecutionId. */
  readonly selectedLoopItem = signal<{ path: string; itemNumber: number } | null>(null)

  /**
   * On-disk address of a finished node currently selected for the detail pane (see ADR-0011) —
   * null means the pane shows live/frozen in-memory state (selectedExecutionId/selectedLoopItem)
   * instead. Takes priority over both when set.
   */
  readonly selectedFetchedAddress = signal<string[] | null>(null)

  /** Persisted nodes fetched from disk, keyed by "/"-joined address. The top stage list IS the
   * explorer (per user feedback — no separate children list in the detail pane): expanding a
   * finished node fetches its content here and splices its children in as rows nested beneath it. */
  private readonly _fetchedNodes = signal<Map<string, WorkflowRunNode>>(new Map())
  readonly fetchedNodes = this._fetchedNodes.asReadonly()

  /** Addresses currently expanded in the stage list — a row whose key is here has its fetched
   * children spliced in as rows nested beneath it. */
  private readonly _expandedAddresses = signal<Set<string>>(new Set())
  readonly expandedAddresses = this._expandedAddresses.asReadonly()

  /** Addresses with a fetch currently in flight, for a small loading indicator on that row. */
  private readonly _pendingAddresses = signal<Set<string>>(new Set())
  readonly pendingAddresses = this._pendingAddresses.asReadonly()

  /** Execution ids that still hold activity, oldest first — the ring buffer's order. */
  private activityOrder: string[] = []

  /**
   * What the detail pane should render right now. A selected loop item always wins — it shows
   * that item's own frozen result (see WorkflowLoopItemState.result), never whatever else happens
   * to be streaming elsewhere in the run. Otherwise falls back to the explicitly selected stage
   * invocation, or the currently running one in follow mode (both selections null).
   */
  readonly selectedDetail = computed<WorkflowSelectedDetail | null>(() => {
    const run = this._run()
    if (run === null) {
      return null
    }
    const loopSelection = this.selectedLoopItem()
    if (loopSelection !== null) {
      const items = run.loop_items[loopSelection.path]
      const item = items?.find((candidate) => candidate.item_number === loopSelection.itemNumber) ?? null
      return item === null ? null : { kind: 'loop_item', path: loopSelection.path, item }
    }
    const explicitId = this.selectedExecutionId()
    const id = explicitId !== null ? explicitId : run.current_execution_id
    if (id === null) {
      return null
    }
    const state = run.execution_by_id[id] ?? null
    return state === null ? null : { kind: 'execution', state }
  })

  constructor() {
    this.agentSvc.events$.subscribe((event) => {
      this._handle(event)
    })
  }

  /** Fetch (once — cached) and return a persisted node by address. */
  private async ensureFetched(address: string[]): Promise<WorkflowRunNode | null> {
    const run = this._run()
    if (run === null) {
      return null
    }
    const key = address.join('/')
    const cached = this._fetchedNodes().get(key)
    if (cached !== undefined) {
      return cached
    }
    this._pendingAddresses.update((set) => new Set(set).add(key))
    try {
      const node = await firstValueFrom(
        this.apiSvc.get_workflow_run_node(run.workflow_name, run.run_id, key),
      )
      this._fetchedNodes.update((map) => new Map(map).set(key, node))
      return node
    } finally {
      this._pendingAddresses.update((set) => {
        const next = new Set(set)
        next.delete(key)
        return next
      })
    }
  }

  getFetchedNode(address: string[]): WorkflowRunNode | null {
    return this._fetchedNodes().get(address.join('/')) ?? null
  }

  isExpanded(address: string[]): boolean {
    return this._expandedAddresses().has(address.join('/'))
  }

  isPending(address: string[]): boolean {
    return this._pendingAddresses().has(address.join('/'))
  }

  /** Select a finished node for the detail pane — fetches it if not already cached. */
  selectFetched(address: string[]): void {
    this.selectedExecutionId.set(null)
    this.selectedLoopItem.set(null)
    this.selectedFetchedAddress.set(address)
    this._pruneUnrelated(address)
    void this.ensureFetched(address)
  }

  /**
   * Only one branch of the tree is ever expanded at a time — not just among a single loop's own
   * items, but tree-wide: selecting a plain top-level stage collapses whatever loop item was open
   * elsewhere just as much as selecting a sibling item does. `address`'s own ancestors (so the
   * branch you're drilling into doesn't collapse itself) and descendants (so re-selecting a node
   * whose own child you'd already drilled into keeps that child visible) are kept; everything else
   * goes.
   */
  private _pruneUnrelated(address: string[]): void {
    const key = address.join('/')
    this._expandedAddresses.update((set) => {
      const next = new Set<string>()
      for (const existing of set) {
        if (existing === key || existing.startsWith(`${key}/`) || key.startsWith(`${existing}/`)) {
          next.add(existing)
        }
      }
      return next
    })
  }

  /**
   * Toggle whether a node's children are spliced into the stage list beneath it. Expanding one
   * implicitly collapses whatever unrelated branch was open elsewhere (see _pruneUnrelated), so
   * there's never ambiguity about which stages belong to which item, or a stale branch left open
   * somewhere the user has since navigated away from.
   */
  toggleExpand(address: string[]): void {
    const key = address.join('/')
    if (this._expandedAddresses().has(key)) {
      this._expandedAddresses.update((set) => {
        const next = new Set<string>()
        for (const existing of set) {
          if (existing !== key && !existing.startsWith(`${key}/`)) {
            next.add(existing)
          }
        }
        return next
      })
      return
    }
    this._pruneUnrelated(address)
    this._expandedAddresses.update((set) => new Set(set).add(key))
    void this.ensureFetched(address)
  }

  /**
   * Collapse whatever's expanded elsewhere in the tree relative to `address`, without expanding
   * `address` itself — for selecting something that has nothing of its own to expand yet (a
   * pending/still-running item, or a plain leaf stage), so a stale branch doesn't linger once
   * you've moved on.
   */
  collapseSiblings(address: string[]): void {
    this._pruneUnrelated(address)
  }

  /** Leave the fetched selection and go back to following the live run. */
  followRunning(): void {
    this.selectedFetchedAddress.set(null)
    this.selectedLoopItem.set(null)
    this.selectedExecutionId.set(null)
  }

  private _handle(event: AgentEvent): void {
    if (event.type === 'workflow_start') {
      this.activityOrder = []
      this.selectedExecutionId.set(null)
      this.selectedLoopItem.set(null)
      this.selectedFetchedAddress.set(null)
      this._fetchedNodes.set(new Map())
      this._expandedAddresses.set(new Set())
      this._pendingAddresses.set(new Set())
      this._run.set({
        workflow_name: event.workflow_name,
        run_id: event.run_id,
        status: 'running',
        started_at: Date.now(),
        finished_at: null,
        nodes: event.nodes,
        execution_by_id: {},
        latest_execution_by_path: {},
        loop_items: {},
        current_execution_id: null,
        total_tokens_processed: 0,
      })
      this.viewOpen.set(true)
      return
    }

    if (this._run() === null) {
      return
    }

    // Anything that blocks on the user is rendered in the message list, which the run view is
    // covering — so hand the view back rather than leaving them staring at a stalled workflow
    // with the prompt hidden behind it. The chip brings the run view back.
    if (event.type === 'tool_confirm' || event.type === 'agent_question' || event.type === 'plan_proposal') {
      this.viewOpen.set(false)
    }

    if (event.type === 'stage_enter') {
      this._onStageEnter(event)
      return
    }
    if (event.type === 'stage_exit') {
      this._onStageExit(event)
      return
    }
    if (event.type === 'loop_item_exit') {
      this._onLoopItemExit(event)
      return
    }
    if ((event.type === 'done' || event.type === 'error' || event.type === 'stopped') && event._pipeline_stage === undefined) {
      const status = event.type === 'done' ? 'done' : event.type === 'stopped' ? 'stopped' : 'error'
      this._run.update((run) =>
        run === null
          ? null
          : {
              ...run,
              status,
              finished_at: Date.now(),
              current_execution_id: null,
            },
      )
      return
    }
    if (event._workflow_execution !== undefined) {
      this._onStageActivity(event, event._workflow_execution)
    }
  }

  private _onStageEnter(event: Extract<AgentEvent, { type: 'stage_enter' }>): void {
    const state: WorkflowStageState = {
      path: event.path,
      execution_id: event.execution_id,
      status: 'running',
      stage_type: event.stage_type,
      invocation_number: event.invocation_number,
      item_number: event.item_number,
      item_total: event.item_total,
      attempt_number: event.attempt_number,
      attempt_total: event.attempt_total,
      iteration_count: 0,
      context_tokens: null,
      response_tokens: 0,
      result: null,
      duration_ms: null,
      activity: [],
      detail_dropped: false,
    }
    this._run.update((run) => {
      if (run === null) {
        return null
      }
      const loopItems = this._markLoopItemRunning(run, event.path, event.item_number, event.item_total)
      return {
        ...run,
        execution_by_id: { ...run.execution_by_id, [event.execution_id]: state },
        latest_execution_by_path: { ...run.latest_execution_by_path, [event.path]: event.execution_id },
        loop_items: loopItems,
        current_execution_id: event.execution_id,
      }
    })
    this._retainActivityFor(event.execution_id)
    // A respond stage streams into the conversation itself, so hand the view back to the chat.
    if (event.stage_type === 'respond') {
      this.viewOpen.set(false)
    }
  }

  private _onStageExit(event: Extract<AgentEvent, { type: 'stage_exit' }>): void {
    this._patchExecution(event.execution_id, (state) => ({
      ...state,
      status: event.status,
      result: event.result,
      duration_ms: event.duration_ms,
    }))
    this._run.update((run) =>
      run === null || run.current_execution_id !== event.execution_id
        ? run
        : { ...run, current_execution_id: null },
    )
  }

  private _onLoopItemExit(event: Extract<AgentEvent, { type: 'loop_item_exit' }>): void {
    this._run.update((run) => {
      if (run === null) {
        return null
      }
      const items = this._ensureLoopItems(run, event.path, event.item_total)
      const updated = items.map((item) =>
        item.item_number === event.item_number
          ? {
              item_number: item.item_number,
              status: event.status,
              attempts_used: event.attempts_used,
              result: event.item_result,
            }
          : item,
      )
      return { ...run, loop_items: { ...run.loop_items, [event.path]: updated } }
    })
  }

  /**
   * File a tagged event under the stage invocation that produced it.
   *
   * Text events coalesce into the trailing entry of the same kind so a streamed thinking block
   * stays one block. Tool calls arrive as a start plus argument chunks — the backend never emits
   * a single complete tool_call event — so the entry is opened on start and filled in as chunks
   * land.
   */
  private _onStageActivity(event: AgentEvent, executionId: string): void {
    if (event.type === 'iteration_end') {
      const promptTokens = event.prompt_tokens
      const responseTokens = event.response_tokens
      this._patchExecution(executionId, (state) => ({
        ...state,
        iteration_count: state.iteration_count + 1,
        context_tokens: promptTokens,
        response_tokens: state.response_tokens + responseTokens,
      }))
      this._run.update((run) =>
        run === null
          ? null
          : { ...run, total_tokens_processed: run.total_tokens_processed + promptTokens + responseTokens },
      )
      return
    }

    if (event.type === 'thinking' || event.type === 'content') {
      const text = event.content
      if (text === undefined || text === '') {
        return
      }
      const kind = event.type
      this._appendActivity(executionId, (activity) => {
        const last = activity.length > 0 ? activity[activity.length - 1] : null
        if (last !== null && last.kind === kind) {
          return [...activity.slice(0, -1), { kind, text: last.text + text }]
        }
        return [...activity, { kind, text }]
      })
      return
    }

    if (event.type === 'tool_call_start') {
      // A tool_call_start can arrive twice for the same tool_id — the think-gated llama.cpp
      // path sends an early one for a live raw-text preview, then a second once the call is
      // fully parsed and validated. Reset the existing entry in place instead of appending a
      // duplicate block.
      const toolId = event.tool_id
      const entry: WorkflowActivityEntry = {
        kind: 'tool_call',
        tool_id: toolId,
        tool_name: event.tool_name,
        args_text: '',
      }
      this._appendActivity(executionId, (activity) => {
        const existingIndex = activity.findIndex((a) => a.kind === 'tool_call' && a.tool_id === toolId)
        if (existingIndex === -1) {
          return [...activity, entry]
        }
        return activity.map((a, i) => (i === existingIndex ? entry : a))
      })
      return
    }

    if (event.type === 'tool_call_raw') {
      // Cosmetic live preview of the tool call's raw generated text, sent ahead of the
      // authoritative tool_call_start/tool_call_chunk pair (see think-gated streaming in
      // llama_server.py). Carries no tool_id, so it appends into the most recently opened
      // tool_call entry — the one tool_call_start just reset for this purpose.
      const fragment = event.fragment
      this._appendActivity(executionId, (activity) => {
        const lastIndex = activity.length - 1
        if (lastIndex < 0 || activity[lastIndex].kind !== 'tool_call') {
          return activity
        }
        return activity.map((entry, i) =>
          i === lastIndex && entry.kind === 'tool_call' ? { ...entry, args_text: entry.args_text + fragment } : entry,
        )
      })
      return
    }

    if (event.type === 'tool_call_chunk') {
      const toolId = event.tool_id
      const chunk = event.chunk
      this._appendActivity(executionId, (activity) =>
        activity.map((entry) =>
          entry.kind === 'tool_call' && entry.tool_id === toolId
            ? { ...entry, args_text: entry.args_text + chunk }
            : entry,
        ),
      )
      return
    }

    if (event.type === 'tool_result') {
      const entry: WorkflowActivityEntry = {
        kind: 'tool_result',
        tool_id: event.tool_id,
        tool_name: event.tool_name ?? '',
        log_message: event.log_message ?? null,
        content: event.content ?? '',
      }
      this._appendActivity(executionId, (activity) => [...activity, entry])
      return
    }

    if (event.type === 'error') {
      const message = event.message
      this._appendActivity(executionId, (activity) => [...activity, { kind: 'error', message }])
    }
  }

  private _patchExecution(
    executionId: string,
    patch: (state: WorkflowStageState) => WorkflowStageState,
  ): void {
    this._run.update((run) => {
      if (run === null) {
        return null
      }
      const existing = run.execution_by_id[executionId]
      if (existing === undefined) {
        return run
      }
      return { ...run, execution_by_id: { ...run.execution_by_id, [executionId]: patch(existing) } }
    })
  }

  private _appendActivity(
    executionId: string,
    update: (activity: WorkflowActivityEntry[]) => WorkflowActivityEntry[],
  ): void {
    this._patchExecution(executionId, (state) =>
      state.detail_dropped ? state : { ...state, activity: update(state.activity) },
    )
  }

  /** Register an execution as activity-holding and release the oldest ones past the retention limit. */
  private _retainActivityFor(executionId: string): void {
    this.activityOrder = [...this.activityOrder.filter((id) => id !== executionId), executionId]
    while (this.activityOrder.length > ACTIVITY_RETAINED_EXECUTIONS) {
      const droppedId = this.activityOrder.shift()
      if (droppedId === undefined) {
        break
      }
      this._patchExecution(droppedId, (state) => ({ ...state, activity: [], detail_dropped: true }))
    }
  }

  /** Loop item slots, created lazily on first sight of the loop so the dot grid has a full row. */
  private _ensureLoopItems(run: WorkflowRun, path: string, itemTotal: number): WorkflowLoopItemState[] {
    const existing = run.loop_items[path]
    if (existing !== undefined && existing.length === itemTotal) {
      return existing
    }
    return Array.from({ length: itemTotal }, (_unused, index) => {
      const previous = existing !== undefined ? existing[index] : undefined
      if (previous !== undefined) {
        return previous
      }
      return { item_number: index + 1, status: 'pending' as const, attempts_used: 0, result: null }
    })
  }

  /**
   * Mark the loop item an inner stage is working on as running.
   *
   * The inner stage's path carries the item numbers, so the owning loop is its parent path.
   */
  private _markLoopItemRunning(
    run: WorkflowRun,
    path: string,
    itemNumber: number | null,
    itemTotal: number | null,
  ): Record<string, WorkflowLoopItemState[]> {
    if (itemNumber === null || itemTotal === null) {
      return run.loop_items
    }
    const separatorIndex = path.lastIndexOf('.')
    if (separatorIndex === -1) {
      return run.loop_items
    }
    const loopPath = path.slice(0, separatorIndex)
    const items = this._ensureLoopItems(run, loopPath, itemTotal)
    const updated = items.map((item) =>
      item.item_number === itemNumber && item.status === 'pending' ? { ...item, status: 'running' as const } : item,
    )
    return { ...run.loop_items, [loopPath]: updated }
  }
}
