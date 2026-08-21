import {
  Component,
  ElementRef,
  OnDestroy,
  ViewChild,
  computed,
  effect,
  inject,
  signal,
  untracked,
} from '@angular/core'
import { CommonModule } from '@angular/common'
import { ChatService } from '../../services/chat.service'
import { WorkflowRunService } from '../../services/workflow-run.service'
import { WorkflowActivityEntryComponent } from './workflow-activity-entry.component'
import {
  WorkflowLoopItemState,
  WorkflowNode,
  WorkflowSelectedDetail,
  WorkflowStageState,
  WorkflowStageStatus,
} from '../../types/message-types'

/**
 * One row in the stage list, in display order. The list is the explorer (per user feedback —
 * no separate children list in the detail pane): a 'static' row comes straight from the run's
 * static tree and, for a top-level (non-loop) stage, shows its live/frozen state; a 'fetched' row
 * is a persisted node's child, spliced in beneath its parent once expanded (ADR-0011) — it never
 * carries live state, only whatever was last written to disk.
 */
type ExplorerRow =
  | { kind: 'static'; node: WorkflowNode; depth: number }
  | {
      kind: 'fetched'
      address: string[]
      segment: string
      stageType: string | null
      status: WorkflowStageStatus | null
      depth: number
    }

const RESULT_VALUE_MAX_CHARS = 48

@Component({
  selector: 'app-workflow-run-panel',
  standalone: true,
  imports: [CommonModule, WorkflowActivityEntryComponent],
  templateUrl: './workflow-run-panel.component.html',
  host: { class: 'flex flex-col min-w-0' },
})
export class WorkflowRunPanelComponent implements OnDestroy {
  readonly workflowSvc = inject(WorkflowRunService)
  private readonly chatSvc = inject(ChatService)

  /** Ticks once a second so the elapsed display advances while the run is live. */
  private readonly now = signal(Date.now())
  private readonly tickHandle = setInterval(() => {
    this.now.set(Date.now())
  }, 1000)

  readonly rows = computed<ExplorerRow[]>(() => {
    const run = this.workflowSvc.run()
    if (run === null) {
      return []
    }
    // Touch these so the row list recomputes as nodes are expanded/fetched, not just on live events.
    this.workflowSvc.expandedAddresses()
    this.workflowSvc.fetchedNodes()
    // The currently-running item only auto-expands while genuinely following the live run —
    // once the user has navigated to inspect something else (an old node, another selection),
    // it stops forcing itself into view: going back to an old, finished node must never still
    // show a flashing/live row it has nothing to do with.
    const isFollowing =
      this.workflowSvc.selectedFetchedAddress() === null &&
      this.workflowSvc.selectedExecutionId() === null &&
      this.workflowSvc.selectedLoopItem() === null
    return this._buildStaticRows(run.nodes, 0, isFollowing)
  })

  readonly elapsedLabel = computed(() => {
    const run = this.workflowSvc.run()
    if (run === null) {
      return ''
    }
    const end = run.finished_at !== null ? run.finished_at : this.now()
    const totalSeconds = Math.max(0, Math.floor((end - run.started_at) / 1000))
    const minutes = Math.floor(totalSeconds / 60)
    const seconds = totalSeconds % 60
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  })

  @ViewChild('detailScroll') private _detailScrollEl!: ElementRef<HTMLElement>
  readonly autoScrollEnabled = signal(true)

  constructor() {
    // Live streaming only: stick to the bottom while activity streams in, same rule as the chat
    // message list. Fetched/historical content never streams, so it never auto-scrolls at all —
    // otherwise this fights the user trying to scroll up through something already finished.
    effect(() => {
      this.workflowSvc.selectedDetail()
      const isFetchedView = this.workflowSvc.selectedFetchedAddress() !== null
      untracked(() => {
        if (!isFetchedView && this.autoScrollEnabled()) {
          queueMicrotask(() => {
            const element = this._detailScrollEl?.nativeElement
            if (element) {
              element.scrollTop = element.scrollHeight
            }
          })
        }
      })
    })

    // Switching what's inspected starts at the top and, for a live selection, re-arms follow.
    effect(() => {
      this.workflowSvc.selectedExecutionId()
      this.workflowSvc.selectedLoopItem()
      this.workflowSvc.selectedFetchedAddress()
      untracked(() => {
        this.autoScrollEnabled.set(true)
        queueMicrotask(() => {
          const element = this._detailScrollEl?.nativeElement
          if (element) {
            element.scrollTop = 0
          }
        })
      })
    })
  }

  onDetailScroll(event: Event): void {
    const element = event.target as HTMLElement
    this.autoScrollEnabled.set(element.scrollHeight - element.scrollTop - element.clientHeight < 50)
  }

  ngOnDestroy(): void {
    clearInterval(this.tickHandle)
  }

  private _buildStaticRows(nodes: WorkflowNode[], depth: number, isFollowing: boolean): ExplorerRow[] {
    const rows: ExplorerRow[] = []
    for (const node of nodes) {
      rows.push({ kind: 'static', node, depth })
      if (node.type !== 'loop') {
        rows.push(...this._buildStaticRows(node.children, depth + 1, isFollowing))
        continue
      }
      // A loop's own inner-stage structure is never flattened here unconditionally — it's shared
      // across every item (ADR-0009: one row per loop, not one per item), so showing it as ambient
      // rows for every item would mean a stage nested under item 1 keeps flashing "running" for
      // whichever item is currently active elsewhere in the loop (only one item ever dispatches
      // at a time, so its live state is exactly the loop's own static children's current state).
      // Two ways in: while genuinely following the live run (nothing else selected), the
      // currently-running item auto-expands its own live progress, recursively, so following a
      // run drills all the way down to whatever's generating right now with no click needed —
      // but the moment the user navigates to inspect anything else, this stops, so going back to
      // an old, finished node never still shows a flashing row that belongs to whatever's
      // currently running elsewhere. A finished item's subtree only appears once the user expands
      // its dot, and then it's fetched from disk (never live) via _buildFetchedRows.
      const items = this.loopItemsFor(node)
      const runningItem = items.find((item) => item.status === 'running')
      if (isFollowing && runningItem !== undefined) {
        rows.push(...this._buildStaticRows(node.children, depth + 1, isFollowing))
      }
      for (const item of items) {
        if (item.status !== 'done' && item.status !== 'failed' && item.status !== 'stopped') {
          continue
        }
        const address = [...node.path.split('.'), String(item.item_number)]
        if (this.workflowSvc.isExpanded(address)) {
          rows.push(...this._buildFetchedRows(address, depth + 1))
        }
      }
    }
    return rows
  }

  private _buildFetchedRows(address: string[], depth: number): ExplorerRow[] {
    const node = this.workflowSvc.getFetchedNode(address)
    if (node === null) {
      return []
    }
    const rows: ExplorerRow[] = []
    for (const child of node.children) {
      if (child.stage_type === 'loop_item') {
        // Rendered as a dot in this node's own dot-grid (template, right after the row for this
        // loop), matching the top-level loop's layout exactly — never its own row. Only its
        // children (once that specific dot is expanded) appear as rows, same as a top-level item.
        const itemAddress = this.childAddress(address, child.segment)
        if (this.workflowSvc.isExpanded(itemAddress)) {
          rows.push(...this._buildFetchedRows(itemAddress, depth + 1))
        }
        continue
      }
      const childAddress = this.childAddress(address, child.segment)
      rows.push({
        kind: 'fetched',
        address: childAddress,
        segment: child.segment,
        stageType: child.stage_type,
        status: child.status,
        depth,
      })
      if (this.workflowSvc.isExpanded(childAddress)) {
        rows.push(...this._buildFetchedRows(childAddress, depth + 1))
      }
    }
    return rows
  }

  childAddress(base: string[], segment: string): string[] {
    return [...base, segment]
  }

  /** Latest invocation recorded for a stage path, or null if it has not run yet. */
  stateFor(node: WorkflowNode): WorkflowStageState | null {
    const run = this.workflowSvc.run()
    if (run === null) {
      return null
    }
    const executionId = run.latest_execution_by_path[node.path]
    if (executionId === undefined) {
      return null
    }
    return run.execution_by_id[executionId] ?? null
  }

  statusOf(node: WorkflowNode): WorkflowStageStatus {
    const state = this.stateFor(node)
    return state === null ? 'pending' : state.status
  }

  statusGlyph(status: WorkflowStageStatus): string {
    if (status === 'running') { return '●' }
    if (status === 'done') { return '✓' }
    if (status === 'failed') { return '✗' }
    if (status === 'skipped') { return '⊘' }
    if (status === 'stopped') { return '■' }
    return '○'
  }

  statusClass(status: WorkflowStageStatus): string {
    if (status === 'running') { return 'wf-status-running' }
    if (status === 'done') { return 'wf-status-done' }
    if (status === 'failed') { return 'wf-status-failed' }
    if (status === 'skipped') { return 'wf-status-skipped' }
    if (status === 'stopped') { return 'wf-status-stopped' }
    return 'wf-status-pending'
  }

  dotClass(item: WorkflowLoopItemState): string {
    if (item.status === 'running') { return 'wf-dot wf-dot-running' }
    if (item.status === 'done') { return 'wf-dot wf-dot-done' }
    if (item.status === 'failed') { return 'wf-dot wf-dot-failed' }
    if (item.status === 'stopped') { return 'wf-dot wf-dot-stopped' }
    return 'wf-dot wf-dot-pending'
  }

  /** A top-level loop's own dot grid. Live runs get this from run.loop_items as items complete;
   * a run reopened from disk (ADR-0011's "Deferred: resumability") never populates that, so it
   * falls back to whatever workflowSvc.openPersistedRun already fetched for this loop's own
   * address — same disk data the fetched-row rendering elsewhere in this component already uses. */
  loopItemsFor(node: WorkflowNode): WorkflowLoopItemState[] {
    const run = this.workflowSvc.run()
    if (run === null) {
      return []
    }
    const live = run.loop_items[node.path]
    if (live !== undefined) {
      return live
    }
    const fetched = this.workflowSvc.getFetchedNode(node.path.split('.'))
    if (fetched === null) {
      return []
    }
    const dispatchedItems = fetched.children.filter((child) => child.stage_type === 'loop_item')
    // Any dispatched item knows the loop's real item_total (routers/workflow_runs.py) — a run
    // stopped or failed partway through only ever has directories for the items that actually
    // ran, so without this the grid would silently shrink to "however many happened to finish"
    // instead of showing every remaining item as still pending.
    const itemTotal = dispatchedItems.reduce((max, child) => Math.max(max, child.item_total ?? 0), dispatchedItems.length)
    const byItemNumber = new Map(dispatchedItems.map((child) => [Number(child.segment), child]))
    return Array.from({ length: itemTotal }, (_unused, index) => {
      const itemNumber = index + 1
      const child = byItemNumber.get(itemNumber)
      return {
        item_number: itemNumber,
        // A loop item is never itself "skipped" (only stages are) — WorkflowLoopItemState's
        // status union has no such case, so fold it into 'pending' defensively.
        status: child === undefined || child.status === null || child.status === 'skipped' ? 'pending' : child.status,
        attempts_used: 0,
        result: null,
      }
    })
  }

  loopDonePercent(node: WorkflowNode): number {
    const items = this.loopItemsFor(node)
    if (items.length === 0) {
      return 0
    }
    const settled = items.filter((item) => item.status === 'done' || item.status === 'failed' || item.status === 'stopped').length
    return Math.round((settled / items.length) * 100)
  }

  loopSettledCount(node: WorkflowNode): number {
    return this.loopItemsFor(node).filter((item) => item.status === 'done' || item.status === 'failed' || item.status === 'stopped').length
  }

  dotTitle(item: WorkflowLoopItemState): string {
    return `item ${item.item_number} — ${item.status} (${item.attempts_used} attempt(s))`
  }

  /** One-line rendering of a stage's product, for the collapsed row. */
  resultSummary(result: unknown): string {
    if (result === null || result === undefined) {
      return ''
    }
    if (typeof result !== 'object') {
      return this._shorten(String(result))
    }
    const record = result as Record<string, unknown>
    if (record['_truncated'] === true) {
      return `⟨${String(record['_type'])}, ${String(record['_size_chars'])} chars⟩`
    }
    return Object.entries(record)
      .map(([key, value]) => `${key}: ${this._shorten(this._stringify(value))}`)
      .join(' · ')
  }

  resultJson(result: unknown): string {
    return JSON.stringify(result, null, 2)
  }

  /** True when this static row's own invocation is the one shown in the detail pane. */
  isStaticRowSelected(node: WorkflowNode, state: WorkflowStageState | null): boolean {
    const fetchedAddress = this.workflowSvc.selectedFetchedAddress()
    if (fetchedAddress !== null) {
      return this._sameAddress(node.path.split('.'), fetchedAddress)
    }
    if (state === null) {
      return false
    }
    const detail = this.workflowSvc.selectedDetail()
    return detail !== null && detail.kind === 'execution' && detail.state.execution_id === state.execution_id
  }

  /** True when this exact loop item dot is the one shown in the detail pane. */
  isDotSelected(node: WorkflowNode, item: WorkflowLoopItemState): boolean {
    const fetchedAddress = this.workflowSvc.selectedFetchedAddress()
    if (fetchedAddress !== null) {
      return this._sameAddress([...node.path.split('.'), String(item.item_number)], fetchedAddress)
    }
    const detail = this.workflowSvc.selectedDetail()
    return (
      detail !== null &&
      detail.kind === 'loop_item' &&
      detail.path === node.path &&
      detail.item.item_number === item.item_number
    )
  }

  isFetchedRowSelected(address: string[]): boolean {
    const fetchedAddress = this.workflowSvc.selectedFetchedAddress()
    return fetchedAddress !== null && this._sameAddress(address, fetchedAddress)
  }

  isFetchedRowExpanded(address: string[]): boolean {
    return this.workflowSvc.isExpanded(address)
  }

  isFetchedRowPending(address: string[]): boolean {
    return this.workflowSvc.isPending(address)
  }

  private _sameAddress(a: string[], b: string[]): boolean {
    return a.length === b.length && a.every((segment, i) => segment === b[i])
  }

  /** The loop variable itself (e.g. the file being processed), pulled out for its own labeled row. */
  loopItemVariable(detail: Extract<WorkflowSelectedDetail, { kind: 'loop_item' }>): unknown {
    const result = detail.item.result as Record<string, unknown> | null
    return result === null ? undefined : result['item']
  }

  /** Everything this loop item produced, excluding the loop variable and the bookkeeping success flag. */
  loopItemProduced(detail: Extract<WorkflowSelectedDetail, { kind: 'loop_item' }>): Record<string, unknown> | null {
    const result = detail.item.result as Record<string, unknown> | null
    if (result === null) {
      return null
    }
    const { item: _item, success: _success, ...produced } = result
    return produced
  }

  private _stringify(value: unknown): string {
    if (value === null || value === undefined) {
      return ''
    }
    if (typeof value === 'object') {
      return JSON.stringify(value)
    }
    return String(value)
  }

  private _shorten(text: string): string {
    const flat = text.replace(/\s+/g, ' ').trim()
    if (flat.length <= RESULT_VALUE_MAX_CHARS) {
      return flat
    }
    return `${flat.slice(0, RESULT_VALUE_MAX_CHARS)}…`
  }

  /** A finished static row is fetched from disk (recoverable regardless of in-memory retention);
   * a running one keeps showing its live activity feed as it streams. Neither has anything to
   * expand inline — a plain stage has no children, and a loop's own children are reached via
   * its dots, not its own row. */
  selectStage(node: WorkflowNode): void {
    const state = this.stateFor(node)
    if (state === null) {
      return
    }
    if (state.status === 'running') {
      this.workflowSvc.collapseSiblings(node.path.split('.'))
      this.workflowSvc.selectedFetchedAddress.set(null)
      this.workflowSvc.selectedLoopItem.set(null)
      this.workflowSvc.selectedExecutionId.set(state.execution_id)
      return
    }
    this.workflowSvc.selectFetched(node.path.split('.'))
  }

  /** A finished dot selects that item (its own frozen result) and expands its subtree inline,
   * spliced into the stage list right beneath the loop. A still-running/pending item has nothing
   * on disk yet, so it falls back to the existing live "loop_item" rendering. */
  selectLoopItem(node: WorkflowNode, item: WorkflowLoopItemState): void {
    const address = [...node.path.split('.'), String(item.item_number)]
    if (item.status === 'done' || item.status === 'failed' || item.status === 'stopped') {
      this.workflowSvc.selectFetched(address)
      this.workflowSvc.toggleExpand(address)
      return
    }
    // Nothing to expand for a pending/running item yet, but a previously-expanded sibling's
    // subtree must not linger just because there's nothing new to replace it with.
    this.workflowSvc.collapseSiblings(address)
    this.workflowSvc.selectedFetchedAddress.set(null)
    this.workflowSvc.selectedExecutionId.set(null)
    this.workflowSvc.selectedLoopItem.set({ path: node.path, itemNumber: item.item_number })
  }

  /** A fetched row (anything spliced in beneath an expanded item) selects itself and toggles its
   * own expansion — clicking a leaf just selects it, since an empty children list expands to nothing. */
  selectFetchedRow(address: string[]): void {
    this.workflowSvc.selectFetched(address)
    this.workflowSvc.toggleExpand(address)
  }

  fetchedRowLabel(row: Extract<ExplorerRow, { kind: 'fetched' }>): string {
    return row.stageType === 'loop_item' ? `item ${row.segment}` : row.segment
  }

  /** A fetched loop's own items — its dot-grid, laid out identically to a top-level loop's. */
  fetchedLoopItems(address: string[]): { segment: string; status: WorkflowStageStatus | null }[] {
    return this.workflowSvc.getFetchedNode(address)?.children ?? []
  }

  fetchedLoopSettledCount(items: { status: WorkflowStageStatus | null }[]): number {
    return items.filter((item) => item.status === 'done' || item.status === 'failed' || item.status === 'stopped').length
  }

  fetchedLoopDonePercent(items: { status: WorkflowStageStatus | null }[]): number {
    if (items.length === 0) {
      return 0
    }
    return Math.round((this.fetchedLoopSettledCount(items) / items.length) * 100)
  }

  fetchedDotClass(status: WorkflowStageStatus | null): string {
    if (status === 'failed') { return 'wf-dot wf-dot-failed' }
    if (status === 'stopped') { return 'wf-dot wf-dot-stopped' }
    if (status === 'done') { return 'wf-dot wf-dot-done' }
    return 'wf-dot wf-dot-pending'
  }

  fetchedDotTitle(item: { segment: string; status: WorkflowStageStatus | null }): string {
    return `item ${item.segment} — ${item.status ?? 'pending'}`
  }

  followRunningStage(): void {
    this.workflowSvc.followRunning()
  }

  /** "Resume from here" only makes sense once the run has actually stopped moving forward on its
   * own (not while it's still 'running') and something with a real on-disk address is selected —
   * see the resumable-workflow-runs plan's Phase 4. */
  readonly canResumeFromHere = computed(() => {
    const run = this.workflowSvc.run()
    if (run === null || (run.status !== 'stopped' && run.status !== 'error')) {
      return false
    }
    return this.workflowSvc.selectedFetchedAddress() !== null
  })

  resumeFromHere(): void {
    const run = this.workflowSvc.run()
    const address = this.workflowSvc.selectedFetchedAddress()
    if (run === null || address === null) {
      return
    }
    this.chatSvc.resumeWorkflowRun(run.workflow_name, run.run_id, address)
  }
}
