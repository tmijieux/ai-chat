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
import { WorkflowRunService } from '../../services/workflow-run.service'
import {
  WorkflowLoopItemState,
  WorkflowNode,
  WorkflowStageState,
  WorkflowStageStatus,
} from '../../types/message-types'

/** One stage row in display order, carrying its nesting depth for indentation. */
type WorkflowRow = {
  node: WorkflowNode
  depth: number
}

const RESULT_VALUE_MAX_CHARS = 48

@Component({
  selector: 'app-workflow-run-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './workflow-run-panel.component.html',
  host: { class: 'flex flex-col min-w-0' },
})
export class WorkflowRunPanelComponent implements OnDestroy {
  readonly workflowSvc = inject(WorkflowRunService)

  /** Ticks once a second so the elapsed display advances while the run is live. */
  private readonly now = signal(Date.now())
  private readonly tickHandle = setInterval(() => {
    this.now.set(Date.now())
  }, 1000)

  readonly rows = computed<WorkflowRow[]>(() => {
    const run = this.workflowSvc.run()
    if (run === null) {
      return []
    }
    return this._flatten(run.nodes, 0)
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

  // Auto-scroll the detail pane, same rule as the chat message list: stick to the bottom while
  // activity streams in, stop the moment the user scrolls up to read, resume when they come back.
  @ViewChild('detailScroll') private _detailScrollEl!: ElementRef<HTMLElement>
  readonly autoScrollEnabled = signal(true)

  constructor() {
    effect(() => {
      // Registers the dependency: every activity append replaces the stage state object, so this
      // re-runs as thinking text streams in, not just when a new entry is added.
      this.workflowSvc.selectedExecution()
      untracked(() => {
        if (this.autoScrollEnabled()) {
          queueMicrotask(() => {
            const element = this._detailScrollEl?.nativeElement
            if (element) {
              element.scrollTop = element.scrollHeight
            }
          })
        }
      })
    })

    // Switching which stage is inspected starts pinned to the bottom again.
    effect(() => {
      this.workflowSvc.selectedExecutionId()
      untracked(() => {
        this.autoScrollEnabled.set(true)
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

  private _flatten(nodes: WorkflowNode[], depth: number): WorkflowRow[] {
    const rows: WorkflowRow[] = []
    for (const node of nodes) {
      rows.push({ node, depth })
      rows.push(...this._flatten(node.children, depth + 1))
    }
    return rows
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
    return '○'
  }

  statusClass(status: WorkflowStageStatus): string {
    if (status === 'running') { return 'wf-status-running' }
    if (status === 'done') { return 'wf-status-done' }
    if (status === 'failed') { return 'wf-status-failed' }
    if (status === 'skipped') { return 'wf-status-skipped' }
    return 'wf-status-pending'
  }

  dotClass(item: WorkflowLoopItemState): string {
    if (item.status === 'running') { return 'wf-dot wf-dot-running' }
    if (item.status === 'done') { return 'wf-dot wf-dot-done' }
    if (item.status === 'failed') { return 'wf-dot wf-dot-failed' }
    return 'wf-dot wf-dot-pending'
  }

  loopItemsFor(node: WorkflowNode): WorkflowLoopItemState[] {
    const run = this.workflowSvc.run()
    if (run === null) {
      return []
    }
    return run.loop_items[node.path] ?? []
  }

  loopDonePercent(node: WorkflowNode): number {
    const items = this.loopItemsFor(node)
    if (items.length === 0) {
      return 0
    }
    const settled = items.filter((item) => item.status === 'done' || item.status === 'failed').length
    return Math.round((settled / items.length) * 100)
  }

  loopSettledCount(node: WorkflowNode): number {
    return this.loopItemsFor(node).filter((item) => item.status === 'done' || item.status === 'failed').length
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

  selectStage(node: WorkflowNode): void {
    const state = this.stateFor(node)
    if (state !== null) {
      this.workflowSvc.selectedExecutionId.set(state.execution_id)
    }
  }

  selectLoopItem(node: WorkflowNode, item: WorkflowLoopItemState): void {
    const state = this.workflowSvc.latestExecutionForLoopItem(node.path, item.item_number)
    if (state !== null) {
      this.workflowSvc.selectedExecutionId.set(state.execution_id)
    }
  }

  followRunningStage(): void {
    this.workflowSvc.selectedExecutionId.set(null)
  }
}
