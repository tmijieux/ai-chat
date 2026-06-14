import { Component, computed, input, output, signal } from '@angular/core'
import { CommonModule } from '@angular/common'
import { ConversationMode, SlashCommand, Workflow } from '../../types/message-types'

const MODES: SlashCommand[] = [
  { type: 'mode', value: 'standard', label: 'standard', description: 'Default agent mode' },
  { type: 'mode', value: 'auto',     label: 'auto',     description: 'Auto-approve safe tool calls' },
  { type: 'mode', value: 'plan',     label: 'plan',     description: 'Plan before editing — no file writes' },
  { type: 'mode', value: 'yolo',     label: 'yolo',     description: 'Autonomous loop with verification' },
]

@Component({
  selector: 'app-slash-command-palette',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './slash-command-palette.component.html',
})
export class SlashCommandPaletteComponent {
  /** Text the user has typed after the leading '/'. */
  readonly filter = input.required<string>()
  readonly workflows = input<Workflow[]>([])

  readonly commandSelected = output<SlashCommand>()
  readonly dismissed = output<void>()

  private readonly _activeIndex = signal(0)
  readonly activeIndex = this._activeIndex.asReadonly()

  readonly filteredModes = computed(() => {
    const f = this.filter().toLowerCase()
    return MODES.filter((m) => m.value.includes(f))
  })

  readonly filteredWorkflows = computed(() => {
    const f = this.filter().toLowerCase()
    return this.workflows()
      .filter((w) => w.name.toLowerCase().includes(f) || w.description.toLowerCase().includes(f))
      .map<SlashCommand>((w) => ({ type: 'workflow', value: w.name, label: w.name, description: w.description }))
  })

  readonly allFiltered = computed<SlashCommand[]>(() => [
    ...this.filteredModes(),
    ...this.filteredWorkflows(),
  ])

  readonly hasResults = computed(() => this.allFiltered().length > 0)

  navigateUp(): void {
    this._activeIndex.update((i) => Math.max(0, i - 1))
  }

  navigateDown(): void {
    this._activeIndex.update((i) => Math.min(this.allFiltered().length - 1, i + 1))
  }

  resetIndex(): void {
    this._activeIndex.set(0)
  }

  selectActive(): void {
    const item = this.allFiltered()[this._activeIndex()]
    if (item !== undefined) {
      this.commandSelected.emit(item)
    }
  }

  selectItem(item: SlashCommand): void {
    this.commandSelected.emit(item)
  }

  isActive(index: number): boolean {
    return this._activeIndex() === index
  }

  /** Returns the absolute index in allFiltered for a mode item. */
  modeIndex(modeIndex: number): number {
    return modeIndex
  }

  /** Returns the absolute index in allFiltered for a workflow item. */
  workflowIndex(workflowIndex: number): number {
    return this.filteredModes().length + workflowIndex
  }
}
