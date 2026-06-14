import {
  Component,
  ElementRef,
  HostListener,
  QueryList,
  ViewChildren,
  input,
  output,
  signal,
} from '@angular/core'

type PlanEntry = {
  label: string
  mode?: string
  requiresText: boolean
  isAbort?: boolean
}

export type PlanAcceptPayload = {
  status: 'accepted' | 'feedback'
  mode?: string
  comment?: string
  feedback?: string
}

@Component({
  selector: 'app-plan-card',
  standalone: true,
  templateUrl: './plan-card.component.html',
  styleUrls: ['./plan-card.component.scss'],
})
export class PlanCardComponent {
  readonly planId = input.required<string>()
  readonly plan = input.required<string>()
  readonly resolved = input(false)
  readonly resolution = input<string | undefined>(undefined)

  readonly accepted = output<PlanAcceptPayload>()
  readonly aborted = output<void>()

  protected readonly ENTRIES: PlanEntry[] = [
    { label: 'Accept → Standard', mode: 'standard', requiresText: false },
    { label: 'Accept → Auto',     mode: 'auto',     requiresText: false },
    { label: 'Accept → YOLO',     mode: 'yolo',     requiresText: false },
    { label: 'Refine…',                             requiresText: true  },
    { label: 'Abort',             isAbort: true,    requiresText: false },
  ]

  protected readonly activeIndex = signal(0)
  protected readonly comments = signal<string[]>(Array(5).fill(''))

  @ViewChildren('entryInput') private _inputs!: QueryList<ElementRef<HTMLInputElement>>

  @HostListener('keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if (this.resolved()) { return }
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      this.activeIndex.update((i) => (i + 1) % this.ENTRIES.length)
      this._focusActiveInput()
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      this.activeIndex.update((i) => (i - 1 + this.ENTRIES.length) % this.ENTRIES.length)
      this._focusActiveInput()
    } else if (event.key === 'Enter') {
      this._submit()
    }
  }

  protected onEntryClick(index: number): void {
    this.activeIndex.set(index)
    if (this.ENTRIES[index].isAbort) {
      this.aborted.emit()
      return
    }
    this._focusActiveInput()
  }

  protected updateComment(index: number, value: string): void {
    const arr = [...this.comments()]
    arr[index] = value
    this.comments.set(arr)
  }

  protected canSubmitActive(): boolean {
    const i = this.activeIndex()
    const entry = this.ENTRIES[i]
    if (entry.isAbort) { return true }
    if (entry.requiresText) { return this.comments()[i].trim() !== '' }
    return true
  }

  private _focusActiveInput(): void {
    const i = this.activeIndex()
    if (this.ENTRIES[i].isAbort) { return }
    queueMicrotask(() => {
      this._inputs.toArray()[i]?.nativeElement.focus()
    })
  }

  private _submit(): void {
    const i = this.activeIndex()
    const entry = this.ENTRIES[i]
    if (entry.isAbort) {
      this.aborted.emit()
      return
    }
    const comment = this.comments()[i].trim()
    if (entry.requiresText && comment === '') { return }
    if (entry.mode !== undefined) {
      this.accepted.emit({ status: 'accepted', mode: entry.mode, comment: comment || undefined })
    } else {
      this.accepted.emit({ status: 'feedback', feedback: comment })
    }
  }
}
