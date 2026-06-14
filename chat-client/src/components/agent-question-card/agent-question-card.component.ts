import {
  AfterViewInit,
  Component,
  ElementRef,
  HostListener,
  QueryList,
  ViewChildren,
  computed,
  input,
  output,
  signal,
} from '@angular/core'

@Component({
  selector: 'app-agent-question-card',
  standalone: true,
  templateUrl: './agent-question-card.component.html',
  styleUrls: ['./agent-question-card.component.scss'],
})
export class AgentQuestionCardComponent implements AfterViewInit {
  readonly questionId = input.required<string>()
  readonly question = input.required<string>()
  readonly options = input<string[]>([])
  readonly resolved = input(false)

  readonly replied = output<string>()

  protected readonly allEntries = computed(() => [...this.options(), 'Other:'])
  protected readonly activeIndex = signal(0)
  protected readonly commentMap = signal<Record<number, string>>({})

  @ViewChildren('entryInput') private _inputs!: QueryList<ElementRef<HTMLInputElement>>

  ngAfterViewInit(): void {
    if (!this.resolved()) {
      this._focusActiveInput()
    }
  }

  @HostListener('keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if (this.resolved()) { return }
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      this.activeIndex.update((i) => (i + 1) % this.allEntries().length)
      this._focusActiveInput()
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      this.activeIndex.update((i) => (i - 1 + this.allEntries().length) % this.allEntries().length)
      this._focusActiveInput()
    } else if (event.key === 'Enter') {
      this._submit()
    }
  }

  protected onEntryClick(index: number): void {
    this.activeIndex.set(index)
    this._focusActiveInput()
  }

  protected getComment(index: number): string {
    return this.commentMap()[index] ?? ''
  }

  protected updateComment(index: number, value: string): void {
    this.commentMap.update((m) => ({ ...m, [index]: value }))
  }

  protected isOtherEntry(index: number): boolean {
    return index === this.allEntries().length - 1
  }

  protected canSubmitActive(): boolean {
    const i = this.activeIndex()
    if (this.isOtherEntry(i)) {
      return (this.commentMap()[i] ?? '').trim() !== ''
    }
    return true
  }

  private _focusActiveInput(): void {
    queueMicrotask(() => {
      this._inputs.toArray()[this.activeIndex()]?.nativeElement.focus()
    })
  }

  private _submit(): void {
    if (!this.canSubmitActive()) { return }
    const i = this.activeIndex()
    const comment = (this.commentMap()[i] ?? '').trim()
    if (this.isOtherEntry(i)) {
      this.replied.emit(comment)
    } else {
      const option = this.allEntries()[i]
      this.replied.emit(comment !== '' ? `${option} — ${comment}` : option)
    }
  }
}
