import { Component, computed, effect, inject, input, output, signal, untracked } from '@angular/core'
import { CommonModule } from '@angular/common'
import { takeUntilDestroyed } from '@angular/core/rxjs-interop'
import { toObservable } from '@angular/core/rxjs-interop'
import { debounceTime, of, switchMap } from 'rxjs'
import { ApiService } from '../../services/api.service'
import { FileSearchResult } from '../../types/message-types'

@Component({
  selector: 'app-file-mention-picker',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './file-mention-picker.component.html',
})
export class FileMentionPickerComponent {
  private api = inject(ApiService)

  /** Text typed after '@' — used as the search query. */
  readonly filter = input.required<string>()
  /** Absolute path of the workspace root to search within. */
  readonly workspacePath = input.required<string>()

  /** Emits the absolute path of the selected file. */
  readonly fileSelected = output<string>()
  /** Emits when the picker should be dismissed (Escape). */
  readonly dismissed = output<void>()

  private readonly _results = signal<FileSearchResult[]>([])
  private readonly _activeIndex = signal(0)
  private readonly _loading = signal(false)

  readonly results = this._results.asReadonly()
  readonly activeIndex = this._activeIndex.asReadonly()
  readonly loading = this._loading.asReadonly()
  readonly hasResults = computed(() => this._results().length > 0)

  constructor() {
    toObservable(this.filter)
      .pipe(
        debounceTime(150),
        switchMap((query) => {
          const workspace = this.workspacePath()
          if (!workspace) {
            return of({ results: [] as FileSearchResult[] })
          }
          untracked(() => this._loading.set(true))
          return this.api.search_files(workspace, query)
        }),
        takeUntilDestroyed(),
      )
      .subscribe({
        next: (response) => {
          this._results.set(response.results)
          this._activeIndex.set(0)
          this._loading.set(false)
        },
        error: () => {
          this._results.set([])
          this._loading.set(false)
        },
      })
  }

  navigateUp(): void {
    this._activeIndex.update((i) => Math.max(0, i - 1))
  }

  navigateDown(): void {
    this._activeIndex.update((i) => Math.min(this._results().length - 1, i + 1))
  }

  resetIndex(): void {
    this._activeIndex.set(0)
  }

  selectActive(): void {
    const item = this._results()[this._activeIndex()]
    if (item !== undefined) {
      this.fileSelected.emit(item.path)
    }
  }

  selectItem(item: FileSearchResult): void {
    this.fileSelected.emit(item.path)
  }

  isActive(index: number): boolean {
    return this._activeIndex() === index
  }
}
