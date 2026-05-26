import { AfterViewInit, Component, computed, ElementRef, inject, input, OnDestroy, OnInit, output, signal, ViewChild } from '@angular/core'
import { ApiService } from '../../services/api.service'

type DirEntry = { name: string; path: string }

@Component({
  selector: 'app-directory-picker',
  standalone: true,
  templateUrl: './directory-picker.component.html',
  styleUrl: './directory-picker.component.scss',
})
export class DirectoryPickerComponent implements OnInit, AfterViewInit, OnDestroy {
  readonly initialPath = input<string | null>(null)
  readonly selected = output<string>()
  readonly cancelled = output<void>()

  private api = inject(ApiService)
  private el = inject(ElementRef)

  @ViewChild('listContainer') listContainer!: ElementRef<HTMLElement>
  @ViewChild('searchInput') searchInput!: ElementRef<HTMLInputElement>

  readonly currentPath = signal('')
  readonly parent = signal<string | null>(null)
  readonly entries = signal<DirEntry[]>([])
  readonly loading = signal(false)
  readonly error = signal<string | null>(null)
  readonly filterText = signal('')
  readonly activePath = signal<string | null>(null)

  // ".." included as a regular entry so filter applies uniformly
  readonly allEntries = computed<DirEntry[]>(() => {
    const p = this.parent()
    return p ? [{ name: '..', path: p }, ...this.entries()] : this.entries()
  })

  readonly filteredEntries = computed(() => {
    const f = this.filterText().toLowerCase().trim()
    return f ? this.allEntries().filter(e => e.name.toLowerCase().includes(f)) : this.allEntries()
  })

  ngOnInit() {
    document.body.appendChild(this.el.nativeElement)
    this.browse(this.initialPath())
  }

  ngAfterViewInit() {
    this.searchInput.nativeElement.focus()
  }

  ngOnDestroy() {
    this.el.nativeElement.remove()
  }

  browse(path: string | null) {
    this.loading.set(true)
    this.error.set(null)
    this.filterText.set('')
    this.activePath.set(null)
    this.api.browse_directory(path).subscribe({
      next: (result) => {
        this.currentPath.set(result.path)
        this.parent.set(result.parent)
        this.entries.set(result.entries)
        this.loading.set(false)
        this.activePath.set(this.allEntries()[0]?.path ?? null)
      },
      error: (err) => {
        this.error.set(err?.error?.detail ?? 'Failed to read directory')
        this.loading.set(false)
      },
    })
  }

  onFilterInput(event: Event) {
    this.filterText.set((event.target as HTMLInputElement).value)
    const prev = this.activePath()
    const nav = this.filteredEntries()
    const stillVisible = prev !== null && nav.some(e => e.path === prev)
    if (!stillVisible) {
      this.activePath.set(nav[0]?.path ?? null)
    }
  }

  onKeydown(event: KeyboardEvent) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      const nav = this.filteredEntries()
      if (nav.length === 0) {
        return
      }
      const cur = nav.findIndex(e => e.path === this.activePath())
      const next = event.key === 'ArrowDown'
        ? Math.min(cur + 1, nav.length - 1)
        : Math.max(cur - 1, 0)
      // cur === -1 (no selection) + any arrow → pick first
      this.activePath.set(nav[cur === -1 ? 0 : next].path)
      this.scrollActiveIntoView()
    } else if (event.key === 'Enter') {
      if (event.ctrlKey) {
        event.preventDefault()
        this.select()
      } else {
        event.preventDefault()
        const active = this.activePath()
        if (active !== null) {
          this.browse(active)
        }
      }
    } else if (event.key === 'Backspace' && !this.filterText()) {
      event.preventDefault()
      const p = this.parent()
      if (p) {
        this.browse(p)
      }
    } else if (event.key === 'Escape') {
      if (this.filterText()) {
        this.filterText.set('')
        this.activePath.set(null)
        ;(event.target as HTMLInputElement).value = ''
      } else {
        this.cancelled.emit()
      }
    }
  }

  isActive(entry: DirEntry): boolean {
    return this.activePath() === entry.path
  }

  private scrollActiveIntoView() {
    const active = this.activePath()
    if (!active) {
      return
    }
    setTimeout(() => {
      const el = this.listContainer?.nativeElement.querySelector(`[data-nav-path="${CSS.escape(active)}"]`)
      el?.scrollIntoView({ block: 'nearest' })
    })
  }

  select() {
    this.selected.emit(this.currentPath())
  }
}
