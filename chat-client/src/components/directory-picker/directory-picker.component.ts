import { Component, ElementRef, inject, OnDestroy, OnInit, output, signal } from '@angular/core'
import { ApiService } from '../../services/api.service'

type DirEntry = { name: string; path: string }

@Component({
  selector: 'app-directory-picker',
  standalone: true,
  templateUrl: './directory-picker.component.html',
  styleUrl: './directory-picker.component.scss',
})
export class DirectoryPickerComponent implements OnInit, OnDestroy {
  readonly selected = output<string>()
  readonly cancelled = output<void>()

  private api = inject(ApiService)
  private el = inject(ElementRef)

  readonly currentPath = signal('')
  readonly parent = signal<string | null>(null)
  readonly entries = signal<DirEntry[]>([])
  readonly loading = signal(false)
  readonly error = signal<string | null>(null)

  ngOnInit() {
    document.body.appendChild(this.el.nativeElement)
    this.browse(null)
  }

  ngOnDestroy() {
    this.el.nativeElement.remove()
  }

  browse(path: string | null) {
    this.loading.set(true)
    this.error.set(null)
    this.api.browse_directory(path).subscribe({
      next: (result) => {
        this.currentPath.set(result.path)
        this.parent.set(result.parent)
        this.entries.set(result.entries)
        this.loading.set(false)
      },
      error: (err) => {
        this.error.set(err?.error?.detail ?? 'Failed to read directory')
        this.loading.set(false)
      },
    })
  }

  select() {
    this.selected.emit(this.currentPath())
  }
}
