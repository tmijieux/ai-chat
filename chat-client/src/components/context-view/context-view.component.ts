import { Component, computed, effect, inject, input, signal, untracked } from '@angular/core'
import { CommonModule } from '@angular/common'
import { ApiService } from '../../services/api.service'
import { ContextEntry } from '../../types/message-types'

@Component({
  selector: 'app-context-view',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './context-view.component.html',
  host: { class: 'flex flex-col' },
})
export class ContextViewComponent {
  private api = inject(ApiService)

  readonly conversationId = input<string | undefined>(undefined)
  /** Increment this to force a refresh (e.g. pass promptTokens() from ChatService). */
  readonly refreshTrigger = input<number>(0)

  readonly entries = signal<ContextEntry[]>([])
  readonly loading = signal(false)

  readonly totalTokens = computed(() => this.entries().reduce((sum, entry) => sum + entry.token_count, 0))

  constructor() {
    effect(() => {
      const conversationId = this.conversationId()
      this.refreshTrigger()
      untracked(() => {
        if (conversationId === undefined || conversationId === null) {
          this.entries.set([])
          return
        }
        this._load(conversationId)
      })
    })
  }

  private _load(conversationId: string): void {
    this.loading.set(true)
    this.api.get_inference_context(conversationId).subscribe({
      next: (result) => {
        this.entries.set(result.entries)
        this.loading.set(false)
      },
      error: () => {
        this.loading.set(false)
      },
    })
  }

  roleBadgeClass(role: string): string {
    if (role === 'system') { return 'ctx-badge-system' }
    if (role === 'user') { return 'ctx-badge-user' }
    if (role === 'assistant') { return 'ctx-badge-assistant' }
    if (role === 'tool') { return 'ctx-badge-tool' }
    return 'badge-neutral'
  }

  entryContentClass(role: string): string {
    if (role === 'system') { return 'ctx-bg-system' }
    if (role === 'user') { return 'ctx-bg-user' }
    if (role === 'assistant') { return 'ctx-bg-assistant' }
    if (role === 'tool') { return 'ctx-bg-tool' }
    return 'ctx-bg-assistant'
  }
}
