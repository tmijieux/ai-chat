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
    if (role === 'system') { return 'bg-purple-900 text-purple-300' }
    if (role === 'user') { return 'bg-blue-900 text-blue-300' }
    if (role === 'assistant') { return 'bg-green-900 text-green-300' }
    if (role === 'tool') { return 'bg-amber-900 text-amber-300' }
    return 'bg-gray-800 text-gray-400'
  }

  entryContentClass(role: string): string {
    if (role === 'system') { return 'bg-purple-950' }
    if (role === 'user') { return 'bg-blue-950' }
    if (role === 'assistant') { return 'bg-gray-900' }
    if (role === 'tool') { return 'bg-amber-950' }
    return 'bg-gray-900'
  }
}
