import { Injectable, signal, computed } from '@angular/core'
import { Subject, Observable } from 'rxjs'
import { AgentEvent, AgentUiMessage } from '../types/message-types'

const WS_URL = 'ws://localhost:8000/api/agent/ws'

@Injectable({ providedIn: 'root' })
export class AgentService {
  private ws: WebSocket | null = null

  // All UI messages for the current agent run
  private _messages = signal<AgentUiMessage[]>([])
  public readonly messages = this._messages.asReadonly()

  // Whether an agent run is in progress
  private _running = signal(false)
  public readonly running = this._running.asReadonly()

  // Current prompt_tokens from the latest iteration_end
  private _promptTokens = signal(0)
  public readonly promptTokens = this._promptTokens.asReadonly()

  // Emits when the agent finishes (done or error)
  private _done$ = new Subject<void>()
  public readonly done$: Observable<void> = this._done$.asObservable()

  // Current streaming phase: null when idle, 'thinking' while thinking streams, 'responding' while content streams
  public readonly phase = computed<'thinking' | 'responding' | null>(() => {
    if (!this._running()) return null
    const msgs = this._messages()
    const last = msgs[msgs.length - 1]
    if (!last || last.done) return null
    if (last.ui_type === 'thinking') return 'thinking'
    if (last.ui_type === 'content') return 'responding'
    return null
  })

  startWithUserMessage(userMessage: string, conversationId?: string): void {
    this._messages.set([{
      id: crypto.randomUUID(),
      ui_type: 'user',
      content: userMessage,
    }])
    this.start(userMessage, conversationId)
  }

  start(userMessage: string, conversationId?: string): void {
    this._running.set(true)
    this._promptTokens.set(0)

    // Accumulator IDs for streaming thinking/content blocks
    let thinkingId: string | null = null
    let contentId: string | null = null

    this.ws = new WebSocket(WS_URL)

    this.ws.onopen = () => {
      this.ws!.send(JSON.stringify({ message: userMessage, conversation_id: conversationId ?? null }))
    }

    this.ws.onmessage = (ev) => {
      const event: AgentEvent = JSON.parse(ev.data)
      this._handleEvent(event, { thinkingId, contentId }, (ids) => {
        thinkingId = ids.thinkingId
        contentId = ids.contentId
      })
    }

    this.ws.onerror = () => {
      this._running.set(false)
      this._done$.next()
    }

    this.ws.onclose = () => {
      this._running.set(false)
    }
  }

  private _handleEvent(
    event: AgentEvent,
    acc: { thinkingId: string | null; contentId: string | null },
    setAcc: (ids: { thinkingId: string | null; contentId: string | null }) => void,
  ): void {
    let { thinkingId, contentId } = acc

    if (event.type === 'thinking' && event.content) {
      if (thinkingId) {
        this._updateMessage(thinkingId, (m) => ({ ...m, content: m.content + event.content }))
      } else {
        thinkingId = crypto.randomUUID()
        this._pushMessage({ id: thinkingId, ui_type: 'thinking', content: event.content! })
      }
    } else if (event.type === 'content' && event.content) {
      if (contentId) {
        this._updateMessage(contentId, (m) => ({ ...m, content: m.content + event.content }))
      } else {
        contentId = crypto.randomUUID()
        if (thinkingId) {
          this._updateMessage(thinkingId, (m) => ({ ...m, done: true }))
        }
        thinkingId = null
        this._pushMessage({ id: contentId, ui_type: 'content', content: event.content! })
      }
    } else if (event.type === 'tool_confirm') {
      this._pushMessage({
        id: event.tool_id!,
        ui_type: 'tool_confirm',
        tool_id: event.tool_id,
        tool_name: event.tool_name,
        tool_args: event.arguments,
        preview: event.preview,
        content: '',
        confirmed: null,
      })
    } else if (event.type === 'tool_result') {
      this._pushMessage({
        id: `result-${event.tool_id}`,
        ui_type: 'tool_result',
        tool_id: event.tool_id,
        tool_name: event.tool_name,
        content: event.content ?? '',
      })
      if (thinkingId) this._updateMessage(thinkingId, (m) => ({ ...m, done: true }))
      if (contentId) this._updateMessage(contentId, (m) => ({ ...m, done: true }))
      thinkingId = null
      contentId = null
    } else if (event.type === 'iteration_end') {
      this._promptTokens.set(event.prompt_tokens ?? 0)
      if (thinkingId) this._updateMessage(thinkingId, (m) => ({ ...m, done: true }))
      if (contentId) this._updateMessage(contentId, (m) => ({ ...m, done: true }))
      thinkingId = null
      contentId = null
      this._pushMessage({ id: crypto.randomUUID(), ui_type: 'iteration_end', content: '', prompt_tokens: event.prompt_tokens })
    } else if (event.type === 'done' || event.type === 'error') {
      this._running.set(false)
      this._done$.next()
    }

    setAcc({ thinkingId, contentId })
  }

  confirm(toolId: string, approved: boolean, reason?: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'confirm', tool_id: toolId, approved, reason: reason ?? null }))
    }
    // Mark the confirmation card in UI
    this._updateMessage(toolId, (m) => ({ ...m, confirmed: approved }))
  }

  abort(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'abort' }))
    }
    this._running.set(false)
  }

  clearMessages(): void {
    this._messages.set([])
  }

  private _pushMessage(msg: AgentUiMessage): void {
    this._messages.update((msgs) => [...msgs, msg])
  }

  private _updateMessage(id: string, updater: (m: AgentUiMessage) => AgentUiMessage): void {
    this._messages.update((msgs) => msgs.map((m) => (m.id === id ? updater(m) : m)))
  }
}
