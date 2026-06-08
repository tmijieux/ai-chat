import { Injectable, signal } from '@angular/core'
import { Observable, Subject } from 'rxjs'
import { AgentEvent } from '../types/message-types'

export type AgentMode = 'classic' | 'pipeline'

function wsUrl(mode: AgentMode): string {
  const path = mode === 'pipeline' ? '/api/agent/pipeline/ws' : '/api/agent/ws'
  return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + path
}

/**
 * Pure WebSocket transport for the agent loop.
 * Owns: socket lifecycle, confirm/abort messages, running flag.
 * Does NOT hold display state — ChatService subscribes to events$ and owns that.
 */
@Injectable({ providedIn: 'root' })
export class AgentService {
  private ws: WebSocket | null = null

  private _running = signal(false)
  public readonly running = this._running.asReadonly()

  private _events$ = new Subject<AgentEvent>()
  /** Raw event stream. ChatService subscribes to accumulate DisplayMessages. */
  public readonly events$: Observable<AgentEvent> = this._events$.asObservable()

  start(userMessage: string, conversationId?: string, userMessageId?: string, mode: AgentMode = 'classic'): void {
    this._running.set(true)
    this.ws = new WebSocket(wsUrl(mode))

    this.ws.onopen = () => {
      this.ws!.send(JSON.stringify({
        message: userMessage,
        conversation_id: conversationId ?? null,
        user_message_id: userMessageId ?? null,
      }))
    }

    this.ws.onmessage = (ev) => {
      const event: AgentEvent = JSON.parse(ev.data)
      this._events$.next(event)
      if (event.type === 'done' || event.type === 'error') {
        this._running.set(false)
      }
    }

    this.ws.onerror = () => {
      this._running.set(false)
      this._events$.next({ type: 'error', message: 'WebSocket error' })
    }

    this.ws.onclose = () => {
      if (this._running()) {
        this._events$.next({ type: 'error', message: 'Connection lost' })
      }
      this._running.set(false)
    }
  }

  confirm(toolId: string, approved: boolean, reason?: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'confirm', tool_id: toolId, approved, reason: reason ?? null }))
    }
  }

  compressionDone(conversationId: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'compression_done', conversation_id: conversationId }))
    }
  }

  abort(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'abort' }))
    }
    this._running.set(false)
  }
}
