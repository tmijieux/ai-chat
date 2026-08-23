import { Injectable } from '@angular/core'
import { Observable, Subject } from 'rxjs'
import { RagIndexEvent } from '../types/message-types'

function wsUrl(spaceId: string): string {
  const path = `/api/rag/spaces/${spaceId}/sources/workspace-path/ws`
  return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + path
}

/**
 * Pure WebSocket transport for a single RAG workspace-path ingestion run — streams per-file
 * progress and a final done/error/cancelled event. Mirrors AgentService's shape (socket
 * lifecycle owned here, ChatService subscribes to events$ and owns display state).
 */
@Injectable({ providedIn: 'root' })
export class RagIndexService {
  private ws: WebSocket | null = null
  private _settled = false

  private _events$ = new Subject<RagIndexEvent>()
  public readonly events$: Observable<RagIndexEvent> = this._events$.asObservable()

  start(spaceId: string, workspace: string, path: string): void {
    this._settled = false
    this.ws = new WebSocket(wsUrl(spaceId))

    this.ws.onopen = () => {
      this.ws!.send(JSON.stringify({ workspace, path }))
    }

    this.ws.onmessage = (ev) => {
      const event: RagIndexEvent = JSON.parse(ev.data)
      if (event.type === 'done' || event.type === 'error' || event.type === 'cancelled') {
        this._settled = true
      }
      this._events$.next(event)
    }

    this.ws.onerror = () => {
      if (!this._settled) {
        this._settled = true
        this._events$.next({ type: 'error', message: 'WebSocket error' })
      }
    }

    this.ws.onclose = () => {
      if (!this._settled) {
        this._settled = true
        this._events$.next({ type: 'error', message: 'Connection lost' })
      }
    }
  }

  cancel(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'cancel' }))
    }
  }
}
