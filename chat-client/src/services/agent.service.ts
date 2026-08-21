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

  start(userMessage: string, conversationId?: string, userMessageId?: string, mode: AgentMode = 'classic', workflowName?: string): void {
    this._openSocket(mode, {
      message: userMessage,
      conversation_id: conversationId ?? null,
      user_message_id: userMessageId ?? null,
      workflow_name: workflowName ?? null,
    })
  }

  /** Resume a previously stopped/failed workflow run in place, continuing from resumeAddress —
   * see ADR-0011's "Deferred: resumability". Goes through the same live event stream as a fresh
   * run (not a fire-and-forget REST call), so the panel updates identically either way. */
  startResume(conversationId: string, workflowName: string, resumeRunId: string, resumeAddress: string[]): void {
    this._openSocket('classic', {
      conversation_id: conversationId,
      workflow_name: workflowName,
      resume_run_id: resumeRunId,
      resume_address: resumeAddress,
    })
  }

  private _openSocket(mode: AgentMode, initPayload: Record<string, unknown>): void {
    this._running.set(true)
    this.ws = new WebSocket(wsUrl(mode))

    this.ws.onopen = () => {
      this.ws!.send(JSON.stringify(initPayload))
    }

    this.ws.onmessage = (ev) => {
      const event: AgentEvent = JSON.parse(ev.data)
      this._events$.next(event)
      if (event.type === 'done' || event.type === 'error' || event.type === 'stopped') {
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

  acceptPlan(planId: string, payload: { status: string; mode?: string; comment?: string; feedback?: string }): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'plan_accept', plan_id: planId, ...payload }))
    }
  }

  replyQuestion(questionId: string, reply: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'user_question_reply', question_id: questionId, reply }))
    }
  }

  setMode(mode: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'set_mode', mode }))
    }
  }

  abort(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'abort' }))
    }
    // Not flipped here: the backend needs a moment to actually unwind the in-flight stage and
    // emit its terminating event ('stopped', now that the workflow engine reliably produces one —
    // see custom_workflow.py's CancelledError handling). Flipping this immediately let the chat
    // input re-enable while the run panel behind it was still showing stale "running" state.
  }
}
