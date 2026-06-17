import { Injectable, inject, signal, computed } from '@angular/core'
import { BehaviorSubject, EMPTY, firstValueFrom, Observable, of, Subscription } from 'rxjs'
import { catchError, retry, shareReplay, switchMap, tap } from 'rxjs/operators'
import { throwError } from 'rxjs'
import {
  AgentToolMeta,
  AlwaysActiveToolMeta,
  AgentToolsResponse,
  ApiResponse,
  Conversation,
  ConversationSettings,
  DisplayMessage,
  Message,
  MessageForQuery,
  SystemPromptTemplate,
  ToolCallEntry,
} from '../types/message-types'
import { ApiService } from './api.service'
import { AgentService } from './agent.service'

// ---------------------------------------------------------------------------
// Shared observable helper — refreshable cached HTTP call
// ---------------------------------------------------------------------------

class RefreshableQuery<T> {
  private _refreshTrigger = new BehaviorSubject<void>(undefined)
  readonly obs$: Observable<T> = this._refreshTrigger.pipe(
    switchMap(() => this._factory()),
    shareReplay(1),
  )
  constructor(private _factory: () => Observable<T>) {}
  refresh() {
    this._refreshTrigger.next()
  }
}

// ---------------------------------------------------------------------------
// ChatService — single source of truth for all display state
// ---------------------------------------------------------------------------

@Injectable({ providedIn: 'root' })
export class ChatService {
  private api = inject(ApiService)
  private agentSvc = inject(AgentService)

  // -------------------------------------------------------------------------
  // Display state — the template reads only from here
  // -------------------------------------------------------------------------

  private _messages = signal<DisplayMessage[]>([])
  /** The full list of messages to render. Source: DB reload + live streaming. */
  public readonly messages = this._messages.asReadonly()

  private _isLoading = signal(false)
  /** True while a non-agentic HTTP stream is open. */
  public readonly isLoading = this._isLoading.asReadonly()

  private _isCompressing = signal(false)
  /** True while the post-agent compression call is in flight. */
  public readonly isCompressing = this._isCompressing.asReadonly()

  private _pipelineMode = signal(false)
  /** When true, the next agent run uses the pipeline orchestrator instead of the classic loop. */
  public readonly pipelineMode = this._pipelineMode.asReadonly()
  togglePipelineMode(): void {
    this._pipelineMode.update((v) => !v)
  }

  /** Delegates to AgentService so the component only needs to inject ChatService. */
  public readonly agentRunning = computed(() => this.agentSvc.running())

  private _promptTokens = signal(0)
  /** Latest prompt_tokens from the most recent agent iteration_end. */
  public readonly promptTokens = this._promptTokens.asReadonly()

  private _callingTool = signal<string | null>(null)
  /** Name of the tool whose arguments are currently being streamed; null when idle. */
  public readonly callingTool = this._callingTool.asReadonly()

  private _streamingToolCallArgs = signal<string>('')
  /** Raw JSON argument string accumulated from tool_call_chunk events; empty when idle. */
  public readonly streamingToolCallArgs = this._streamingToolCallArgs.asReadonly()

  /** Current streaming phase, derived from the last message in the list. */
  public readonly phase = computed<'thinking' | 'responding' | null>(() => {
    if (!this.agentSvc.running()) {
      return null
    }
    const msgs = this._messages()
    const last = msgs[msgs.length - 1]
    if (!last) {
      return null
    } else if (last.kind === 'thinking' && !last.done) {
      return 'thinking'
    } else if (last.kind === 'assistant' && last.streaming) {
      return 'responding'
    }
    return null
  })

  // -------------------------------------------------------------------------
  // Conversation list (sidebar)
  // -------------------------------------------------------------------------

  conversations = new RefreshableQuery(() =>
    this.api.get_conversations().pipe(retry({ count: 10, delay: 2000 })),
  )
  lastWorkingDirectory = new RefreshableQuery(() =>
    this.api.get_app_setting('last_working_directory').pipe(retry({ count: 10, delay: 2000 })),
  )

  // -------------------------------------------------------------------------
  // Active conversation metadata
  // -------------------------------------------------------------------------

  private _conversation: Conversation | undefined

  private _conversationId = signal<string | undefined>(undefined)
  public readonly currentConversationId = this._conversationId.asReadonly()

  private readonly DEFAULT_SETTINGS: ConversationSettings = {
    active_prompt_id: null,
    active_tool_names: [],
    working_directory: null,
    mode: 'standard',
  }

  private _conversationSettings = signal<ConversationSettings>(this.DEFAULT_SETTINGS)
  public readonly currentConversationSettings = this._conversationSettings.asReadonly()

  private _prompts = signal<SystemPromptTemplate[]>([])
  /** All system prompt templates — loaded once, shared across the service. */
  public readonly prompts = this._prompts.asReadonly()

  private _allToolNames = signal<string[]>([])
  /** Names of all registered agent tools — loaded once on startup. */
  public readonly allToolNames = this._allToolNames.asReadonly()

  private _allTools = signal<AgentToolMeta[]>([])
  /** Full tool metadata including token_count — loaded once on startup. */
  public readonly allTools = this._allTools.asReadonly()

  private _alwaysActiveTools = signal<AlwaysActiveToolMeta[]>([])
  /** Tools always injected by the framework (not user-toggleable). */
  public readonly alwaysActiveTools = this._alwaysActiveTools.asReadonly()

  private _toolFrameworkOverhead = signal<number>(0)
  /** Fixed token overhead for the tool-calling framework (independent of which tools are active). */
  public readonly toolFrameworkOverhead = this._toolFrameworkOverhead.asReadonly()

  private _stackingOverheadPerAdditionalTool = signal<number>(0)
  /** Extra tokens each additional tool (2nd, 3rd, ...) adds beyond its schema content. */
  public readonly stackingOverheadPerAdditionalTool =
    this._stackingOverheadPerAdditionalTool.asReadonly()

  // Active subscription to agentSvc.events$ — replaced on each new agent run
  private _agentEventSub: Subscription | null = null
  private _lastWorkingDirectory = signal<string | null>(null)

  constructor() {
    this.lastWorkingDirectory.obs$.subscribe((setting) => {
      if (setting.value !== null) {
        this._lastWorkingDirectory.set(setting.value)
      }
      if (this._conversationId() === undefined && setting.value !== null) {
        this._conversationSettings.set({
          ...this._conversationSettings(),
          working_directory: setting.value,
        })
      }
    })
    this.api.get_system_prompts().pipe(
      retry({ count: 10, delay: 2000 }),
      catchError(() => of([])),
    ).subscribe((p) => {
      this._prompts.set(p)
      // Patch pending new-chat settings with the default prompt if none is selected yet
      if (this._conversationId() === undefined) {
        const defaultId = p.find((prompt) => prompt.is_default)?.id ?? null
        if (defaultId !== null) {
          this._conversationSettings.set({
            ...this._conversationSettings(),
            active_prompt_id: defaultId,
          })
        }
      }
    })
    this.api.get_agent_tools().pipe(
      retry({ count: 10, delay: 2000 }),
      catchError(() => of(null)),
    ).subscribe((response: AgentToolsResponse | null) => {
      if (!response) {
        return
      }
      const names = response.tools.map((t) => t.name)
      this._allToolNames.set(names)
      this._allTools.set(response.tools)
      this._alwaysActiveTools.set(response.always_active_tools)
      this._toolFrameworkOverhead.set(response.framework_overhead)
      this._stackingOverheadPerAdditionalTool.set(response.stacking_overhead_per_additional_tool)
      // Patch the pending new-chat settings so tools are enabled by default
      if (this._conversationId() === undefined) {
        this._conversationSettings.set({
          ...this._conversationSettings(),
          active_tool_names: names,
        })
      }
    })
  }

  // -------------------------------------------------------------------------
  // Navigation
  // -------------------------------------------------------------------------

  startNewChat(): void {
    this._stopAgentAndClearSub()
    this._messages.set([])
    this._isLoading.set(false)
    this._promptTokens.set(0)
    this._conversation = undefined
    this._conversationId.set(undefined)
    const defaultPromptId = this._prompts().find((p) => p.is_default)?.id ?? null
    this._conversationSettings.set({
      active_prompt_id: defaultPromptId,
      active_tool_names: this._allToolNames(),
      working_directory: this._lastWorkingDirectory(),
      mode: 'standard',
    })
  }

  selectConversation(conversation: Conversation | undefined): void {
    this._stopAgentAndClearSub()
    if (conversation === undefined) {
      this._messages.set([])
      this._conversation = undefined
      this._conversationId.set(undefined)
      this._conversationSettings.set(this.DEFAULT_SETTINGS)
      return
    }
    this.api.get_conversation_messages(conversation.id).subscribe((dbMessages) => {
      this._messages.set(this._fromDbMessages(dbMessages))
      this._conversation = conversation
      this._conversationId.set(conversation.id)
      const parsed = conversation.settings ? JSON.parse(conversation.settings) : {}
      const settings: ConversationSettings = { ...this.DEFAULT_SETTINGS, ...parsed }
      this._conversationSettings.set(settings)
      if (settings.working_directory != null) {
        this._lastWorkingDirectory.set(settings.working_directory)
        this.api.put_app_setting('last_working_directory', settings.working_directory).subscribe()
      }
      const lastWithTokens = [...dbMessages].reverse().find((m) => m.token_count != null)
      this._promptTokens.set(lastWithTokens?.token_count ?? 0)
    })
  }

  selectConversationById(id: string): void {
    this.api.get_conversation(id).pipe(
      catchError(() => EMPTY),
    ).subscribe((conv) => this.selectConversation(conv))
  }

  async editUserMessage(msgId: string, newContent: string): Promise<void> {
    const convId = this._conversationId()
    if (!convId) {
      return
    }

    const { id: newMsgId } = await firstValueFrom(this.api.branch_message(msgId, newContent))
    await this._reloadFromDb()

    const settings = this._conversationSettings()
    this._subscribeToAgentEvents(newMsgId)
    this.agentSvc.start(newContent, convId, newMsgId, this._pipelineMode() ? 'pipeline' : 'classic')
  }

  async deleteMessage(msgId: string, subtree: boolean): Promise<void> {
    const convId = this._conversationId()
    if (!convId) {
      return
    }
    await firstValueFrom(this.api.delete_message(convId, msgId, subtree))
    await this._reloadFromDb()
    const msgs = this._messages()
    const last = [...msgs]
      .reverse()
      .find(
        (m) =>
          (m.kind === 'user' || m.kind === 'assistant' || m.kind === 'tool_result') &&
          m.token_count != null,
      )
    const tokenCount =
      last && (last.kind === 'user' || last.kind === 'assistant' || last.kind === 'tool_result')
        ? last.token_count
        : null
    this._promptTokens.set(tokenCount ?? 0)
  }

  async navigateSibling(siblingId: string): Promise<void> {
    const convId = this._conversationId()
    if (!convId) {
      return
    }
    const { messages } = await firstValueFrom(this.api.set_active_branch(convId, siblingId))
    this._messages.set(this._fromDbMessages(messages))
    const last = [...messages].reverse().find((m) => m.token_count != null)
    this._promptTokens.set(last?.token_count ?? 0)
  }

  // -------------------------------------------------------------------------
  // Agentic chat
  // -------------------------------------------------------------------------

  startAgentRun(input: string, imageIds: string[] = [], workflowName?: string): void {
    const effectiveInput = input !== '' ? input : workflowName !== undefined ? `/${workflowName}` : input
    const userMsg: DisplayMessage = { kind: 'user', id: crypto.randomUUID(), content: effectiveInput }
    this._messages.update((msgs) => [...msgs, userMsg])
    this._promptTokens.set(0)

    const persist = !this._conversation
      ? this._createConversation(userMsg, imageIds)
      : this._addUserMessageToDb(userMsg, imageIds)

    persist.then(() => {
      const convId = this._conversationId()
      this._subscribeToAgentEvents(userMsg.id)
      const agentMode = workflowName !== undefined ? 'classic' : this._pipelineMode() ? 'pipeline' : 'classic'
      this.agentSvc.start(input, convId, userMsg.id, agentMode, workflowName)
    })
  }

  uploadImage(file: File) {
    return this.api.upload_image(file)
  }

  confirmTool(toolId: string, approved: boolean, reason?: string): void {
    this.agentSvc.confirm(toolId, approved, reason)
    this._messages.update((msgs) =>
      msgs.map((m) =>
        m.kind === 'tool_confirm' && m.tool_id === toolId ? { ...m, confirmed: approved } : m,
      ),
    )
  }

  abortAgent(): void {
    this.agentSvc.abort()
  }

  acceptPlan(planId: string, payload: { status: string; mode?: string; comment?: string; feedback?: string }): void {
    this.agentSvc.acceptPlan(planId, payload)
    const resolution = payload.status === 'accepted'
      ? `Accepted — executing in ${payload.mode ? payload.mode.charAt(0).toUpperCase() + payload.mode.slice(1) : 'Standard'} mode`
      : 'Feedback sent — waiting for revised plan'
    this._messages.update((msgs) =>
      msgs.map((m) =>
        m.kind === 'plan_proposal' && m.plan_id === planId
          ? { ...m, resolved: true, resolution }
          : m,
      ),
    )
  }

  replyQuestion(questionId: string, reply: string): void {
    this.agentSvc.replyQuestion(questionId, reply)
    this._messages.update((msgs) =>
      msgs.map((m) =>
        m.kind === 'agent_question' && m.question_id === questionId
          ? { ...m, resolved: true }
          : m,
      ),
    )
  }

  // -------------------------------------------------------------------------
  // Settings / conversation management
  // -------------------------------------------------------------------------

  updateConversationSettings(settings: ConversationSettings): Observable<unknown> {
    const previousMode = this._conversationSettings()?.mode
    this._conversationSettings.set(settings)
    if (this.agentSvc.running() && settings.mode !== previousMode) {
      this.agentSvc.setMode(settings.mode)
    }
    const id = this._conversationId()
    if (!id) {
      if (settings.working_directory !== null) {
        return this.api
          .put_app_setting('last_working_directory', settings.working_directory)
          .pipe(tap(() => this.lastWorkingDirectory.refresh()))
      }
      return new Observable((s) => s.complete())
    }
    return this.api.put_conversation_settings(id, settings).pipe(
      tap(() => {
        this.conversations.refresh()
        if (settings.working_directory !== null) {
          this.lastWorkingDirectory.refresh()
        }
      }),
    )
  }

  reloadPrompts(): void {
    this.api.get_system_prompts().subscribe((p) => this._prompts.set(p))
  }

  async deleteConversation(conv: Conversation): Promise<void> {
    await firstValueFrom(this.api.delete_conversation(conv.id))
    this.conversations.refresh()
    if (conv.id === this._conversation?.id) {
      console.log('consol')
      this.startNewChat()
    }
  }

  // -------------------------------------------------------------------------
  // Private: agent event accumulation
  // -------------------------------------------------------------------------

  private _subscribeToAgentEvents(userMessageId: string): void {
    this._agentEventSub?.unsubscribe()

    const conv = this._conversation!

    let idOfCurrentlyStreamingThinkingMessage: string | null = null
    let lastThinkingMessageId: string | null = null
    let pendingThinkingContent = ''

    let idOfCurrentlyStreamingAssistantMessage: string | null = null
    let pendingContentText = ''
    let pendingToolCalls: ToolCallEntry[] = []
    // Accumulates tool calls as they stream (tool_call_start + tool_call_chunk events).
    // Finalized into pendingToolCalls on generation_end before tool execution starts.
    let streamingToolCallsAcc: Map<string, { id: string; name: string; argsStr: string }> = new Map()

    let patchedUserMessage = false
    let lastKnownCtxTokens = 0
    // Set on generation_end; consumed and cleared in stopStreamingTheAssistantMessageSaveItAndClearIt.
    let capturedGenerationCtxTokens: number | null = null

    let saveQueue: Promise<void> = Promise.resolve()
    const enqueue = (fn: () => Promise<unknown>) => {
      saveQueue = saveQueue.then(async () => {
        const MAX_ATTEMPTS = 4
        for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
          try {
            await fn()
            return
          } catch (err) {
            if (attempt === MAX_ATTEMPTS - 1) {
              console.error('Save failed after retries:', err)
              return
            }
            await new Promise<void>((resolve) => setTimeout(resolve, 300 * (attempt + 1)))
          }
        }
      })
    }

    const saveAssistant = (id: string, content: string, thinking: string, toolCalls: ToolCallEntry[], tokenCount: number | null = null) => {
      enqueue(async () => {
        let tokenDelta: number | null = null
        if (tokenCount !== null) {
          const currentMsgs = this._messages()
          const idx = currentMsgs.findIndex((m) => m.id === id)
          if (idx >= 0) {
            const prevCount = this._findCumulativeTokenCountOfClosestPrecedingMessageThatHasOne(currentMsgs, idx)
            tokenDelta = prevCount !== null ? tokenCount - prevCount : null
            this._messages.update((msgs) =>
              msgs.map((m) =>
                m.id === id && m.kind === 'assistant' ? { ...m, token_count: tokenCount, token_delta: tokenDelta } : m,
              ),
            )
          }
        }
        await firstValueFrom(
          this.api.post_message(conv.id, {
            id,
            role: 'assistant',
            content,
            thinking: thinking !== '' ? thinking : undefined,
            tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
            token_count: tokenCount ?? undefined,
            token_delta: tokenDelta ?? undefined,
          }),
        )
      })
    }

    const markThinkingMessageAsDoneAndClearIt = () => {
      if (!idOfCurrentlyStreamingThinkingMessage) return
      this._messages.update((msgs) =>
        msgs.map((m) =>
          m.id === idOfCurrentlyStreamingThinkingMessage && m.kind === 'thinking'
            ? { ...m, done: true }
            : m,
        ),
      )
      idOfCurrentlyStreamingThinkingMessage = null
    }

    const stopStreamingTheAssistantMessageSaveItAndClearIt = (tokenCount: number | null = null) => {
      if (!idOfCurrentlyStreamingAssistantMessage) return
      this._messages.update((msgs) =>
        msgs.map((m) =>
          m.id === idOfCurrentlyStreamingAssistantMessage && m.kind === 'assistant'
            ? { ...m, streaming: false }
            : m,
        ),
      )
      const id = idOfCurrentlyStreamingAssistantMessage,
        content = pendingContentText,
        thinking = pendingThinkingContent
      idOfCurrentlyStreamingAssistantMessage = null
      pendingContentText = ''
      pendingThinkingContent = ''
      const toolCalls = pendingToolCalls
      pendingToolCalls = []
      // Patch the in-memory message with tool_calls so the UI shows them immediately
      this._messages.update((msgs) =>
        msgs.map((m) =>
          m.id === id && m.kind === 'assistant' ? { ...m, tool_calls: toolCalls.length > 0 ? toolCalls : undefined } : m,
        ),
      )
      saveAssistant(id, content, thinking, toolCalls, tokenCount)
    }

    this._agentEventSub = this.agentSvc.events$.subscribe((event) => {
      if (event._pipeline_stage !== undefined && event.type !== 'tool_confirm') {
        return
      }
      if (event.type === 'thinking' && event.content) {
        pendingThinkingContent += event.content
        if (idOfCurrentlyStreamingThinkingMessage) {
          this._messages.update((msgs) =>
            msgs.map((m) =>
              m.id === idOfCurrentlyStreamingThinkingMessage && m.kind === 'thinking'
                ? { ...m, content: m.content + event.content }
                : m,
            ),
          )
        } else {
          idOfCurrentlyStreamingThinkingMessage = crypto.randomUUID()
          lastThinkingMessageId = idOfCurrentlyStreamingThinkingMessage
          this._messages.update((msgs) => [
            ...msgs,
            {
              kind: 'thinking',
              id: idOfCurrentlyStreamingThinkingMessage!,
              content: event.content!,
              done: false,
            },
          ])
        }
      } else if (event.type === 'content' && event.content) {
        pendingContentText += event.content
        if (idOfCurrentlyStreamingAssistantMessage) {
          this._messages.update((msgs) =>
            msgs.map((m) =>
              m.id === idOfCurrentlyStreamingAssistantMessage && m.kind === 'assistant'
                ? { ...m, content: m.content + event.content }
                : m,
            ),
          )
        } else {
          idOfCurrentlyStreamingAssistantMessage = crypto.randomUUID()
          markThinkingMessageAsDoneAndClearIt()
          this._messages.update((msgs) => [
            ...msgs,
            {
              kind: 'assistant',
              id: idOfCurrentlyStreamingAssistantMessage!,
              content: event.content!,
              streaming: true,
            },
          ])
        }
      } else if (event.type === 'tool_call_start') {
        this._callingTool.set(event.tool_name)
        this._streamingToolCallArgs.set('')
        streamingToolCallsAcc.set(event.tool_id, { id: event.tool_id, name: event.tool_name, argsStr: '' })
      } else if (event.type === 'tool_call_chunk') {
        this._streamingToolCallArgs.update((s) => s + event.chunk)
        const accEntry = streamingToolCallsAcc.get(event.tool_id)
        if (accEntry !== undefined) {
          accEntry.argsStr += event.chunk
        }
      } else if (event.type === 'tool_call') {
        // Update with finalized parsed args (for streamed calls) or add new entry (for recovered calls).
        const existingEntry = streamingToolCallsAcc.get(event.tool_id)
        if (existingEntry !== undefined) {
          existingEntry.argsStr = JSON.stringify(event.arguments)
        } else {
          streamingToolCallsAcc.set(event.tool_id, { id: event.tool_id, name: event.tool_name, argsStr: JSON.stringify(event.arguments) })
        }
      } else if (event.type === 'tool_confirm') {
        this._messages.update((msgs) => {
          const updated = event.evaluator_reason
            ? msgs.map((m) =>
                m.kind === 'tool_evaluating' && m.tool_id === event.tool_id
                  ? { ...m, verdict: 'dangerous' as const, reason: event.evaluator_reason }
                  : m,
              )
            : msgs
          return [
            ...updated,
            {
              kind: 'tool_confirm' as const,
              id: event.tool_id!,
              tool_id: event.tool_id!,
              tool_name: event.tool_name ?? '',
              args: event.arguments ?? {},
              preview: event.preview ?? '',
              diff_lines: event.diff_lines,
              confirmed: null,
            },
          ]
        })
      } else if (event.type === 'tool_evaluating') {
        this._messages.update((msgs) => [
          ...msgs,
          { kind: 'tool_evaluating' as const, id: event.tool_id, tool_id: event.tool_id, tool_name: event.tool_name },
        ])
      } else if (event.type === 'tool_auto_approved') {
        this._messages.update((msgs) =>
          msgs.map((m) => m.kind === 'tool_evaluating' && m.tool_id === event.tool_id ? { ...m, verdict: 'safe' as const } : m),
        )
      } else if (event.type === 'tool_result') {
        this._callingTool.set(null)
        this._streamingToolCallArgs.set('')
        // For recovered tool calls (not streamed during generation), streamingToolCallsAcc will
        // have entries here. Finalize and attach them before adding the result message.
        if (streamingToolCallsAcc.size > 0) {
          const thinkingIdBeforeResult = idOfCurrentlyStreamingThinkingMessage
          for (const entry of streamingToolCallsAcc.values()) {
            let args: Record<string, unknown> = {}
            try {
              args = JSON.parse(entry.argsStr)
            } catch { /* leave args empty */ }
            pendingToolCalls.push({ id: entry.id, name: entry.name, args })
          }
          streamingToolCallsAcc = new Map()
          const tokenCount = capturedGenerationCtxTokens
          capturedGenerationCtxTokens = null
          stopStreamingTheAssistantMessageSaveItAndClearIt(tokenCount)
          if (pendingToolCalls.length > 0) {
            const orphanedToolCalls = pendingToolCalls
            pendingToolCalls = []
            if (thinkingIdBeforeResult !== null) {
              this._messages.update((msgs) =>
                msgs.map((m) =>
                  m.id === thinkingIdBeforeResult && m.kind === 'thinking'
                    ? { ...m, tool_calls: orphanedToolCalls }
                    : m,
                ),
              )
            }
            if (pendingThinkingContent) {
              const thinking = pendingThinkingContent
              pendingThinkingContent = ''
              saveAssistant(crypto.randomUUID(), '', thinking, orphanedToolCalls, tokenCount)
            }
          } else if (pendingThinkingContent) {
            const thinking = pendingThinkingContent
            pendingThinkingContent = ''
            saveAssistant(crypto.randomUUID(), '', thinking, [], tokenCount)
          }
        }
        markThinkingMessageAsDoneAndClearIt()
        const resultId = `result-${event.tool_id}`
        const resultContent = event.content ?? ''
        const logMessage = event.log_message ?? null
        const toolTokenCount = event.ctx_tokens ?? lastKnownCtxTokens
        this._messages.update((msgs) => [
          ...msgs,
          {
            kind: 'tool_result',
            id: resultId,
            tool_name: event.tool_name ?? '',
            log_message: logMessage,
            content: resultContent,
            token_count: toolTokenCount,
          },
        ])
        // token_count is included in the POST; delta computed in the callback after the
        // assistant POST has run and updated in-memory token_count on the preceding message.
        enqueue(async () => {
          const currentMsgs = this._messages()
          const idx = currentMsgs.findIndex((m) => m.id === resultId)
          const prevCount = this._findCumulativeTokenCountOfClosestPrecedingMessageThatHasOne(currentMsgs, idx)
          const tokenDelta = prevCount !== null ? toolTokenCount - prevCount : null
          this._messages.update((msgs) =>
            msgs.map((m) =>
              m.id === resultId ? { ...m, token_delta: tokenDelta } : m,
            ),
          )
          await firstValueFrom(
            this.api.post_message(conv.id, {
              id: resultId,
              role: 'tool',
              content: resultContent,
              log_message: logMessage,
              token_count: toolTokenCount,
              token_delta: tokenDelta ?? undefined,
            }),
          )
        })
      } else if (event.type === 'generation_end') {
        if (streamingToolCallsAcc.size > 0) {
          // Finalize all streamed tool calls into pendingToolCalls before execution starts.
          for (const entry of streamingToolCallsAcc.values()) {
            let args: Record<string, unknown> = {}
            try {
              args = JSON.parse(entry.argsStr)
            } catch { /* leave args empty if JSON is malformed */ }
            pendingToolCalls.push({ id: entry.id, name: entry.name, args })
          }
          streamingToolCallsAcc = new Map()
          // Finalize the assistant message now — all tool calls are known before execution starts.
          const thinkingIdAtGenEnd = idOfCurrentlyStreamingThinkingMessage
          stopStreamingTheAssistantMessageSaveItAndClearIt(event.ctx_tokens)
          // Handle thinking-only messages (no content) that had tool calls.
          if (pendingToolCalls.length > 0) {
            const orphanedToolCalls = pendingToolCalls
            pendingToolCalls = []
            if (thinkingIdAtGenEnd !== null) {
              this._messages.update((msgs) =>
                msgs.map((m) =>
                  m.id === thinkingIdAtGenEnd && m.kind === 'thinking'
                    ? { ...m, tool_calls: orphanedToolCalls }
                    : m,
                ),
              )
            }
            if (pendingThinkingContent) {
              const thinking = pendingThinkingContent
              pendingThinkingContent = ''
              saveAssistant(crypto.randomUUID(), '', thinking, orphanedToolCalls, event.ctx_tokens)
            }
          }
        } else {
          capturedGenerationCtxTokens = event.ctx_tokens
        }
      } else if (event.type === 'iteration_end') {
        const tokens = event.prompt_tokens ?? 0
        this._promptTokens.set(tokens)
        markThinkingMessageAsDoneAndClearIt()
        const tokenCountForIteration = capturedGenerationCtxTokens
        capturedGenerationCtxTokens = null
        stopStreamingTheAssistantMessageSaveItAndClearIt(tokenCountForIteration)
        if (pendingThinkingContent) {
          const thinking = pendingThinkingContent
          pendingThinkingContent = ''
          saveAssistant(crypto.randomUUID(), '', thinking, [], tokenCountForIteration)
        }
      } else if (event.type === 'plan_proposal') {
        this._messages.update((msgs) => [
          ...msgs,
          {
            kind: 'plan_proposal' as const,
            id: crypto.randomUUID(),
            plan_id: event.plan_id,
            plan: event.plan,
            resolved: false,
          },
        ])
      } else if (event.type === 'agent_question') {
        this._messages.update((msgs) => [
          ...msgs,
          {
            kind: 'agent_question' as const,
            id: crypto.randomUUID(),
            question_id: event.question_id,
            question: event.question,
            options: event.options,
            resolved: false,
          },
        ])
      } else if (event.type === 'mode_changed') {
        const newSettings = { ...this._conversationSettings(), mode: event.mode }
        this.updateConversationSettings(newSettings).subscribe()
      } else if (event.type === 'ctx_update') {
        lastKnownCtxTokens = event.ctx_tokens ?? 0
        this._promptTokens.set(event.ctx_tokens ?? 0)
        if (!patchedUserMessage) {
          patchedUserMessage = true
          const capturedTokens = lastKnownCtxTokens
          enqueue(async () => {
            const currentMsgs = this._messages()
            const idx = currentMsgs.findIndex((m) => m.id === userMessageId)
            const prevCount = this._findCumulativeTokenCountOfClosestPrecedingMessageThatHasOne(currentMsgs, idx)
            const tokenDelta = prevCount !== null ? capturedTokens - prevCount : null
            await firstValueFrom(this.api.patch_message_token_count(userMessageId, capturedTokens, tokenDelta))
            this._messages.update((msgs) =>
              msgs.map((m) =>
                m.id === userMessageId ? { ...m, token_count: capturedTokens, token_delta: tokenDelta } : m,
              ),
            )
          })
        }
      } else if (event.type === 'compressing') {
        const convId = this._conversationId()
        if (convId) {
          enqueue(async () => {
            this._isCompressing.set(true)
            try {
              const result = await firstValueFrom(this.api.compress_conversation(convId, true, true))
              if (result.ctx_tokens != null) {
                this._promptTokens.set(result.ctx_tokens)
              }
              // Reload so the user sees compressed summaries; safe because the backend is
              // suspended and the save queue has already flushed all prior DB writes.
              await this._reloadFromDb()
            } finally {
              this._isCompressing.set(false)
            }
            this.agentSvc.compressionDone(convId)
          })
        }
      } else if (event.type === 'done' || event.type === 'error') {
        this._callingTool.set(null)
        this._streamingToolCallArgs.set('')
        this._messages.update((msgs) => msgs.filter((m) => m.kind !== 'tool_confirm' && m.kind !== 'tool_evaluating'))
        if (event.type === 'error') {
          const errorText = event.message ?? 'An error occurred'
          // Tool results are already patched immediately on arrival — just reload and show error.
          enqueue(async () => {
            await this._reloadFromDb()
            this._messages.update((msgs) => [
              ...msgs,
              { kind: 'error', id: crypto.randomUUID(), message: errorText },
            ])
          })
        } else {
          if (event.finished_without_response === true) {
            const degenerateId = lastThinkingMessageId !== null ? lastThinkingMessageId : crypto.randomUUID()
            const thinkingContent = pendingThinkingContent !== '' ? pendingThinkingContent : undefined
            pendingThinkingContent = ''
            if (lastThinkingMessageId !== null) {
              this._messages.update((msgs) =>
                msgs.map((m) =>
                  m.id === lastThinkingMessageId && m.kind === 'thinking'
                    ? { ...m, is_degenerate: true }
                    : m,
                ),
              )
            }
            enqueue(() =>
              firstValueFrom(
                this.api.post_message(conv.id, {
                  id: degenerateId,
                  role: 'assistant',
                  content: '',
                  thinking: thinkingContent,
                  is_degenerate: true,
                }),
              ),
            )
          }
          enqueue(() => this._compressConversation())
          // Reload from DB to get has_children and sibling metadata, which are only computed
          // server-side and are not available on display messages created during the agent run.
          enqueue(() => this._reloadFromDb())
        }
        this._agentEventSub?.unsubscribe()
        this._agentEventSub = null
      }
    })
  }

  private async _compressConversation(): Promise<void> {
    const id = this._conversationId()
    if (!id) {
      return
    }
    this._isCompressing.set(true)
    try {
      await firstValueFrom(this.api.compress_conversation(id))
    } finally {
      this._isCompressing.set(false)
    }
  }

  private async _reloadFromDb(): Promise<void> {
    const id = this._conversationId()
    if (!id) {
      return
    }
    const dbMessages = await firstValueFrom(this.api.get_conversation_messages(id))
    this._messages.set(this._fromDbMessages(dbMessages))
  }

  // -------------------------------------------------------------------------
  // Private: type conversions between DisplayMessage and DB Message
  // -------------------------------------------------------------------------

  private _findCumulativeTokenCountOfClosestPrecedingMessageThatHasOne(
    msgs: DisplayMessage[],
    idx: number,
  ): number | null {
    for (let i = idx - 1; i >= 0; i--) {
      const m = msgs[i]
      if ('token_count' in m && m.token_count != null) {
        return m.token_count
      }
    }
    return null
  }

  /** Convert DB messages (from GET /messages) into DisplayMessages for rendering. */
  private _fromDbMessages(dbMessages: Message[]): DisplayMessage[] {
    const result: DisplayMessage[] = []
    for (const m of dbMessages) {
      const siblingMeta = {
        sibling_count: m.sibling_count,
        sibling_index: m.sibling_index,
        prev_sibling_id: m.prev_sibling_id,
        next_sibling_id: m.next_sibling_id,
        has_children: m.has_children,
      }
      if (m.role === 'user') {
        result.push({
          kind: 'user',
          id: m.id,
          content: m.content,
          images: m.images,
          token_count: m.token_count,
          token_delta: m.token_delta,
          context_excluded: m.context_excluded,
          ...siblingMeta,
        })
      } else if (m.role === 'assistant') {
        if (m.content) {
          result.push({
            kind: 'assistant',
            id: m.id,
            content: m.content,
            thinking: m.thinking ?? undefined,
            tool_calls: m.tool_calls ?? undefined,
            token_count: m.token_count,
            token_delta: m.token_delta,
            context_excluded: m.context_excluded,
            ...siblingMeta,
          })
        } else if (m.thinking) {
          // Thinking-only message (no content) — show as a collapsed thinking block
          result.push({
            kind: 'thinking',
            id: m.id,
            content: m.thinking,
            done: true,
            tool_calls: m.tool_calls ?? undefined,
            is_degenerate: m.is_degenerate,
            ...siblingMeta,
          })
        } else if (m.is_degenerate) {
          // Degenerate stop with no thinking — show as an empty collapsed thinking block
          result.push({
            kind: 'thinking',
            id: m.id,
            content: '',
            done: true,
            is_degenerate: true,
            ...siblingMeta,
          })
        }
      } else if (m.role === 'tool') {
        result.push({
          kind: 'tool_result',
          id: m.id,
          tool_name: (() => { try { return JSON.parse(m.content).tool ?? '' } catch { return '' } })(),
          log_message: m.log_message ?? null,
          content: m.content,
          compressed_summary: m.compressed_summary ?? null,
          compression_label: m.compression_label ?? null,
          compressed_token_count: m.compressed_token_count ?? null,
          token_count: m.token_count,
          token_delta: m.token_delta,
          context_excluded: m.context_excluded,
          ...siblingMeta,
        })
      }
    }
    return result
  }

  /** Convert DisplayMessages into the flat role/content format the chat API expects. */
  private _toOllamaMessages(messages: DisplayMessage[]): MessageForQuery[] {
    const result: MessageForQuery[] = []
    for (const m of messages) {
      if (m.kind === 'user') {
        result.push({ role: 'user', content: m.content })
      } else if (m.kind === 'assistant' && m.content) {
        result.push({ role: 'assistant', content: m.content })
      } else if (m.kind === 'tool_result') {
        result.push({ role: 'tool', content: m.content })
      }
      // thinking, tool_confirm, streaming assistant with no content → not sent to Ollama
    }
    return result
  }

  // -------------------------------------------------------------------------
  // Private: DB persistence helpers
  // -------------------------------------------------------------------------

  private async _createConversation(userMsg: { id: string; content: string }, imageIds: string[] = []): Promise<void> {
    const title = userMsg.content.substring(0, 20)
    const conversation = await firstValueFrom(this.api.post_conversation(title))
    this._conversation = conversation
    this._conversationId.set(conversation.id)

    // Persist whatever the user configured before sending (tools, prompt, agentic mode)
    const pendingSettings = this._conversationSettings()
    if (pendingSettings) {
      await firstValueFrom(this.api.put_conversation_settings(conversation.id, pendingSettings))
    }

    this.conversations.refresh()

    await firstValueFrom(
      this.api.post_message(conversation.id, {
        id: userMsg.id,
        role: 'user',
        content: userMsg.content,
        ...(imageIds.length > 0 ? { image_ids: imageIds } : {}),
      }),
    )
  }

  private async _addUserMessageToDb(userMsg: { id: string; content: string }, imageIds: string[] = []): Promise<void> {
    if (!this._conversation) {
      return
    }
    await firstValueFrom(
      this.api.post_message(this._conversation.id, {
        id: userMsg.id,
        role: 'user',
        content: userMsg.content,
        ...(imageIds.length > 0 ? { image_ids: imageIds } : {}),
      }),
    )
  }

  private _prependSystemPrompt(messages: MessageForQuery[]): MessageForQuery[] {
    const promptId = this._conversationSettings().active_prompt_id
    if (promptId === null || promptId === undefined) {
      return messages
    }
    const prompt = this._prompts().find((p) => p.id === promptId)
    if (!prompt) {
      return messages
    }
    return [{ role: 'system', content: prompt.content }, ...messages]
  }

  private _stopAgentAndClearSub(): void {
    if (this.agentSvc.running()) {
      this.agentSvc.abort()
    }
    this._agentEventSub?.unsubscribe()
    this._agentEventSub = null
  }

  async transcribe(blob: Blob, language: string | null = 'fr'): Promise<string> {
    const res = await firstValueFrom(this.api.post_transcribe(blob, language))
    return res.text
  }
}
