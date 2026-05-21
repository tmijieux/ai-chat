import { Injectable, inject, signal, computed } from '@angular/core'
import { BehaviorSubject, firstValueFrom, Observable, Subscription } from 'rxjs'
import { catchError, shareReplay, switchMap, tap } from 'rxjs/operators'
import { throwError } from 'rxjs'
import {
  AgentToolMeta,
  AgentToolsResponse,
  ApiResponse,
  Conversation,
  ConversationSettings,
  DisplayMessage,
  Message,
  MessageForQuery,
  SystemPromptTemplate,
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

  /** Delegates to AgentService so the component only needs to inject ChatService. */
  public readonly agentRunning = computed(() => this.agentSvc.running())

  private _promptTokens = signal(0)
  /** Latest prompt_tokens from the most recent agent iteration_end. */
  public readonly promptTokens = this._promptTokens.asReadonly()

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

  conversations = new RefreshableQuery(() => this.api.get_conversations())

  // -------------------------------------------------------------------------
  // Active conversation metadata
  // -------------------------------------------------------------------------

  private _conversation: Conversation | undefined

  private _conversationId = signal<string | undefined>(undefined)
  public readonly currentConversationId = this._conversationId.asReadonly()

  private readonly DEFAULT_SETTINGS: ConversationSettings = {
    agentic_mode: true,
    active_prompt_id: null,
    active_tool_names: [],
    working_directory: null,
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

  private _toolFrameworkOverhead = signal<number>(0)
  /** Fixed token overhead for the tool-calling framework (independent of which tools are active). */
  public readonly toolFrameworkOverhead = this._toolFrameworkOverhead.asReadonly()

  // Active subscription to agentSvc.events$ — replaced on each new agent run
  private _agentEventSub: Subscription | null = null

  constructor() {
    this.api.get_system_prompts().subscribe((p) => {
      this._prompts.set(p)
      // Patch pending new-chat settings with the default prompt if none is selected yet
      if (this._conversationId() === undefined) {
        const defaultId = p.find((prompt) => prompt.is_default === 1)?.id ?? null
        if (defaultId !== null) {
          this._conversationSettings.set({
            ...this._conversationSettings(),
            active_prompt_id: defaultId,
          })
        }
      }
    })
    this.api.get_agent_tools().subscribe((response: AgentToolsResponse) => {
      const names = response.tools.map((t) => t.name)
      this._allToolNames.set(names)
      this._allTools.set(response.tools)
      this._toolFrameworkOverhead.set(response.framework_overhead)
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
    const defaultPromptId = this._prompts().find((p) => p.is_default === 1)?.id ?? null
    const lastWorkingDir = this._conversationSettings().working_directory
    this._conversationSettings.set({
      agentic_mode: true,
      active_prompt_id: defaultPromptId,
      active_tool_names: this._allToolNames(),
      working_directory: lastWorkingDir,
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
      this._conversationSettings.set(
        conversation.settings ? JSON.parse(conversation.settings) : this.DEFAULT_SETTINGS,
      )
      const lastWithTokens = [...dbMessages].reverse().find((m) => m.token_count != null)
      this._promptTokens.set(lastWithTokens?.token_count ?? 0)
    })
  }

  // -------------------------------------------------------------------------
  // Non-agentic chat
  // -------------------------------------------------------------------------

  sendMessage(input: string): void {
    const ollamaMessages: MessageForQuery[] = [
      ...this._prependSystemPrompt(this._toOllamaMessages(this._messages())),
      { role: 'user', content: input },
    ]

    const userMsg: DisplayMessage = { kind: 'user', id: crypto.randomUUID(), content: input }
    this._messages.update((msgs) => [...msgs, userMsg])

    const persist = !this._conversation
      ? this._createConversation(userMsg)
      : this._addUserMessageToDb(userMsg)

    persist.then(() => this._streamNonAgenticResponse(ollamaMessages))
  }

  async editUserMessage(msgId: string, newContent: string): Promise<void> {
    const convId = this._conversationId()
    if (!convId) {
      return
    }

    const { id: newMsgId } = await firstValueFrom(this.api.branch_message(msgId, newContent))
    await this._reloadFromDb()

    const settings = this._conversationSettings()
    if (settings?.agentic_mode) {
      this._subscribeToAgentEvents()
      this.agentSvc.start(newContent, convId, newMsgId)
    } else {
      const ollamaMessages = this._prependSystemPrompt(this._toOllamaMessages(this._messages()))
      this._streamNonAgenticResponse(ollamaMessages)
    }
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

  startAgentRun(input: string): void {
    const userMsg: DisplayMessage = { kind: 'user', id: crypto.randomUUID(), content: input }
    this._messages.update((msgs) => [...msgs, userMsg])
    this._promptTokens.set(0)

    const persist = !this._conversation
      ? this._createConversation(userMsg)
      : this._addUserMessageToDb(userMsg)

    persist.then(() => {
      const convId = this._conversationId()
      this._subscribeToAgentEvents()
      this.agentSvc.start(input, convId)
    })
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

  // -------------------------------------------------------------------------
  // Settings / conversation management
  // -------------------------------------------------------------------------

  async setAgenticMode(enabled: boolean): Promise<void> {
    const current = this._conversationSettings()
    const updated = { ...current, agentic_mode: enabled }
    this._conversationSettings.set(updated)
    const id = this._conversationId()
    if (!id) {
      return
    }
    await firstValueFrom(this.api.put_conversation_settings(id, updated))
  }

  updateConversationSettings(settings: ConversationSettings): Observable<unknown> {
    this._conversationSettings.set(settings)
    const id = this._conversationId()
    if (!id) {
      return new Observable((s) => s.complete())
    }
    return this.api
      .put_conversation_settings(id, settings)
      .pipe(tap(() => this.conversations.refresh()))
  }

  reloadPrompts(): void {
    this.api.get_system_prompts().subscribe((p) => this._prompts.set(p))
  }

  async deleteConversation(conv: Conversation): Promise<void> {
    await firstValueFrom(this.api.delete_conversation(conv.id))
    this.conversations.refresh()
    if (conv.id === this._conversation?.id) {
      this.startNewChat()
    }
  }

  // -------------------------------------------------------------------------
  // Private: non-agentic streaming
  // -------------------------------------------------------------------------

  private _streamNonAgenticResponse(ollamaMessages: MessageForQuery[]): void {
    this._isLoading.set(true)
    const loadingMsg: DisplayMessage = {
      kind: 'assistant',
      id: 'streaming',
      content: '',
      streaming: true,
    }
    this._messages.update((msgs) => [...msgs, loadingMsg])

    this.api
      .generate_chat_response(ollamaMessages)
      .pipe(
        tap((response) => {
          if (response.type !== 3) {
            return
          }
          const chunks: ApiResponse[] = (response.partialText ?? '')
            .split('\n')
            .filter((line) => line.endsWith('}'))
            .map((line) => JSON.parse(line))

          const thinking = chunks.map((c) => c.message.thinking ?? '').join('')
          const content = chunks.map((c) => c.message.content).join('')

          this._messages.update((msgs) =>
            msgs.map((m) =>
              m.id === 'streaming' && m.kind === 'assistant'
                ? { ...m, content, thinking: thinking || undefined }
                : m,
            ),
          )

          const lastChunk = chunks[chunks.length - 1]
          if (lastChunk?.done) {
            this._isLoading.set(false)
            if (this._conversation) {
              const conv = this._conversation
              const finalId = crypto.randomUUID()
              this._messages.update((msgs) =>
                msgs.map((m) =>
                  m.id === 'streaming' && m.kind === 'assistant'
                    ? { ...m, id: finalId, streaming: false }
                    : m,
                ),
              )
              ;(async () => {
                const savedMsg: Message = {
                  id: finalId,
                  role: 'assistant',
                  content,
                  thinking: thinking || undefined,
                }
                await firstValueFrom(this.api.post_message(conv.id, savedMsg))
                await this._computeTokenCountForLastMessage()
              })()
            }
          }
        }),
        catchError((err) => {
          this._isLoading.set(false)
          return throwError(() => err)
        }),
      )
      .subscribe()
  }

  // -------------------------------------------------------------------------
  // Private: agent event accumulation
  // -------------------------------------------------------------------------

  private _subscribeToAgentEvents(): void {
    this._agentEventSub?.unsubscribe()

    const conv = this._conversation!

    let idOfCurrentlyStreamingThinkingMessage: string | null = null
    let pendingThinkingContent = ''

    let idOfCurrentlyStreamingAssistantMessage: string | null = null
    let pendingContentText = ''

    // tool_result IDs saved this iteration, awaiting token_count from the next iteration_end
    let pendingToolResultIds: string[] = []

    let saveQueue: Promise<void> = Promise.resolve()
    const enqueue = (fn: () => Promise<unknown>) => {
      saveQueue = saveQueue.then(() => fn()).then(() => {}).catch((err) => console.error('Save error:', err))
    }

    const saveAssistant = (id: string, content: string, thinking: string) => {
      enqueue(() => firstValueFrom(this.api.post_message(conv.id, {
        id,
        role: 'assistant',
        content,
        thinking: thinking || undefined,
      })))
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

    const stopStreamingTheAssistantMessageSaveItAndClearIt = () => {
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
      saveAssistant(id, content, thinking)
    }

    this._agentEventSub = this.agentSvc.events$.subscribe((event) => {
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
      } else if (event.type === 'tool_confirm') {
        this._messages.update((msgs) => [
          ...msgs,
          {
            kind: 'tool_confirm',
            id: event.tool_id!,
            tool_id: event.tool_id!,
            tool_name: event.tool_name ?? '',
            args: event.arguments ?? {},
            preview: event.preview ?? '',
            confirmed: null,
          },
        ])
      } else if (event.type === 'tool_result') {
        markThinkingMessageAsDoneAndClearIt()
        const resultId = `result-${event.tool_id}`
        const resultContent = event.content ?? ''
        this._messages.update((msgs) => [
          ...msgs,
          {
            kind: 'tool_result',
            id: resultId,
            tool_name: event.tool_name ?? '',
            content: resultContent,
          },
        ])
        stopStreamingTheAssistantMessageSaveItAndClearIt()
        if (pendingThinkingContent) {
          const thinking = pendingThinkingContent
          pendingThinkingContent = ''
          saveAssistant(crypto.randomUUID(), '', thinking)
        }
        enqueue(() => firstValueFrom(this.api.post_message(conv.id, {
          id: resultId,
          role: 'tool',
          content: resultContent,
        })))
        pendingToolResultIds.push(resultId)
      } else if (event.type === 'iteration_end') {
        const tokens = event.prompt_tokens ?? 0
        this._promptTokens.set(tokens)
        markThinkingMessageAsDoneAndClearIt()
        stopStreamingTheAssistantMessageSaveItAndClearIt()
        for (const id of pendingToolResultIds) {
          const capturedId = id
          enqueue(() => firstValueFrom(this.api.patch_message_token_count(capturedId, tokens)).then(() => {
            this._messages.update((msgs) =>
              msgs.map((m) =>
                m.id === capturedId && m.kind === 'tool_result' ? { ...m, token_count: tokens } : m,
              ),
            )
          }))
        }
        pendingToolResultIds = []
      } else if (event.type === 'done' || event.type === 'error') {
        this._messages.update((msgs) => msgs.filter((m) => m.kind !== 'tool_confirm'))
        enqueue(() => this._computeTokenCountForLastMessage())
        this._agentEventSub?.unsubscribe()
        this._agentEventSub = null
      }
    })
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
          token_count: m.token_count,
          ...siblingMeta,
        })
      } else if (m.role === 'assistant') {
        if (m.content) {
          result.push({
            kind: 'assistant',
            id: m.id,
            content: m.content,
            thinking: m.thinking ?? undefined,
            token_count: m.token_count,
            ...siblingMeta,
          })
        } else if (m.thinking) {
          // Thinking-only message (no content) — show as a collapsed thinking block
          result.push({
            kind: 'thinking',
            id: m.id,
            content: m.thinking,
            done: true,
            ...siblingMeta,
          })
        }
      } else if (m.role === 'tool') {
        result.push({
          kind: 'tool_result',
          id: m.id,
          tool_name: '',
          content: m.content,
          token_count: m.token_count,
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

  private async _createConversation(userMsg: { id: string; content: string }): Promise<void> {
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
      }),
    )
  }

  private async _addUserMessageToDb(userMsg: { id: string; content: string }): Promise<void> {
    if (!this._conversation) {
      return
    }
    await firstValueFrom(
      this.api.post_message(this._conversation.id, {
        id: userMsg.id,
        role: 'user',
        content: userMsg.content,
      }),
    )
  }

  private async _computeTokenCountForLastMessage(): Promise<void> {
    const id = this._conversationId()
    if (!id) {
      return
    }
    const result = await firstValueFrom(this.api.compute_conversation_token_count(id))
    this._promptTokens.set(result.token_count)
    this._messages.update((msgs) => {
      const copy = [...msgs]
      const last = copy[copy.length - 1]
      if (!last) {
        return copy
      }
      if (last.kind === 'user') {
        copy[copy.length - 1] = { ...last, token_count: result.token_count }
      } else if (last.kind === 'assistant') {
        copy[copy.length - 1] = { ...last, token_count: result.token_count }
      } else if (last.kind === 'tool_result') {
        copy[copy.length - 1] = { ...last, token_count: result.token_count }
      }
      return copy
    })
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
}
