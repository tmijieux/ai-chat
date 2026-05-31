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
  lastWorkingDirectory = new RefreshableQuery(() =>
    this.api.get_app_setting('last_working_directory'),
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
    this.api.get_system_prompts().subscribe((p) => {
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
    this.api.get_agent_tools().subscribe((response: AgentToolsResponse) => {
      const names = response.tools.map((t) => t.name)
      this._allToolNames.set(names)
      this._allTools.set(response.tools)
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
      const settings: ConversationSettings = conversation.settings
        ? JSON.parse(conversation.settings)
        : this.DEFAULT_SETTINGS
      this._conversationSettings.set(settings)
      if (settings.working_directory != null) {
        this._lastWorkingDirectory.set(settings.working_directory)
        this.api.put_app_setting('last_working_directory', settings.working_directory).subscribe()
      }
      const lastWithTokens = [...dbMessages].reverse().find((m) => m.token_count != null)
      this._promptTokens.set(lastWithTokens?.token_count ?? 0)
    })
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
    this.agentSvc.start(newContent, convId, newMsgId)
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

  startAgentRun(input: string, imageIds: string[] = []): void {
    const userMsg: DisplayMessage = { kind: 'user', id: crypto.randomUUID(), content: input }
    this._messages.update((msgs) => [...msgs, userMsg])
    this._promptTokens.set(0)

    const persist = !this._conversation
      ? this._createConversation(userMsg, imageIds)
      : this._addUserMessageToDb(userMsg, imageIds)

    persist.then(() => {
      const convId = this._conversationId()
      this._subscribeToAgentEvents(userMsg.id)
      this.agentSvc.start(input, convId, userMsg.id)
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

  // -------------------------------------------------------------------------
  // Settings / conversation management
  // -------------------------------------------------------------------------

  updateConversationSettings(settings: ConversationSettings): Observable<unknown> {
    this._conversationSettings.set(settings)
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
    let pendingThinkingContent = ''

    let idOfCurrentlyStreamingAssistantMessage: string | null = null
    let pendingContentText = ''

    // The prompt_token count reported by Ollama in iteration_end is the number of tokens in the
    // messages sent to that iteration — which includes the tool results from the previous iteration.
    // We therefore patch tool result IDs from the previous iteration when the next iteration_end arrives.
    let toolResultIdsFromCurrentIteration: string[] = []
    let toolResultIdsFromPreviousIteration: string[] = [userMessageId]

    let saveQueue: Promise<void> = Promise.resolve()
    const enqueue = (fn: () => Promise<unknown>) => {
      saveQueue = saveQueue
        .then(() => fn())
        .then(() => {})
        .catch((err) => console.error('Save error:', err))
    }

    const saveAssistant = (id: string, content: string, thinking: string) => {
      enqueue(() =>
        firstValueFrom(
          this.api.post_message(conv.id, {
            id,
            role: 'assistant',
            content,
            thinking: thinking || undefined,
          }),
        ),
      )
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
            diff_lines: event.diff_lines,
            confirmed: null,
          },
        ])
      } else if (event.type === 'tool_result') {
        markThinkingMessageAsDoneAndClearIt()
        const resultId = `result-${event.tool_id}`
        const resultContent = event.content ?? ''
        const logMessage = event.log_message ?? null
        this._messages.update((msgs) => [
          ...msgs,
          {
            kind: 'tool_result',
            id: resultId,
            tool_name: event.tool_name ?? '',
            log_message: logMessage,
            content: resultContent,
          },
        ])
        stopStreamingTheAssistantMessageSaveItAndClearIt()
        if (pendingThinkingContent) {
          const thinking = pendingThinkingContent
          pendingThinkingContent = ''
          saveAssistant(crypto.randomUUID(), '', thinking)
        }
        enqueue(() =>
          firstValueFrom(
            this.api.post_message(conv.id, {
              id: resultId,
              role: 'tool',
              content: resultContent,
              log_message: logMessage,
            }),
          ),
        )
        toolResultIdsFromCurrentIteration.push(resultId)
      } else if (event.type === 'iteration_end') {
        const tokens = event.prompt_tokens ?? 0
        this._promptTokens.set(tokens)
        markThinkingMessageAsDoneAndClearIt()
        stopStreamingTheAssistantMessageSaveItAndClearIt()
        for (const id of toolResultIdsFromPreviousIteration) {
          const capturedId = id
          const currentMsgs = this._messages()
          const idx = currentMsgs.findIndex((m) => m.id === capturedId)
          const prevCount = this._findCumulativeTokenCountOfClosestPrecedingMessageThatHasOne(
            currentMsgs,
            idx,
          )
          const delta = prevCount !== null ? tokens - prevCount : null
          enqueue(() =>
            firstValueFrom(this.api.patch_message_token_count(capturedId, tokens, delta)).then(
              () => {
                this._messages.update((msgs) =>
                  msgs.map((m) =>
                    m.id === capturedId ? { ...m, token_count: tokens, token_delta: delta } : m,
                  ),
                )
              },
            ),
          )
        }
        toolResultIdsFromPreviousIteration = toolResultIdsFromCurrentIteration
        toolResultIdsFromCurrentIteration = []
      } else if (event.type === 'done' || event.type === 'error') {
        this._messages.update((msgs) => msgs.filter((m) => m.kind !== 'tool_confirm'))
        if (event.type === 'error') {
          const errorText = event.message ?? 'An error occurred'
          // Reload first, then append error so the reload doesn't wipe the error bubble.
          enqueue(async () => {
            await this._reloadFromDb()
            this._messages.update((msgs) => [
              ...msgs,
              { kind: 'error', id: crypto.randomUUID(), message: errorText },
            ])
          })
        } else {
          enqueue(() => this._computeTokenCountForLastMessage())
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
    await firstValueFrom(this.api.compress_conversation(id))
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

  private async _computeTokenCountForLastMessage(): Promise<void> {
    const id = this._conversationId()
    if (!id) {
      return
    }
    const msgs = this._messages()
    const lastIdx = msgs.length - 1
    const last = msgs[lastIdx]
    if (!last) {
      return
    }
    const prevCount = this._findCumulativeTokenCountOfClosestPrecedingMessageThatHasOne(
      msgs,
      lastIdx,
    )
    const result = await firstValueFrom(this.api.compute_conversation_token_count(id))
    this._promptTokens.set(result.token_count)
    const delta = prevCount !== null ? result.token_count - prevCount : null
    await firstValueFrom(this.api.patch_message_token_count(last.id, result.token_count, delta))
    this._messages.update((msgs) => {
      const copy = [...msgs]
      const last = copy[copy.length - 1]
      if (!last) {
        return copy
      }
      if (last.kind === 'user' || last.kind === 'assistant' || last.kind === 'tool_result') {
        copy[copy.length - 1] = { ...last, token_count: result.token_count, token_delta: delta }
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

  async transcribe(blob: Blob, language: string | null = 'fr'): Promise<string> {
    const res = await firstValueFrom(this.api.post_transcribe(blob, language))
    return res.text
  }
}
