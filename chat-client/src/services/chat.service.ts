import { Injectable, inject, signal } from '@angular/core'
import { Observable, BehaviorSubject, firstValueFrom } from 'rxjs'
import { catchError, tap, shareReplay, switchMap } from 'rxjs/operators'
import { HttpClient, HttpEvent } from '@angular/common/http'
import { throwError } from 'rxjs'
import { ApiResponse, Conversation, ConversationHistory, Message } from '../types/message-types'

class MyShare<T> {
  private _obs$: Observable<T>
  constructor(observableFactory: () => Observable<T>) {
    this._obs$ = this._refreshSubject.pipe(
      switchMap(() => observableFactory()),
      shareReplay(1),
    )
  }
  private _refreshSubject = new BehaviorSubject<void>(undefined)
  public get obs$(): Observable<T> {
    return this._obs$
  }
  public refresh() {
    this._refreshSubject.next()
  }
}

function myShared<T>(observableFactory: () => Observable<T>) {
  return new MyShare(observableFactory)
}

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private http = inject(HttpClient)

  // --- State Management using BehaviorSubject ---
  private historySubject = new BehaviorSubject<ConversationHistory>([])
  public history$: Observable<ConversationHistory> = this.historySubject.asObservable()

  // Loading state
  private isLoadingSubject = new BehaviorSubject<boolean>(false)
  public isLoading$: Observable<boolean> = this.isLoadingSubject.asObservable()

  // Token count
  private contextTokensSubject = new BehaviorSubject<number>(0)
  public contextTokens$: Observable<number> = this.contextTokensSubject.asObservable()

  conversations = myShared(() => {
    return this.http.get<Conversation[]>('http://localhost:8000/api/conversations')
  })
  // ------------------

  constructor() {}

  /**
   * Sets a new conversation state and resets the entire chat history and tokens.
   * @param title The title for the new conversation.
   */
  public startNewChat(): void {
    this.isLoadingSubject.next(false)
    this.contextTokensSubject.next(0)
    this.historySubject.next([])
    this.conversation = undefined
    this._conversationId.set(undefined)
    this._conversationSettings.set(null)
  }

  public async createConversation(message: Message) {
    const conversation = await firstValueFrom(
      this.http.post<Conversation>('http://localhost:8000/api/conversations', {
        title: message.content.substring(0, 20),
      }),
    )
    this.conversation = conversation
    this._conversationId.set(conversation.id)
    this.conversations.refresh()
    const pendingSettings = this._conversationSettings()
    if (pendingSettings) {
      await firstValueFrom(
        this.http.put(`http://localhost:8000/api/conversations/${conversation.id}/settings`, pendingSettings)
      )
    }
    await this.addMessageToConversation(conversation, message)
  }

  async addMessageToConversation(conversation: Conversation, message: Message): Promise<string> {
    const res = await firstValueFrom(
      this.http.post<{ id: string; parent_id: string | null }>(
        'http://localhost:8000/api/messages',
        message,
        { params: { conversationId: conversation.id } },
      ),
    )
    return res.id
  }

  private conversation: Conversation | undefined = undefined

  private _conversationId = signal<string | undefined>(undefined)
  public currentConversationId = this._conversationId.asReadonly()

  private _conversationSettings = signal<{ agentic_mode: boolean } | null>(null)
  public currentConversationSettings = this._conversationSettings.asReadonly()

  public selectConversation(conversation: Conversation | undefined) {
    if (conversation === undefined) {
      this.historySubject.next([])
      this.conversation = undefined
      this._conversationId.set(undefined)
      this._conversationSettings.set(null)
    } else {
      this.http
        .get<Message[]>(`http://localhost:8000/api/conversations/${conversation.id}/messages`)
        .subscribe((messages) => {
          this.historySubject.next(messages)
          this.conversation = conversation
          this._conversationId.set(conversation.id)
          const settings = conversation.settings ? JSON.parse(conversation.settings) : null
          this._conversationSettings.set(settings)
          const lastWithCount = [...messages].reverse().find(m => m.token_count != null)
          this.contextTokensSubject.next(lastWithCount?.token_count ?? 0)
        })
    }
  }

  /**
   * Sends the chat history and returns an Observable that streams the response.
   */
  public sendMessage(history: ConversationHistory): Observable<HttpEvent<string>> {
    // 1. Set loading state
    this.isLoadingSubject.next(true)

    // Build messages array for backend
    const messagesArray: { role: string; content: string }[] = history.map((m) => ({
      role: m.role,
      content: m.content,
      thinking: m.thinking,
    }))
    const lastMessage = history[history.length - 1]
    if (history.length === 1) {
      this.createConversation(lastMessage)
    } else if (this.conversation) {
      this.addMessageToConversation(this.conversation, lastMessage)
    }

    const svcHistory = this.historySubject.getValue()
    svcHistory.push(history[history.length - 1])
    svcHistory.push({
      id: 'next-loading',
      role: 'assistant',
      content: '',
      thinking: '',
      loading: true,
    })
    this.historySubject.next(svcHistory)

    // Call backend API and stream response
    return this.http
      .post(
        'http://localhost:8000/api/chat',
        { messages: messagesArray },
        {
          observe: 'events',
          responseType: 'text',
          reportProgress: true,
        },
      )
      .pipe(
        tap((response) => {
          if (response.type === 3) {
            const lines = (response.partialText ?? '').split('\n')
            const objs: ApiResponse[] = lines
              .filter((t) => t.endsWith('}'))
              .map((t) => JSON.parse(t))
            const thinking = objs.map((o) => o.message.thinking ?? '').join('')
            const content = objs.map((o) => o.message.content).join('')
            const back = svcHistory[svcHistory.length - 1]
            back.content = content
            back.thinking = thinking
            // console.log('thinking=', thinking)
            // console.log('content=', content)
            if (back.thinking && !back.content) {
              console.log('started thinking')
              back.loading = false
              back.thinking_visible = true
            } else if (back.content) {
              console.log('finished thinking')
              back.loading = false
              back.thinking_visible = false
            }
            this.historySubject.next([...svcHistory])

            const lastResponse = objs[objs.length - 1]
            if (lastResponse.done) {
              this.isLoadingSubject.next(false)
              if (this.conversation) {
                const conv = this.conversation
                ;(async () => {
                  await this.addMessageToConversation(conv, back)
                  await this.countTokensForCurrentConversation()
                })()
              }
            }
          }
        }),
        catchError((error) => {
          this.isLoadingSubject.next(false)
          console.error('API Call Failed:', error)
          return throwError(() => error)
        }),
      )
  }

  public async countTokensForCurrentConversation(): Promise<void> {
    const id = this._conversationId()
    if (!id) return
    const result = await firstValueFrom(
      this.http.post<{ token_count: number; message_id: string }>(
        `http://localhost:8000/api/conversations/${id}/count-tokens`,
        {}
      )
    )
    this.contextTokensSubject.next(result.token_count)
    const history = this.historySubject.getValue()
    if (history.length > 0) {
      history[history.length - 1].token_count = result.token_count
      this.historySubject.next([...history])
    }
  }

  public async addMessageToCurrentConversation(role: 'assistant' | 'tool', content: string, thinking?: string, tokenCount?: number): Promise<void> {
    if (!this.conversation) return
    const msg: Message = { id: crypto.randomUUID(), role, content, thinking, token_count: tokenCount }
    await this.addMessageToConversation(this.conversation, msg)
  }

  public async prepareAgentConversation(userMessage: string): Promise<string | undefined> {
    this.contextTokensSubject.next(0)
    const msg: Message = { id: crypto.randomUUID(), role: 'user', content: userMessage }
    if (!this.conversation) {
      await this.createConversation(msg)
    } else {
      await this.addMessageToConversation(this.conversation, msg)
    }
    return this._conversationId()
  }

  async setAgenticMode(enabled: boolean): Promise<void> {
    const current = this._conversationSettings() ?? { agentic_mode: false, active_prompt_ids: [], active_tool_names: [], tools_enabled: true }
    const updated = { ...current, agentic_mode: enabled }
    this._conversationSettings.set(updated)
    const id = this._conversationId()
    if (!id) return
    await firstValueFrom(
      this.http.put(`http://localhost:8000/api/conversations/${id}/settings`, updated)
    )
  }

  async deleteConversation(conv: Conversation) {
    await firstValueFrom(this.http.delete(`http://localhost:8000/api/conversations/${conv.id}`))
    this.conversations.refresh()
    if (conv.id === this.conversation?.id) {
      console.log('was selected conversation !!!!!')
      this.selectConversation(undefined)
    } else {
      console.log('was NOT selected conversation ?!?!?!')
    }
  }
}
