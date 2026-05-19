import { Injectable, inject } from '@angular/core'
import { Observable, BehaviorSubject, defer, firstValueFrom } from 'rxjs'
import {
  catchError,
  tap,
  map,
  last,
  shareReplay,
  switchMap,
  distinctUntilChanged,
} from 'rxjs/operators'
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
  public startNewChat(title: string): void {
    // State mutation sequence:
    this.isLoadingSubject.next(false)
    this.contextTokensSubject.next(0)
    this.historySubject.next([])
  }

  public async createConversation(message: Message) {
    const conversation = await firstValueFrom(
      this.http.post<Conversation>('http://localhost:8000/api/conversations', {
        title: message.content.substring(0, 20),
      }),
    )
    this.conversation = conversation
    this.conversations.refresh()
    await this.addMessageToConversation(conversation, message)
  }

  async addMessageToConversation(conversation: Conversation, message: Message) {
    await firstValueFrom(
      this.http.post<{ id: string }>('http://localhost:8000/api/messages', message, {
        params: { conversationId: conversation.id },
      }),
    )
  }

  private conversation: Conversation | undefined = undefined
  public selectConversation(conversation: Conversation | undefined) {
    if (conversation === undefined) {
      this.historySubject.next([])
      this.conversation = undefined
    } else {
      this.http
        .get<Message[]>(`http://localhost:8000/api/conversations/${conversation.id}/messages`)
        .subscribe((messages) => {
          this.historySubject.next(messages)
          this.conversation = conversation
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
              // Update token count from response
              this.isLoadingSubject.next(false)
              // Update final token count
              this.contextTokensSubject.next(lastResponse.prompt_eval_count)
              if (this.conversation) {
                this.addMessageToConversation(this.conversation, back)
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
