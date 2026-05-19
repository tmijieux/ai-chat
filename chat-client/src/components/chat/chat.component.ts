import { Component, inject, model, OnDestroy, OnInit } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { ChatService } from '../../services/chat.service'
import { map, Observable } from 'rxjs'
import { Conversation, ConversationHistory, Message } from '../../types/message-types'
import { ActivatedRoute, Router } from '@angular/router'

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
  host: {
    class: 'flex h-full',
  },
})
export class ChatComponent implements OnInit, OnDestroy {
  private route = inject(ActivatedRoute)
  private router = inject(Router)
  readonly chatSvc = inject(ChatService)

  // Local state to hold the current history snapshot for UI rendering
  currentHistory: Message[] = []

  // Observable references for the template
  history$: Observable<Message[]> = this.chatSvc.history$
  isLoading$: Observable<boolean> = this.chatSvc.isLoading$
  contextTokens$: Observable<number> = this.chatSvc.contextTokens$

  // Local input state (standard TS property)
  currentInput = model('')

  private historySub = this.history$.subscribe((h) => {
    this.currentHistory = [...h]
  })
  conversations$ = this.chatSvc.conversations.obs$.pipe(
    map((conversations) => conversations.map((c) => ({ ...c, menuOpened: false }))),
  )

  constructor() {
    this.route.queryParamMap.subscribe((pm) => {
      console.log('pm=', pm)
      const convId = pm.get('conversationId')
      if (convId) {
        this.chatSvc.selectConversation({ id: convId, title: '???' })
      }
    })
  }

  ngOnInit(): void {}

  ngOnDestroy() {
    this.historySub.unsubscribe()
  }

  /**
   * Sends the message and handles the Observable stream response.
   * @param event The submit event object.
   */
  sendMessage(event: any): void {
    const input = this.currentInput()
    if (!input.trim()) {
      return
    }
    this.currentInput.set('')

    const userMessage: Message = { id: 'next-loading', role: 'user', content: input }

    // 1. Immediately update the component's local history cache (for the user message)
    this.currentHistory.push(userMessage)

    // 2. Get the history that the backend needs (the current state array)
    const historyToSend: ConversationHistory = [...this.currentHistory]

    // 3. Call the service and subscribe to the stream Observable
    this.chatSvc.sendMessage(historyToSend).subscribe()
  }

  openConversationMenu() {}

  deleteConversation(conv: Conversation) {
    this.chatSvc.deleteConversation(conv)
  }
}
