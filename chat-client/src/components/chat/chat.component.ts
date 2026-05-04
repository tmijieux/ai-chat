import { Component, inject, OnDestroy, OnInit } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { ChatService } from '../../services/chat.service'
import { map, Observable } from 'rxjs'
import { Conversation, ConversationHistory, Message } from '../../message-types'

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
  // Inject the service (which holds the state)
  readonly chatSvc = inject(ChatService)
  constructor() {}

  // Local state to hold the current history snapshot for UI rendering
  currentHistory: ConversationHistory = []

  // Observable references for the template
  history$: Observable<ConversationHistory> = this.chatSvc.history$
  isLoading$: Observable<boolean> = this.chatSvc.isLoading$
  contextTokens$: Observable<number> = this.chatSvc.contextTokens$
  currentConversationTitle$: Observable<string> = this.chatSvc.currentConversationTitle$

  // Local input state (standard TS property)
  currentInput: string = ''

  historySub = this.history$.subscribe((h) => {
    this.currentHistory = [...h]
  })
  conversations$ = this.chatSvc.conversations.obs$.pipe(
    map((conversations) => conversations.map((c) => ({ ...c, menuOpened: false }))),
  )

  ngOnInit(): void {
    // Initialize with a new chat title
    this.chatSvc.startNewChat('New Chat')
  }

  ngOnDestroy() {
    this.historySub.unsubscribe()
  }

  // --- Event Handlers ---

  /**
   * Handles starting a new conversation.
   * @param title The title derived from the user action.
   */
  handleMenuClick(action: string): void {
    if (action === 'new-chat') {
      // 1. Use the service method to reset state.
      this.chatSvc.startNewChat('New Chat')
      // 2. Manually update the component's local state cache (optional but helpful)
      this.currentHistory = []
    }
  }

  /**
   * Sends the message and handles the Observable stream response.
   * @param event The submit event object.
   */
  sendMessage(event: any): void {
    const input = this.currentInput
    if (!input.trim()) {
      return
    }
    this.currentInput = ''

    const userMessage: Message = { role: 'user', content: input }

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
