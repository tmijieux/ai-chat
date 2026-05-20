import { Component, computed, inject, model, OnDestroy } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { ChatService } from '../../services/chat.service'
import { AgentService } from '../../services/agent.service'
import { toSignal } from '@angular/core/rxjs-interop'
import { map, Observable, Subscription } from 'rxjs'
import { Conversation, ConversationHistory, Message } from '../../types/message-types'
import { ActivatedRoute } from '@angular/router'

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
export class ChatComponent implements OnDestroy {
  private route = inject(ActivatedRoute)
  readonly chatSvc = inject(ChatService)
  readonly agentSvc = inject(AgentService)

  currentHistory: Message[] = []

  history$: Observable<Message[]> = this.chatSvc.history$
  isLoading$: Observable<boolean> = this.chatSvc.isLoading$

  private readonly contextTokensSig = toSignal(this.chatSvc.contextTokens$, { initialValue: 0 })
  displayTokens = computed(() => this.contextTokensSig() || this.agentSvc.promptTokens())

  private readonly historySig = toSignal(this.chatSvc.history$, { initialValue: [] as Message[] })
  historyWithMeta = computed(() => {
    const msgs = this.historySig()
    return msgs.map((msg, i) => {
      let prevTokenCount: number | null = null
      for (let j = i - 1; j >= 0; j--) {
        if (msgs[j].token_count != null) { prevTokenCount = msgs[j].token_count as number; break }
      }
      return {
        ...msg,
        tokenContribution: msg.token_count != null && prevTokenCount != null
          ? msg.token_count - prevTokenCount
          : null,
        tokenPct: msg.token_count != null ? Math.round(msg.token_count / 16384 * 100) : null,
      }
    })
  })

  currentInput = model('')

  // Agent-mode signals
  agentRunning = this.agentSvc.running
  agentMessages = this.agentSvc.messages
  agentPhase = this.agentSvc.phase

  private historySub = this.history$.subscribe((h) => {
    this.currentHistory = [...h]
  })

  private agentDoneSub: Subscription = this.agentSvc.done$.subscribe(async () => {
    await this.saveAgentMessages()
    await this.chatSvc.countTokensForCurrentConversation()
  })

  conversations$ = this.chatSvc.conversations.obs$.pipe(
    map((conversations) => conversations.map((c) => ({ ...c, menuOpened: false }))),
  )

  constructor() {
    this.route.queryParamMap.subscribe((pm) => {
      const convId = pm.get('conversationId')
      if (convId) {
        this.chatSvc.selectConversation({ id: convId, title: '???', created_at: '', active_message_id: null, settings: null })
      }
    })
  }

  ngOnDestroy() {
    this.historySub.unsubscribe()
    this.agentDoneSub.unsubscribe()
  }

  private async saveAgentMessages(): Promise<void> {
    const allMsgs = this.agentSvc.messages()

    // Collect prompt_tokens from each iteration_end in order
    const iterEnds = allMsgs
      .filter(m => m.ui_type === 'iteration_end')
      .map(m => m.prompt_tokens ?? 0)

    // Track which iteration (0-based) each message belongs to
    let iterIndex = 0
    const msgIter: number[] = allMsgs.map(m => {
      if (m.ui_type === 'iteration_end') { iterIndex++; return -1 }
      return iterIndex
    })

    let pendingThinking: string | undefined
    for (let i = 0; i < allMsgs.length; i++) {
      const msg = allMsgs[i]
      if (msg.ui_type === 'user' || msg.ui_type === 'iteration_end') continue

      if (msg.ui_type === 'thinking') {
        pendingThinking = msg.content
      } else if (msg.ui_type === 'content') {
        await this.chatSvc.addMessageToCurrentConversation('assistant', msg.content, pendingThinking)
        pendingThinking = undefined
      } else if (msg.ui_type === 'tool_result') {
        if (pendingThinking !== undefined) {
          await this.chatSvc.addMessageToCurrentConversation('assistant', '', pendingThinking)
          pendingThinking = undefined
        }
        // Token count = prompt of the next iteration (which consumed these tool results as input)
        const k = msgIter[i]
        const tokenCount = k + 1 < iterEnds.length ? iterEnds[k + 1] : undefined
        await this.chatSvc.addMessageToCurrentConversation('tool', msg.content, undefined, tokenCount)
      }
    }
  }

  async sendMessage(event: Event | null): Promise<void> {
    // Allow Shift+Enter to insert newline; plain Enter sends
    if (event && (event as KeyboardEvent).shiftKey) return
    event?.preventDefault()

    const input = this.currentInput()
    if (!input.trim()) return
    this.currentInput.set('')

    // Check if the current conversation has agentic mode enabled
    const settings = this.chatSvc.currentConversationSettings()
    if (settings?.agentic_mode) {
      const convId = await this.chatSvc.prepareAgentConversation(input)
      this.agentSvc.startWithUserMessage(input, convId)
    } else {
      const userMessage: Message = { id: 'next-loading', role: 'user', content: input }
      this.currentHistory.push(userMessage)
      const historyToSend: ConversationHistory = [...this.currentHistory]
      this.chatSvc.sendMessage(historyToSend).subscribe()
    }
  }

  confirmTool(toolId: string, approved: boolean): void {
    this.agentSvc.confirm(toolId, approved)
  }

  abortAgent(): void {
    this.agentSvc.abort()
  }

  openConversationMenu() {}

  deleteConversation(conv: Conversation) {
    this.chatSvc.deleteConversation(conv)
  }
}
