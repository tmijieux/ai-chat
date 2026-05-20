import { Component, computed, inject, model, OnDestroy, signal } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { ChatService } from '../../services/chat.service'
import { map } from 'rxjs'
import {
  Conversation,
  ConversationSettings,
  DisplayMessageWithMeta,
  TokenMeta,
} from '../../types/message-types'
import { ActivatedRoute, RouterLink } from '@angular/router'
import { ConversationSettingsComponent } from '../conversation-settings/conversation-settings.component'

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, ConversationSettingsComponent],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
  host: { class: 'flex h-full' },
})
export class ChatComponent implements OnDestroy {
  private route = inject(ActivatedRoute)
  readonly chatSvc = inject(ChatService)

  readonly currentInput = model('')
  readonly drawerOpen = signal(false)

  readonly activePrompt = computed(() => {
    const settings = this.chatSvc.currentConversationSettings()
    if (settings?.active_prompt_id === null || settings?.active_prompt_id === undefined) {
      return null
    }
    return this.chatSvc.prompts().find((p) => p.id === settings.active_prompt_id) ?? null
  })

  // Enrich each message with token contribution metadata for the tooltip display.
  readonly messagesWithMeta = computed<DisplayMessageWithMeta[]>(() => {
    const msgs = this.chatSvc.messages()
    let prevTokenCount: number | null = null
    return msgs.map((msg) => {
      const tokenCount =
        msg.kind === 'user' || msg.kind === 'assistant' || msg.kind === 'tool_result'
          ? (msg.token_count ?? null)
          : null

      const tokenMeta: TokenMeta | undefined =
        tokenCount != null
          ? {
              token_count: tokenCount,
              token_contribution: prevTokenCount != null ? tokenCount - prevTokenCount : null,
              token_pct: Math.round((tokenCount / 16384) * 100),
            }
          : undefined

      if (tokenCount != null) prevTokenCount = tokenCount
      return { ...msg, token_meta: tokenMeta }
    })
  })

  readonly conversations$ = this.chatSvc.conversations.obs$.pipe(
    map((conversations) => conversations.map((c) => ({ ...c, menuOpened: false }))),
  )

  constructor() {
    this.route.queryParamMap.subscribe((pm) => {
      const convId = pm.get('conversationId')
      if (convId) {
        this.chatSvc.selectConversation({
          id: convId,
          title: '???',
          created_at: '',
          active_message_id: null,
          settings: null,
        })
      }
    })
  }

  ngOnDestroy() {}

  // -------------------------------------------------------------------------
  // User actions — all delegated to ChatService
  // -------------------------------------------------------------------------

  async sendMessage(event: Event | null): Promise<void> {
    if (event && (event as KeyboardEvent).shiftKey) return
    event?.preventDefault()
    const input = this.currentInput().trim()
    if (!input) return
    this.currentInput.set('')

    const settings = this.chatSvc.currentConversationSettings()
    if (settings?.agentic_mode) {
      this.chatSvc.startAgentRun(input)
    } else {
      this.chatSvc.sendMessage(input)
    }
  }

  confirmTool(toolId: string, approved: boolean): void {
    this.chatSvc.confirmTool(toolId, approved)
  }

  abortAgent(): void {
    this.chatSvc.abortAgent()
  }

  selectConversation(conv: Conversation | undefined): void {
    this.drawerOpen.set(false)
    this.chatSvc.selectConversation(conv)
  }

  startNewChat(): void {
    this.drawerOpen.set(false)
    this.chatSvc.startNewChat()
  }

  deleteConversation(conv: Conversation): void {
    this.chatSvc.deleteConversation(conv)
  }

  onSettingsChanged(settings: ConversationSettings): void {
    this.chatSvc.updateConversationSettings(settings)
  }
}
