import { Component, computed, HostListener, inject, model, OnDestroy, signal } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { ChatService } from '../../services/chat.service'
import {
  Conversation,
  ConversationSettings,
  DisplayMessageWithMeta,
  TokenMeta,
} from '../../types/message-types'
import { ActivatedRoute, RouterLink } from '@angular/router'
import { ConversationSettingsComponent } from '../conversation-settings/conversation-settings.component'
import { CollapsibleBubbleComponent } from '../collapsible-bubble/collapsible-bubble.component'
import { MarkdownComponent } from 'ngx-markdown'

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    RouterLink,
    ConversationSettingsComponent,
    CollapsibleBubbleComponent,
    MarkdownComponent,
  ],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
  host: { class: 'flex h-full' },
})
export class ChatComponent implements OnDestroy {
  private route = inject(ActivatedRoute)
  readonly chatSvc = inject(ChatService)

  readonly currentInput = model('')
  readonly drawerOpen = signal(false)
  readonly rejectingToolId = signal<string | null>(null)
  readonly rejectReason = signal('')

  // Edit state
  readonly editingMessageId = signal<string | null>(null)
  readonly editContent = signal('')

  // Action menu state
  readonly openMenuId = signal<string | null>(null)
  readonly openConvMenuId = signal<string | null>(null)

  // Voice dictation
  readonly isRecording = signal(false)
  readonly isTranscribing = signal(false)
  private _mediaRecorder: MediaRecorder | null = null
  private _audioChunks: Blob[] = []

  // Raw markdown toggle
  private rawModeIds = signal(new Set<string>())
  readonly CTX_LIMIT = 2 ** 15

  isRaw(msgId: string): boolean {
    return this.rawModeIds().has(msgId)
  }

  toggleRaw(msgId: string): void {
    const s = new Set(this.rawModeIds())
    s.has(msgId) ? s.delete(msgId) : s.add(msgId)
    this.rawModeIds.set(s)
  }

  @HostListener('document:click')
  closeMenu(): void {
    this.openMenuId.set(null)
    this.openConvMenuId.set(null)
  }

  toggleConvMenu(convId: string, event: Event): void {
    event.stopPropagation()
    this.openConvMenuId.set(this.openConvMenuId() === convId ? null : convId)
  }

  readonly activePrompt = computed(() => {
    const settings = this.chatSvc.currentConversationSettings()
    if (settings.active_prompt_id === null || settings.active_prompt_id === undefined) {
      return null
    }
    return this.chatSvc.prompts().find((p) => p.id === settings.active_prompt_id) ?? null
  })

  readonly activeToolsTokenCount = computed(() => {
    const activeNames = new Set(this.chatSvc.currentConversationSettings().active_tool_names)
    if (activeNames.size === 0) return null
    const enabledTools = this.chatSvc.allTools().filter((t) => activeNames.has(t.name))
    const N = enabledTools.length
    const toolTokens = enabledTools.reduce((sum, t) => sum + t.token_count, 0)
    const stackingOverhead = this.chatSvc.stackingOverheadPerAdditionalTool() * (N - 1)
    return this.chatSvc.toolFrameworkOverhead() + toolTokens + stackingOverhead
  })

  // Enrich each message with token metadata for the tooltip display.
  readonly messagesWithMeta = computed<DisplayMessageWithMeta[]>(() => {
    const msgs = this.chatSvc.messages()
    const promptTokens = this.activePrompt()?.token_count ?? 0
    const toolsTokens = this.activeToolsTokenCount() ?? 0
    let prevTokenCount: number | null = promptTokens + toolsTokens
    return msgs.map((msg) => {
      if (msg.kind !== 'user' && msg.kind !== 'assistant' && msg.kind !== 'tool_result') {
        return { ...msg }
      }
      const tokenCount = msg.token_count ?? null
      if (tokenCount == null) {
        return { ...msg }
      }
      const tokenMeta: TokenMeta = {
        token_count: tokenCount,
        token_delta: prevTokenCount !== null ? tokenCount - prevTokenCount : null,
        token_pct: Math.round((tokenCount / this.CTX_LIMIT) * 100),
      }
      prevTokenCount = tokenCount
      return { ...msg, token_meta: tokenMeta }
    })
  })

  readonly conversations$ = this.chatSvc.conversations.obs$

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
    this.chatSvc.startAgentRun(input)
  }

  startReject(toolId: string): void {
    this.rejectingToolId.set(toolId)
    this.rejectReason.set('')
  }

  sendRejection(toolId: string): void {
    const reason = this.rejectReason().trim() || undefined
    this.rejectingToolId.set(null)
    this.confirmTool(toolId, false, reason)
  }

  confirmTool(toolId: string, approved: boolean, reason?: string): void {
    this.chatSvc.confirmTool(toolId, approved, reason)
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
    this.chatSvc.updateConversationSettings(settings).subscribe()
  }

  // -------------------------------------------------------------------------
  // Edit
  // -------------------------------------------------------------------------

  startEdit(msgId: string, content: string): void {
    this.editingMessageId.set(msgId)
    this.editContent.set(content)
  }

  cancelEdit(): void {
    this.editingMessageId.set(null)
    this.editContent.set('')
  }

  async submitEdit(msgId: string): Promise<void> {
    const content = this.editContent().trim()
    if (!content) return
    this.cancelEdit()
    await this.chatSvc.editUserMessage(msgId, content)
  }

  onEditKeydown(event: KeyboardEvent, msgId: string): void {
    if (event.key === 'Escape') {
      this.cancelEdit()
      return
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      this.submitEdit(msgId)
    }
  }

  // -------------------------------------------------------------------------
  // Action menu
  // -------------------------------------------------------------------------

  toggleMenu(msgId: string, event: Event): void {
    event.stopPropagation()
    this.openMenuId.set(this.openMenuId() === msgId ? null : msgId)
  }

  async onDeleteMessage(msgId: string, subtree: boolean): Promise<void> {
    this.openMenuId.set(null)
    await this.chatSvc.deleteMessage(msgId, subtree)
  }

  // -------------------------------------------------------------------------
  // Sibling navigation
  // -------------------------------------------------------------------------

  async onNavigateSibling(siblingId: string): Promise<void> {
    await this.chatSvc.navigateSibling(siblingId)
  }

  // -------------------------------------------------------------------------
  // Voice dictation
  // -------------------------------------------------------------------------

  async toggleMic(): Promise<void> {
    if (this.isRecording()) {
      this._mediaRecorder?.stop()
    } else {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      this._mediaRecorder = new MediaRecorder(stream)
      this._audioChunks = []
      this._mediaRecorder.ondataavailable = (e) => {
        this._audioChunks.push(e.data)
      }
      this._mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(this._audioChunks, { type: 'audio/webm' })
        this.isRecording.set(false)
        this.isTranscribing.set(true)
        try {
          const text = await this.chatSvc.transcribe(blob)
          this.currentInput.set((this.currentInput() + ' ' + text).trim())
        } finally {
          this.isTranscribing.set(false)
        }
      }
      this._mediaRecorder.start()
      this.isRecording.set(true)
    }
  }
}
