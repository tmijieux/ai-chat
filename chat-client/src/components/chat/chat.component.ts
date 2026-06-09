import { Component, computed, effect, ElementRef, HostListener, inject, OnDestroy, signal, untracked, ViewChild } from '@angular/core'
import { CommonModule } from '@angular/common'
import { ChatService } from '../../services/chat.service'
import {
  Conversation,
  ConversationSettings,
  DiffLine,
  DisplayMessageWithMeta,
  TokenMeta,
  ToolCallEntry,
} from '../../types/message-types'
import { ActivatedRoute, Router, RouterLink } from '@angular/router'
import { ConversationSettingsComponent } from '../conversation-settings/conversation-settings.component'
import { CollapsibleBubbleComponent } from '../collapsible-bubble/collapsible-bubble.component'
import { MarkdownComponent } from 'ngx-markdown'
import { ChatInputComponent } from '../chat-input/chat-input.component'
import { AppStatusService } from '../../services/app-status.service'
import { ToolResultComponent } from '../tool-result/tool-result.component'

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    ConversationSettingsComponent,
    CollapsibleBubbleComponent,
    MarkdownComponent,
    ChatInputComponent,
    ToolResultComponent,
  ],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
  host: { class: 'flex h-full' },
})
export class ChatComponent implements OnDestroy {
  private route = inject(ActivatedRoute)
  private router = inject(Router)
  readonly chatSvc = inject(ChatService)
  readonly appStatus = inject(AppStatusService)

  readonly drawerOpen = signal(false)
  readonly rejectingToolId = signal<string | null>(null)
  readonly rejectReason = signal('')

  // Edit state
  readonly editingMessageId = signal<string | null>(null)
  readonly editContent = signal('')

  // Action menu state
  readonly openMenuId = signal<string | null>(null)
  readonly openConvMenuId = signal<string | null>(null)

  // Auto-scroll
  @ViewChild('scrollContainer') private _scrollEl!: ElementRef<HTMLElement>
  readonly autoScrollEnabled = signal(true)

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

  // Enrich each message with token metadata and layout flags.
  readonly messagesWithMeta = computed<
    (DisplayMessageWithMeta & { turnStart: boolean; toolResultStart: boolean })[]
  >(() => {
    const msgs = this.chatSvc.messages()
    const promptTokens = this.activePrompt()?.token_count ?? 0
    const toolsTokens = this.activeToolsTokenCount() ?? 0
    let prevTokenCount: number | null = promptTokens + toolsTokens
    return msgs.map((msg, i) => {
      const turnStart = i > 0 && msg.kind === 'user'
      const toolResultStart = i > 0 && msg.kind === 'tool_result'
      if (msg.kind !== 'user' && msg.kind !== 'assistant' && msg.kind !== 'tool_result') {
        return { ...msg, turnStart, toolResultStart }
      }
      const tokenCount = msg.token_count ?? null
      if (tokenCount == null) {
        return { ...msg, turnStart, toolResultStart }
      }
      const tokenMeta: TokenMeta = {
        token_count: tokenCount,
        token_delta: prevTokenCount !== null ? tokenCount - prevTokenCount : null,
        token_pct: Math.round((tokenCount / this.CTX_LIMIT) * 100),
      }
      prevTokenCount = tokenCount
      return { ...msg, token_meta: tokenMeta, turnStart, toolResultStart }
    })
  })

  private _lcsEditFileDiff(oldStr: string, newStr: string): DiffLine[] {
    const oldLines = oldStr.split('\n')
    const newLines = newStr.split('\n')
    const m = oldLines.length
    const n = newLines.length
    const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0))
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        dp[i][j] = oldLines[i - 1] === newLines[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
    type Op = ['eq' | 'rm' | 'add', number, number]
    const ops: Op[] = []
    let i = m, j = n
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && oldLines[i - 1] === newLines[j - 1]) {
        ops.unshift(['eq', i - 1, j - 1]); i--; j--
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        ops.unshift(['add', -1, j - 1]); j--
      } else {
        ops.unshift(['rm', i - 1, -1]); i--
      }
    }
    const result: DiffLine[] = []
    let oldLine = 1
    for (const [kind, oldIdx, newIdx] of ops) {
      if (kind === 'eq') {
        oldLine++
      } else if (kind === 'rm') {
        result.push({ type: 'removed', line: oldLine++, text: oldLines[oldIdx] })
      } else {
        result.push({ type: 'added', line: null, text: newLines[newIdx] })
      }
    }
    return result
  }

  toolCallDiffLines(tc: ToolCallEntry): DiffLine[] | null {
    if (tc.name !== 'edit_file') {
      return null
    }
    const oldString = tc.args['old_string']
    const newString = tc.args['new_string']
    if (typeof oldString !== 'string' || typeof newString !== 'string') {
      return null
    }
    return this._lcsEditFileDiff(oldString, newString)
  }

  formatToolArgs(args: Record<string, unknown>): string {
    return Object.entries(args)
      .map(([key, value]) => {
        const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
        const truncated = text.length > 300 ? text.slice(0, 300) + '…' : text
        return `${key}: ${truncated}`
      })
      .join('\n\n')
  }

  toolCallNames(toolCalls: ToolCallEntry[] | null | undefined): string {
    if (toolCalls === null || toolCalls === undefined || toolCalls.length === 0) {
      return 'tool calls'
    }
    return toolCalls.map((tc) => tc.name).join(', ')
  }

  readonly conversations$ = this.chatSvc.conversations.obs$

  constructor() {
    effect(() => {
      this.messagesWithMeta()
      untracked(() => {
        if (this.autoScrollEnabled()) {
          queueMicrotask(() => {
            const el = this._scrollEl?.nativeElement
            if (el) {
              el.scrollTop = el.scrollHeight
            }
          })
        }
      })
    })

    // URL → Service: load conversation from URL on page load / back-forward navigation
    this.route.queryParamMap.subscribe((pm) => {
      const convId = pm.get('conversationId') ?? undefined
      if (convId !== undefined && convId !== this.chatSvc.currentConversationId()) {
        this.chatSvc.selectConversationById(convId)
      } else if (convId === undefined && this.chatSvc.currentConversationId() !== undefined) {
        this.chatSvc.startNewChat()
      }
    })

    // Service → URL: keep URL in sync when active conversation changes in-app
    effect(() => {
      const convId = this.chatSvc.currentConversationId()
      untracked(() => {
        const current = this.route.snapshot.queryParamMap.get('conversationId') ?? undefined
        if (convId === current) {
          return
        }
        this.router.navigate([], {
          relativeTo: this.route,
          queryParams: convId !== undefined ? { conversationId: convId } : {},
          replaceUrl: true,
        })
      })
    })
  }

  ngOnDestroy() {}

  // -------------------------------------------------------------------------
  // Scroll
  // -------------------------------------------------------------------------

  onScrollContainer(event: Event): void {
    const el = event.target as HTMLElement
    this.autoScrollEnabled.set(el.scrollHeight - el.scrollTop - el.clientHeight < 50)
  }

  scrollToBottom(): void {
    const el = this._scrollEl?.nativeElement
    if (el) {
      el.scrollTop = el.scrollHeight
    }
    this.autoScrollEnabled.set(true)
  }

  // -------------------------------------------------------------------------
  // User actions — all delegated to ChatService
  // -------------------------------------------------------------------------

  onSubmitted(data: { text: string; imageIds: string[] }): void {
    this.autoScrollEnabled.set(true)
    this.chatSvc.startAgentRun(data.text, data.imageIds)
  }

  startReject(toolId: string): void {
    this.rejectingToolId.set(toolId)
    this.rejectReason.set('')
  }

  onRejectKeydown(event: Event, toolId: string): void {
    const ke = event as KeyboardEvent
    if (!ke.shiftKey) {
      ke.preventDefault()
      this.sendRejection(toolId)
    }
  }

  sendRejection(toolId: string): void {
    const reason = this.rejectReason().trim() || undefined
    this.rejectingToolId.set(null)
    this.confirmTool(toolId, false, reason)
  }

  confirmTool(toolId: string, approved: boolean, reason?: string): void {
    this.chatSvc.confirmTool(toolId, approved, reason)
  }

  selectConversation(conv: Conversation | undefined): void {
    this.drawerOpen.set(false)
    this.autoScrollEnabled.set(true)
    this.chatSvc.selectConversation(conv)
  }

  startNewChat(): void {
    this.drawerOpen.set(false)
    this.autoScrollEnabled.set(true)
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

}
