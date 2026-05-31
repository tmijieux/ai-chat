import { Component, computed, effect, ElementRef, HostListener, inject, model, OnDestroy, signal, untracked, ViewChild } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { firstValueFrom } from 'rxjs'
import { ChatService } from '../../services/chat.service'
import {
  Conversation,
  ConversationSettings,
  DisplayMessageWithMeta,
  PendingImage,
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

  // Image attachments
  readonly pendingImages = signal<PendingImage[]>([])
  readonly isUploading = computed(() => this.pendingImages().some((p) => p.uploading))

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
  // Tool result parsing helpers
  // -------------------------------------------------------------------------

  parseGrepResult(content: string): {
    files: { name: string; lines: { no: number; text: string; match: boolean }[] }[]
    pattern: string
    total: number
    truncated: boolean
  } | null {
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'grep_files' || !Array.isArray(r.matches)) return null
      const map = new Map<string, { no: number; text: string; match: boolean }[]>()
      for (const m of r.matches) {
        if (!map.has(m.file)) map.set(m.file, [])
        map.get(m.file)!.push({ no: m.line, text: m.content, match: !!m.match })
      }
      return {
        files: [...map.entries()].map(([name, lines]) => ({ name, lines })),
        pattern: r.pattern ?? '',
        total: r.total ?? 0,
        truncated: !!r.truncated,
      }
    } catch {
      return null
    }
  }

  formatSimpleToolResult(content: string): { icon: string; text: string } | null {
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'write_file' && r.tool !== 'edit_file') return null
      const icon = r.status === 'success' ? '✓' : '✗'
      const msg = r.message ? ` — ${r.message}` : ''
      return { icon, text: `${r.tool}: ${r.path ?? ''}${msg}` }
    } catch {
      return null
    }
  }

  formatShellResult(content: string): { icon: string; output: string } | null {
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'run_shell') return null
      const ok = r.status === 'success'
      const icon = ok ? '✓ exit 0' : '✗ exit 1'
      const output = ok ? (r.output ?? '') : (r.error?.message ?? '')
      return { icon, output }
    } catch {
      return null
    }
  }

  grepHeaderSuffix(content: string, compressed: string | null | undefined): string {
    if (compressed) return ''
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'grep_files' || r.total == null) return ''
      const n: number = r.total
      return ` → ${n} ${n === 1 ? 'match' : 'matches'}`
    } catch {
      return ''
    }
  }

  // -------------------------------------------------------------------------
  // User actions — all delegated to ChatService
  // -------------------------------------------------------------------------

  async sendMessage(event: Event | null): Promise<void> {
    if (event && (event as KeyboardEvent).shiftKey) return
    event?.preventDefault()
    const input = this.currentInput().trim()
    if (!input && this.pendingImages().length === 0) return
    const imageIds = this.pendingImages().filter((p) => !p.uploading && p.id).map((p) => p.id!)
    this.currentInput.set('')
    this.pendingImages.set([])
    this.autoScrollEnabled.set(true)
    this.chatSvc.startAgentRun(input, imageIds)
  }

  attachImages(files: FileList | File[]): void {
    const arr = Array.from(files)
    for (const file of arr) {
      if (!file.type.startsWith('image/')) continue
      const localUrl = URL.createObjectURL(file)
      const entry: PendingImage = { localUrl, uploading: true }
      this.pendingImages.update((imgs) => [...imgs, entry])
      firstValueFrom(this.chatSvc.uploadImage(file)).then(({ id, mime_type }) => {
        this.pendingImages.update((imgs) =>
          imgs.map((img) => (img.localUrl === localUrl ? { ...img, id, mime_type, uploading: false } : img)),
        )
      }).catch(() => {
        this.pendingImages.update((imgs) => imgs.filter((img) => img.localUrl !== localUrl))
      })
    }
  }

  removeImage(img: PendingImage): void {
    URL.revokeObjectURL(img.localUrl)
    this.pendingImages.update((imgs) => imgs.filter((i) => i.localUrl !== img.localUrl))
  }

  onPaste(event: ClipboardEvent): void {
    const items = event.clipboardData?.items
    if (!items) return
    const imageFiles: File[] = []
    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (file) imageFiles.push(file)
      }
    }
    if (imageFiles.length > 0) {
      event.preventDefault()
      this.attachImages(imageFiles)
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault()
  }

  onDrop(event: DragEvent): void {
    event.preventDefault()
    const files = event.dataTransfer?.files
    if (files) this.attachImages(files)
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
