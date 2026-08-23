import { AfterViewInit, Component, computed, effect, ElementRef, HostListener, inject, input, output, signal, untracked, ViewChild } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { takeUntilDestroyed } from '@angular/core/rxjs-interop'
import { firstValueFrom } from 'rxjs'
import { ChatService } from '../../services/chat.service'
import { ApiService } from '../../services/api.service'
import { VoiceDictationService } from '../../services/voice-dictation.service'
import { AppStatusService } from '../../services/app-status.service'
import { ConversationMode, PendingImage, RagCommandName, SlashCommand, Workflow } from '../../types/message-types'
import { SlashCommandPaletteComponent } from '../slash-command-palette/slash-command-palette.component'
import { FileMentionPickerComponent } from '../file-mention-picker/file-mention-picker.component'

@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [CommonModule, FormsModule, SlashCommandPaletteComponent, FileMentionPickerComponent],
  templateUrl: './chat-input.component.html',
  styleUrls: ['./chat-input.component.scss'],
})
export class ChatInputComponent implements AfterViewInit {
  private chatSvc = inject(ChatService)
  private api = inject(ApiService)
  readonly voiceSvc = inject(VoiceDictationService)
  readonly appStatus = inject(AppStatusService)

  @ViewChild('textarea') private _textareaRef!: ElementRef<HTMLTextAreaElement>
  @ViewChild(SlashCommandPaletteComponent) private _palette?: SlashCommandPaletteComponent
  @ViewChild(FileMentionPickerComponent) private _filePicker?: FileMentionPickerComponent

  // True when the outer context is processing a response and sending is blocked.
  // While busy, the send button is replaced by a stop button.
  readonly busy = input(false)

  // Emitted when the user clicks the stop button during a busy state.
  // The outer component decides what to cancel (agent run, streaming response, etc.).
  readonly stopRequested = output<void>()

  readonly submitted = output<{ text: string; imageIds: string[]; workflowName?: string; ragCommand?: RagCommandName; commandLabel?: string }>()

  readonly currentInput = signal('')
  readonly pendingImages = signal<PendingImage[]>([])
  readonly isUploading = computed(() => this.pendingImages().some((p) => p.uploading))

  // Slash command palette
  readonly paletteOpen = signal(false)
  readonly availableWorkflows = signal<Workflow[]>([])
  private _workflowsLoaded = false
  private _pendingWorkflowName = signal<string | undefined>(undefined)
  private _pendingRagCommand = signal<RagCommandName | undefined>(undefined)
  /** The command token (mode name or workflow name) last chosen, kept only so it can be echoed into the sent message for display. */
  private _pendingCommandLabel = signal<string | undefined>(undefined)

  /** The text after the leading '/' when the palette is open (used for filtering). */
  readonly paletteFilter = computed(() => {
    const input = this.currentInput()
    if (!input.startsWith('/')) {
      return ''
    }
    // Only filter on the command token (before the first space).
    const spaceIndex = input.indexOf(' ')
    return spaceIndex === -1 ? input.slice(1) : input.slice(1, spaceIndex)
  })

  // File mention picker — triggered by '@' anywhere in the input.
  readonly _workspacePath = computed(() => this.chatSvc.currentConversationSettings().working_directory)

  /** When a bare '@…' (no whitespace after '@') exists at the end of the input, returns its position and filter text. */
  readonly fileMentionContext = computed<{ atIndex: number; filterText: string } | null>(() => {
    if (this._workspacePath() === null) {
      return null
    }
    const input = this.currentInput()
    const lastAt = input.lastIndexOf('@')
    if (lastAt === -1) {
      return null
    }
    const afterAt = input.slice(lastAt + 1)
    if (/\s/.test(afterAt)) {
      return null
    }
    return { atIndex: lastAt, filterText: afterAt }
  })

  readonly fileMentionOpen = computed(() => this.fileMentionContext() !== null)

  // Slash commands whose parameter is a path where a directory (not just a file) is the natural
  // target — the @ picker offers directories alongside files only while typing one of these.
  private readonly DIRECTORY_PARAM_COMMANDS: string[] = ['rag-index']

  readonly mentionIncludeDirs = computed(() => {
    const input = this.currentInput()
    if (!input.startsWith('/')) {
      return false
    }
    const spaceIndex = input.indexOf(' ')
    if (spaceIndex === -1) {
      return false
    }
    const token = input.slice(1, spaceIndex)
    return this.DIRECTORY_PARAM_COMMANDS.includes(token)
  })

  // Text that was in the input before recording started; partials/final are appended to it.
  private _startPrefix = ''
  // Timer ID for the Alt hold-to-record 500ms delay.
  private _altTimer: ReturnType<typeof setTimeout> | null = null

  constructor() {
    // Open/close palette based on whether the input starts with '/' and has no space yet.
    effect(() => {
      const input = this.currentInput()
      const shouldOpen = input.startsWith('/') && !input.includes(' ')
      untracked(() => {
        if (shouldOpen && !this._workflowsLoaded) {
          this._workflowsLoaded = true
          this.api.get_workflows().subscribe((workflows) => this.availableWorkflows.set(workflows))
        }
        if (shouldOpen !== this.paletteOpen()) {
          this.paletteOpen.set(shouldOpen)
          if (shouldOpen) {
            this._palette?.resetIndex()
          }
        }
      })
    })

    // Auto-grow the textarea (up to its CSS max-height, where it scrolls internally instead) as
    // content spans multiple lines, and auto-scroll to the bottom while dictating so newly
    // appended speech stays visible. queueMicrotask so scrollHeight is read after the browser
    // has laid out the DOM update ngModel just triggered, not the stale pre-update value.
    effect(() => {
      this.currentInput()
      const isSpeechMode = this.voiceSvc.isRecording() || this.voiceSvc.isTranscribing()
      queueMicrotask(() => {
        const el = this._textareaRef?.nativeElement
        if (!el) {
          return
        }
        el.style.height = 'auto'
        el.style.height = `${el.scrollHeight}px`
        if (isSpeechMode) {
          el.scrollTop = el.scrollHeight
        }
      })
    })

    // Append each partial transcript to the prefix captured at recording start.
    effect(() => {
      const partial = this.voiceSvc.partialText()
      if (!partial) {
        return
      }
      untracked(() => {
        this.currentInput.set((this._startPrefix + ' ' + partial).trim())
      })
    })

    this.voiceSvc.finalResult$.pipe(takeUntilDestroyed()).subscribe(({ raw }) => {
      if (raw) {
        this.currentInput.set((this._startPrefix + ' ' + raw).trim())
      }
    })
  }

  ngAfterViewInit(): void {
    this._textareaRef?.nativeElement.focus()
  }

  // -------------------------------------------------------------------------
  // Hold-to-record: mousedown on button / Alt hold on keyboard
  // -------------------------------------------------------------------------

  onMicMousedown(event: MouseEvent): void {
    event.preventDefault()  // keep focus in the textarea
    this._textareaRef?.nativeElement.focus()
    if (!this.voiceSvc.isRecording()) {
      this._startPrefix = this.currentInput()
      this.voiceSvc.startRecording().catch(() => {})
    }
  }

  @HostListener('document:mouseup')
  onDocumentMouseup(): void {
    if (this.voiceSvc.isRecording()) {
      this.voiceSvc.stopRecording()
    }
  }

  // Alt chosen for hold-to-record; Ctrl+Space is an alternative if Alt conflicts with OS/browser shortcuts.
  // True while AltGraph is physically held down.
  private _altHeld = false

  @HostListener('document:keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'L' && event.ctrlKey && event.shiftKey && !event.altKey) {
      event.preventDefault()
      if (!this.voiceSvc.isRecording() && !this.voiceSvc.isTranscribing()) {
        this.voiceSvc.toggleLang()
      }
      return
    }

    // Windows synthesizes a phantom Control keydown alongside every real AltGraph keydown —
    // both on initial press and on every OS key-repeat pulse while AltGraph is held. Ignore it
    // outright so it doesn't hit the "any other key cancels" branch below and clear/cancel the
    // hold-to-record timer before it ever fires.
    if (event.key === 'Control' && event.getModifierState('AltGraph')) {
      return
    }

    if (event.key === 'AltGraph') {
      if (this.voiceSvc.isRecording() || this._altTimer !== null) {
        return
      }
      // Block if focus is in a text input other than our own textarea.
      const active = document.activeElement
      const isOtherInput = active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement
      if (isOtherInput && active !== this._textareaRef?.nativeElement) {
        return
      }
      event.preventDefault()
      this._altHeld = true
      this._altTimer = setTimeout(() => {
        this._altTimer = null
        // Re-check: this fires 500ms after being scheduled, and isRecording() may have changed
        // in the meantime (e.g. the mouse button was used instead) — starting anyway would
        // orphan whatever session is already active. VoiceDictationService.startRecording()
        // also refuses this itself, but no need to even attempt it (getUserMedia) here.
        if (this.voiceSvc.isRecording()) {
          return
        }
        this._startPrefix = this.currentInput()
        this.voiceSvc.startRecording().catch(() => {})
      }, 500)
      return
    }

    // Any other key while AltGraph is held → cancel.
    if (this._altHeld) {
      if (this._altTimer !== null) {
        clearTimeout(this._altTimer)
        this._altTimer = null
      } else {
        this.voiceSvc.cancelRecording()
      }
    }
  }

  @HostListener('document:keyup', ['$event'])
  onKeyup(event: KeyboardEvent): void {
    if (event.key !== 'AltGraph') {
      return
    }
    this._altHeld = false
    if (this._altTimer !== null) {
      clearTimeout(this._altTimer)
      this._altTimer = null
      return
    }
    this.voiceSvc.stopRecording()
  }

  onFileMentionSelected(absolutePath: string): void {
    const context = this.fileMentionContext()
    if (context === null) {
      return
    }
    const input = this.currentInput()
    // The leading '@' was only the trigger for this picker — drop it so the inserted path is
    // clean for anything that reads this text literally (the RAG slash commands, or the model
    // itself), instead of looking like the path starts with an '@' character.
    this.currentInput.set(input.slice(0, context.atIndex) + absolutePath + ' ')
    this._textareaRef?.nativeElement.focus()
  }

  onSlashCommandSelected(command: SlashCommand): void {
    this.paletteOpen.set(false)
    // Strip the command token from the input, keep any trailing text as the message body.
    const raw = this.currentInput()
    const spaceIndex = raw.indexOf(' ')
    const remainder = spaceIndex === -1 ? '' : raw.slice(spaceIndex + 1)
    this.currentInput.set(remainder)
    this._textareaRef?.nativeElement.focus()
    this._pendingCommandLabel.set(command.label)

    if (command.type === 'mode') {
      const settings = this.chatSvc.currentConversationSettings()
      this.chatSvc.updateConversationSettings({ ...settings, mode: command.value as ConversationMode }).subscribe()
    } else if (command.type === 'workflow') {
      this._pendingWorkflowName.set(command.value)
    } else if (command.type === 'rag') {
      this._pendingRagCommand.set(command.value)
    }
  }

  private readonly KNOWN_MODES: ConversationMode[] = ['standard', 'auto', 'plan', 'yolo']
  private readonly KNOWN_RAG_COMMANDS: RagCommandName[] = ['rag-index', 'rag-search']

  async sendMessage(event: Event | null): Promise<void> {
    if (event && (event as KeyboardEvent).shiftKey) {
      return
    }
    // Enter while the palette is open completes the command token the same way Tab does —
    // it does not execute/apply it yet. Every slash command can take trailing text (a message
    // body, a path, a query), so a single Enter can't tell whether the user is done typing;
    // completing the token and requiring a second Enter (now with the palette closed) is
    // consistent and never silently swallows a parameter the user was about to type.
    if (this.paletteOpen()) {
      event?.preventDefault()
      this._completeActiveSlashCommand()
      return
    }
    if (this.fileMentionOpen()) {
      event?.preventDefault()
      this._filePicker?.selectActive()
      return
    }
    event?.preventDefault()

    let messageText = this.currentInput().trim()
    let workflowName = this._pendingWorkflowName()
    let ragCommand = this._pendingRagCommand()
    let commandLabel = this._pendingCommandLabel()

    // Parse a leading /command prefix if the user typed or Tab-completed it.
    // This is skipped when the command was already consumed via palette Enter.
    if (messageText.startsWith('/') && workflowName === undefined && ragCommand === undefined) {
      const spaceIndex = messageText.indexOf(' ')
      const token = spaceIndex === -1 ? messageText.slice(1) : messageText.slice(1, spaceIndex)
      const remainder = spaceIndex === -1 ? '' : messageText.slice(spaceIndex + 1).trim()
      if (this.KNOWN_MODES.includes(token as ConversationMode)) {
        const settings = this.chatSvc.currentConversationSettings()
        this.chatSvc.updateConversationSettings({ ...settings, mode: token as ConversationMode }).subscribe()
        messageText = remainder
        commandLabel = token
      } else if (this.KNOWN_RAG_COMMANDS.includes(token as RagCommandName)) {
        ragCommand = token as RagCommandName
        messageText = remainder
        commandLabel = token
      } else if (token.length > 0) {
        workflowName = token
        messageText = remainder
        commandLabel = token
      }
    }

    if (!messageText && this.pendingImages().length === 0 && workflowName === undefined && ragCommand === undefined) {
      this.currentInput.set('')
      this._pendingCommandLabel.set(undefined)
      return
    }
    const imageIds = this.pendingImages()
      .filter((p) => !p.uploading && p.id)
      .map((p) => p.id!)
    this.currentInput.set('')
    this.pendingImages.set([])
    this._pendingWorkflowName.set(undefined)
    this._pendingRagCommand.set(undefined)
    this._pendingCommandLabel.set(undefined)
    this.voiceSvc.dismissCorrection()
    this.submitted.emit({ text: messageText, imageIds, workflowName, ragCommand, commandLabel })
  }

  attachImages(files: FileList | File[]): void {
    for (const file of Array.from(files)) {
      if (!file.type.startsWith('image/')) {
        continue
      }
      const localUrl = URL.createObjectURL(file)
      const entry: PendingImage = { localUrl, uploading: true }
      this.pendingImages.update((imgs) => [...imgs, entry])
      firstValueFrom(this.chatSvc.uploadImage(file))
        .then(({ id, mime_type }) => {
          this.pendingImages.update((imgs) =>
            imgs.map((img) => (img.localUrl === localUrl ? { ...img, id, mime_type, uploading: false } : img)),
          )
        })
        .catch(() => {
          this.pendingImages.update((imgs) => imgs.filter((img) => img.localUrl !== localUrl))
        })
    }
  }

  useCorrection(corrected: string): void {
    this.currentInput.set((this._startPrefix + ' ' + corrected).trim())
    this.voiceSvc.dismissCorrection()
    this._textareaRef?.nativeElement.focus()
  }

  removeImage(img: PendingImage): void {
    URL.revokeObjectURL(img.localUrl)
    this.pendingImages.update((imgs) => imgs.filter((i) => i.localUrl !== img.localUrl))
  }

  /** Fill in the highlighted palette command's token with a trailing space so the user can type
   * its parameter — does not apply/execute the command yet. The palette closes automatically
   * because the space is detected by the open/close effect; the command itself is parsed at
   * send time (see sendMessage's leading-/command parsing). Shared by Tab and Enter. */
  private _completeActiveSlashCommand(): void {
    const item = this._palette?.getActiveItem()
    if (item === undefined) {
      return
    }
    this.currentInput.set('/' + item.label + ' ')
    queueMicrotask(() => {
      const el = this._textareaRef?.nativeElement
      if (el) {
        el.selectionStart = el.selectionEnd = el.value.length
      }
    })
  }

  onTextareaKeydown(event: KeyboardEvent): void {
    if (this.paletteOpen()) {
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        this._palette?.navigateUp()
      } else if (event.key === 'ArrowDown') {
        event.preventDefault()
        this._palette?.navigateDown()
      } else if (event.key === 'Tab') {
        event.preventDefault()
        this._completeActiveSlashCommand()
      } else if (event.key === 'Escape') {
        event.preventDefault()
        this.paletteOpen.set(false)
        this.currentInput.set('')
      }
      return
    }

    if (this.fileMentionOpen()) {
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        this._filePicker?.navigateUp()
      } else if (event.key === 'ArrowDown') {
        event.preventDefault()
        this._filePicker?.navigateDown()
      } else if (event.key === 'Tab') {
        // Without this, Tab falls through to native focus traversal and jumps out of the
        // textarea onto the next focusable control (the STT language switch button) instead of
        // picking the highlighted entry the way it does in the slash command palette.
        event.preventDefault()
        this._filePicker?.selectActive()
      } else if (event.key === 'Escape') {
        event.preventDefault()
        // Remove the '@' and filter text, leaving the rest of the message intact.
        const context = this.fileMentionContext()
        if (context !== null) {
          this.currentInput.set(this.currentInput().slice(0, context.atIndex))
        }
      }
      return
    }
  }

  onPaste(event: ClipboardEvent): void {
    const items = event.clipboardData?.items
    if (!items) {
      return
    }
    const imageFiles: File[] = []
    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (file) {
          imageFiles.push(file)
        }
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
    if (files) {
      this.attachImages(files)
    }
  }

}
