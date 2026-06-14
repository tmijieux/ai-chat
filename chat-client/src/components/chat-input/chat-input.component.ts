import { AfterViewInit, Component, computed, effect, ElementRef, HostListener, inject, input, output, signal, untracked, ViewChild } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { takeUntilDestroyed } from '@angular/core/rxjs-interop'
import { firstValueFrom } from 'rxjs'
import { ChatService } from '../../services/chat.service'
import { ApiService } from '../../services/api.service'
import { VoiceDictationService } from '../../services/voice-dictation.service'
import { AppStatusService } from '../../services/app-status.service'
import { ConversationMode, PendingImage, SlashCommand, Workflow } from '../../types/message-types'
import { SlashCommandPaletteComponent } from '../slash-command-palette/slash-command-palette.component'

@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [CommonModule, FormsModule, SlashCommandPaletteComponent],
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

  // True when the outer context is processing a response and sending is blocked.
  // While busy, the send button is replaced by a stop button.
  readonly busy = input(false)

  // Emitted when the user clicks the stop button during a busy state.
  // The outer component decides what to cancel (agent run, streaming response, etc.).
  readonly stopRequested = output<void>()

  readonly submitted = output<{ text: string; imageIds: string[]; workflowName?: string }>()

  readonly currentInput = signal('')
  readonly pendingImages = signal<PendingImage[]>([])
  readonly isUploading = computed(() => this.pendingImages().some((p) => p.uploading))

  // Slash command palette
  readonly paletteOpen = signal(false)
  readonly availableWorkflows = signal<Workflow[]>([])
  private _workflowsLoaded = false
  private _pendingWorkflowName = signal<string | undefined>(undefined)

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

  onSlashCommandSelected(command: SlashCommand): void {
    this.paletteOpen.set(false)
    // Strip the command token from the input, keep any trailing text as the message body.
    const raw = this.currentInput()
    const spaceIndex = raw.indexOf(' ')
    const remainder = spaceIndex === -1 ? '' : raw.slice(spaceIndex + 1)
    this.currentInput.set(remainder)
    this._textareaRef?.nativeElement.focus()

    if (command.type === 'mode') {
      const settings = this.chatSvc.currentConversationSettings()
      this.chatSvc.updateConversationSettings({ ...settings, mode: command.value as ConversationMode }).subscribe()
    } else if (command.type === 'workflow') {
      this._pendingWorkflowName.set(command.value)
    }
  }

  private readonly KNOWN_MODES: ConversationMode[] = ['standard', 'auto', 'plan', 'yolo']

  async sendMessage(event: Event | null): Promise<void> {
    if (event && (event as KeyboardEvent).shiftKey) {
      return
    }
    // Route Enter to palette selection when palette is open.
    if (this.paletteOpen()) {
      event?.preventDefault()
      this._palette?.selectActive()
      return
    }
    event?.preventDefault()

    let messageText = this.currentInput().trim()
    let workflowName = this._pendingWorkflowName()

    // Parse a leading /command prefix if the user typed or Tab-completed it.
    // This is skipped when the command was already consumed via palette Enter.
    if (messageText.startsWith('/') && workflowName === undefined) {
      const spaceIndex = messageText.indexOf(' ')
      const token = spaceIndex === -1 ? messageText.slice(1) : messageText.slice(1, spaceIndex)
      const remainder = spaceIndex === -1 ? '' : messageText.slice(spaceIndex + 1).trim()
      if (this.KNOWN_MODES.includes(token as ConversationMode)) {
        const settings = this.chatSvc.currentConversationSettings()
        this.chatSvc.updateConversationSettings({ ...settings, mode: token as ConversationMode }).subscribe()
        messageText = remainder
      } else if (token.length > 0) {
        workflowName = token
        messageText = remainder
      }
    }

    if (!messageText && this.pendingImages().length === 0) {
      return
    }
    const imageIds = this.pendingImages()
      .filter((p) => !p.uploading && p.id)
      .map((p) => p.id!)
    this.currentInput.set('')
    this.pendingImages.set([])
    this._pendingWorkflowName.set(undefined)
    this.voiceSvc.dismissCorrection()
    this.submitted.emit({ text: messageText, imageIds, workflowName })
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

  onTextareaKeydown(event: KeyboardEvent): void {
    if (!this.paletteOpen()) {
      return
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault()
      this._palette?.navigateUp()
    } else if (event.key === 'ArrowDown') {
      event.preventDefault()
      this._palette?.navigateDown()
    } else if (event.key === 'Tab') {
      event.preventDefault()
      const item = this._palette?.getActiveItem()
      if (item !== undefined) {
        // Fill in the command token with a trailing space so the user can type the prompt.
        // The palette closes automatically because the space is detected by the effect.
        // The command itself is parsed at send time.
        this.currentInput.set('/' + item.label + ' ')
        queueMicrotask(() => {
          const el = this._textareaRef?.nativeElement
          if (el) {
            el.selectionStart = el.selectionEnd = el.value.length
          }
        })
      }
    } else if (event.key === 'Escape') {
      event.preventDefault()
      this.paletteOpen.set(false)
      this.currentInput.set('')
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
