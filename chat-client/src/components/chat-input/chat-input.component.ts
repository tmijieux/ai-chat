import { Component, computed, effect, ElementRef, HostListener, inject, input, output, signal, untracked, ViewChild } from '@angular/core'
import { CommonModule } from '@angular/common'
import { FormsModule } from '@angular/forms'
import { takeUntilDestroyed } from '@angular/core/rxjs-interop'
import { firstValueFrom } from 'rxjs'
import { ChatService } from '../../services/chat.service'
import { VoiceDictationService } from '../../services/voice-dictation.service'
import { PendingImage } from '../../types/message-types'

@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat-input.component.html',
  styleUrls: ['./chat-input.component.scss'],
})
export class ChatInputComponent {
  private chatSvc = inject(ChatService)
  readonly voiceSvc = inject(VoiceDictationService)

  @ViewChild('textarea') private _textareaRef!: ElementRef<HTMLTextAreaElement>

  // True when the outer context is processing a response and sending is blocked.
  // While busy, the send button is replaced by a stop button.
  readonly busy = input(false)

  // Emitted when the user clicks the stop button during a busy state.
  // The outer component decides what to cancel (agent run, streaming response, etc.).
  readonly stopRequested = output<void>()

  readonly submitted = output<{ text: string; imageIds: string[] }>()

  readonly currentInput = signal('')
  readonly pendingImages = signal<PendingImage[]>([])
  readonly isUploading = computed(() => this.pendingImages().some((p) => p.uploading))

  // Text that was in the input before recording started; partials/final are appended to it.
  private _startPrefix = ''
  // Timer ID for the Alt hold-to-record 500ms delay.
  private _altTimer: ReturnType<typeof setTimeout> | null = null

  constructor() {
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
  @HostListener('document:keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Alt') {
      return
    }
    if (this.voiceSvc.isRecording() || this._altTimer !== null) {
      return
    }
    if (!this._isTextareaFocused()) {
      return
    }
    event.preventDefault()
    this._altTimer = setTimeout(() => {
      this._altTimer = null
      this._startPrefix = this.currentInput()
      this.voiceSvc.startRecording().catch(() => {})
    }, 500)
  }

  @HostListener('document:keyup', ['$event'])
  onKeyup(event: KeyboardEvent): void {
    if (event.key !== 'Alt') {
      return
    }
    if (this._altTimer !== null) {
      clearTimeout(this._altTimer)
      this._altTimer = null
      return
    }
    this.voiceSvc.stopRecording()
  }

  private _isTextareaFocused(): boolean {
    return document.activeElement === this._textareaRef?.nativeElement
  }

  async sendMessage(event: Event | null): Promise<void> {
    if (event && (event as KeyboardEvent).shiftKey) {
      return
    }
    event?.preventDefault()
    const input = this.currentInput().trim()
    if (!input && this.pendingImages().length === 0) {
      return
    }
    const imageIds = this.pendingImages()
      .filter((p) => !p.uploading && p.id)
      .map((p) => p.id!)
    this.currentInput.set('')
    this.pendingImages.set([])
    this.submitted.emit({ text: input, imageIds })
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
    this.currentInput.set(corrected)
    this.voiceSvc.dismissCorrection()
  }

  removeImage(img: PendingImage): void {
    URL.revokeObjectURL(img.localUrl)
    this.pendingImages.update((imgs) => imgs.filter((i) => i.localUrl !== img.localUrl))
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
