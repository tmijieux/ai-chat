import { Component, ElementRef, effect, inject, signal, viewChild } from '@angular/core'
import { FormsModule } from '@angular/forms'
import { SystemAudioTranscriptionService } from '../../services/system-audio-transcription.service'

/**
 * Right-side drawer for System Audio Transcription: pick a source and language, start/stop
 * capture, and read the live transcript. All state lives in SystemAudioTranscriptionService.
 */
@Component({
  selector: 'app-system-audio-panel',
  imports: [FormsModule],
  templateUrl: './system-audio-panel.component.html',
})
export class SystemAudioPanelComponent {
  readonly svc = inject(SystemAudioTranscriptionService)

  /** Curated subset of Whisper's languages — the ones commonly wanted for subtitling foreign
   * web audio, plus auto-detect. Whisper supports far more; this keeps the picker scannable. */
  readonly LANGUAGES: { code: string; label: string }[] = [
    { code: '', label: 'Auto-detect' },
    { code: 'en', label: 'English' },
    { code: 'fr', label: 'French' },
    { code: 'es', label: 'Spanish' },
    { code: 'de', label: 'German' },
    { code: 'it', label: 'Italian' },
    { code: 'pt', label: 'Portuguese' },
    { code: 'nl', label: 'Dutch' },
    { code: 'ru', label: 'Russian' },
    { code: 'uk', label: 'Ukrainian' },
    { code: 'pl', label: 'Polish' },
    { code: 'tr', label: 'Turkish' },
    { code: 'ar', label: 'Arabic' },
    { code: 'fa', label: 'Persian' },
    { code: 'he', label: 'Hebrew' },
    { code: 'hi', label: 'Hindi' },
    { code: 'zh', label: 'Chinese' },
    { code: 'ja', label: 'Japanese' },
    { code: 'ko', label: 'Korean' },
    { code: 'vi', label: 'Vietnamese' },
    { code: 'th', label: 'Thai' },
    { code: 'id', label: 'Indonesian' },
  ]

  readonly copied = signal(false)
  private readonly transcriptBox = viewChild<ElementRef<HTMLDivElement>>('transcript')

  constructor() {
    // Auto-scroll the transcript to the bottom as new text streams in.
    effect(() => {
      this.svc.committedText()
      this.svc.provisionalText()
      const el = this.transcriptBox()?.nativeElement
      if (el !== undefined) {
        queueMicrotask(() => {
          el.scrollTop = el.scrollHeight
        })
      }
    })
  }

  get running(): boolean {
    const state = this.svc.state()
    return state === 'connecting' || state === 'listening' || state === 'transcribing'
  }

  copyTranscript(): void {
    const text = (this.svc.committedText() + ' ' + this.svc.provisionalText()).trim()
    if (text.length === 0) {
      return
    }
    navigator.clipboard.writeText(text).then(() => {
      this.copied.set(true)
      setTimeout(() => this.copied.set(false), 1500)
    })
  }
}
