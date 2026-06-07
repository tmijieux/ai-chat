import { inject, Injectable, signal } from '@angular/core'
import { Subject, Subscription, firstValueFrom } from 'rxjs'
import { ApiService } from './api.service'

export interface DictationResult {
  raw: string
  corrected: string
}

export interface DictationCorrection {
  raw: string
  corrected: string
}

@Injectable({ providedIn: 'root' })
export class VoiceDictationService {
  private api = inject(ApiService)

  private _isRecording = signal(false)
  private _isTranscribing = signal(false)
  private _partialText = signal('')
  private _lastCorrection = signal<DictationCorrection | null>(null)

  readonly isRecording = this._isRecording.asReadonly()
  readonly isTranscribing = this._isTranscribing.asReadonly()
  readonly partialText = this._partialText.asReadonly()
  readonly lastCorrection = this._lastCorrection.asReadonly()
  readonly finalResult$ = new Subject<DictationResult>()

  dismissCorrection(): void {
    this._lastCorrection.set(null)
  }

  private _mediaRecorder: MediaRecorder | null = null
  private _stream: MediaStream | null = null
  private _audioChunks: Blob[] = []
  private _chunksAtLastPartial = 0
  private _partialSub: Subscription | null = null
  // Set in onstop; prevents new partials from starting after recording ends.
  private _stopping = false
  // Set in onstop when we want the next partial completion to chain to /api/correct.
  private _finalPending = false

  async startRecording(): Promise<void> {
    this._stopping = false
    this._finalPending = false
    this._audioChunks = []
    this._chunksAtLastPartial = 0
    this._partialText.set('')
    this._lastCorrection.set(null)

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    this._stream = stream
    this._mediaRecorder = new MediaRecorder(stream)

    this._mediaRecorder.ondataavailable = (e) => {
      this._audioChunks.push(e.data)
      this._maybeFirePartial()
    }

    this._mediaRecorder.onstop = () => {
      this._stream?.getTracks().forEach((t) => t.stop())
      this._isRecording.set(false)
      this._stopping = true

      if (this._audioChunks.length < 2) {
        return
      }

      this._isTranscribing.set(true)

      if (this._partialSub !== null) {
        // A partial is already in-flight — mark it as final so it chains to /api/correct on completion.
        this._finalPending = true
      } else {
        // Nothing in-flight — fire one transcription now with all audio.
        this._firePartial(true)
      }
    }

    this._mediaRecorder.start(200)
    this._isRecording.set(true)
  }

  stopRecording(): void {
    if (!this._isRecording()) {
      return
    }
    this._mediaRecorder?.stop()
  }

  private _maybeFirePartial(): void {
    if (this._stopping) {
      return
    }
    if (this._partialSub !== null) {
      return
    }
    if (this._audioChunks.length - this._chunksAtLastPartial < 3) {
      return
    }
    this._firePartial(false)
  }

  private _firePartial(isFinal: boolean): void {
    this._chunksAtLastPartial = this._audioChunks.length
    const blob = new Blob(this._audioChunks, { type: 'audio/webm' })

    this._partialSub = this.api.post_transcribe(blob, 'fr').subscribe({
      next: ({ text }) => {
        this._partialSub = null
        if (isFinal) {
          this._finalize(text)
          return
        }
        if (this._finalPending) {
          // This partial's blob predates the last ondataavailable chunks — re-fire with complete audio.
          this._finalPending = false
          this._firePartial(true)
          return
        }
        if (text) {
          this._partialText.set(text)
        }
        // Re-check immediately: more chunks may have arrived while Whisper was running.
        this._maybeFirePartial()
      },
      error: () => {
        this._partialSub = null
        if (isFinal || this._finalPending) {
          this._finalPending = false
          this._isTranscribing.set(false)
        }
      },
    })
  }

  private _finalize(raw: string): void {
    this._isTranscribing.set(false)
    this.finalResult$.next({ raw, corrected: raw })
    if (!raw) { return }
    firstValueFrom(this.api.post_correct(raw, 'fr'))
      .then(({ text }) => {
        if (text && text !== raw) {
          this._lastCorrection.set({ raw, corrected: text })
        }
      })
      .catch(() => {})
  }
}
