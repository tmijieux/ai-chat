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
    this._correctionRequestId++
    this._lastCorrection.set(null)
  }

  // Incremented each time a correction request is started or invalidated (send, dismiss, new
  // recording). The post_correct callback compares its captured ID against this value to discard
  // responses that arrived after the user already sent the message.
  private _correctionRequestId = 0
  private _mediaRecorder: MediaRecorder | null = null
  private _stream: MediaStream | null = null
  private _audioChunks: Blob[] = []
  private _chunksAtLastPartial = 0
  private _partialSub: Subscription | null = null
  // Set in onstop; prevents new partials from starting after recording ends.
  private _stopping = false
  // Set in onstop when we want the next partial completion to chain to /api/correct.
  private _finalPending = false
  // True while getUserMedia is in-flight; lets stopRecording() signal intent before the stream exists.
  private _acquiring = false

  constructor() {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.stopRecording()
        this._acquiring = false
      }
    })
  }

  async startRecording(): Promise<void> {
    if (this._acquiring) {
      return
    }
    this._acquiring = true
    this._stopping = false
    this._finalPending = false
    this._audioChunks = []
    this._chunksAtLastPartial = 0
    this._partialText.set('')
    this._correctionRequestId++
    this._lastCorrection.set(null)

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } finally {
      this._acquiring = false
    }

    // stopRecording() was called while getUserMedia was pending — discard and bail.
    if (this._stopping) {
      stream!.getTracks().forEach((t) => t.stop())
      return
    }

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
    if (this._acquiring) {
      this._stopping = true
      return
    }
    if (!this._isRecording()) {
      return
    }
    this._mediaRecorder?.stop()
  }

  cancelRecording(): void {
    if (this._acquiring) {
      this._stopping = true
      return
    }
    if (!this._isRecording()) {
      return
    }
    // Override onstop so it releases the stream without transcribing.
    this._mediaRecorder!.onstop = () => {
      this._stream?.getTracks().forEach((t) => t.stop())
      this._isRecording.set(false)
      this._isTranscribing.set(false)
      this._partialText.set('')
      this._partialSub?.unsubscribe()
      this._partialSub = null
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
    const requestId = this._correctionRequestId
    firstValueFrom(this.api.post_correct(raw, 'fr'))
      .then(({ text }) => {
        if (text && text !== raw && this._correctionRequestId === requestId) {
          this._lastCorrection.set({ raw, corrected: text })
        }
      })
      .catch(() => {})
  }
}
