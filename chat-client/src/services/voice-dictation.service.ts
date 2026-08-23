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

// All mutable state belonging to one hold-to-record instance, bundled so a closure captures the
// session it was created for directly instead of reading shared fields off the service. This is
// what makes late-firing events from a stop()-ed recorder harmless: MediaRecorder.stop() flushes
// its final ondataavailable/onstop asynchronously, after the call returns — if a new session has
// already started by then, an old handler still writing into shared `this.` fields would corrupt
// the new session's state (a headerless tail chunk landing ahead of the new recorder's real
// header; a stale onstop stopping the new session's mic tracks). Writing into an abandoned
// session object instead is a no-op nobody ever reads again.
interface RecordingSession {
  recorder: MediaRecorder
  stream: MediaStream
  chunks: Blob[]
  chunksAtLastPartial: number
  partialSub: Subscription | null
  stopping: boolean
  finalPending: boolean
}

@Injectable({ providedIn: 'root' })
export class VoiceDictationService {
  private api = inject(ApiService)

  private _isRecording = signal(false)
  private _isTranscribing = signal(false)
  private _partialText = signal('')
  private _lastCorrection = signal<DictationCorrection | null>(null)
  private _lang = signal<'fr' | 'en'>('fr')

  readonly isRecording = this._isRecording.asReadonly()
  readonly isTranscribing = this._isTranscribing.asReadonly()
  readonly partialText = this._partialText.asReadonly()
  readonly lastCorrection = this._lastCorrection.asReadonly()
  readonly lang = this._lang.asReadonly()

  toggleLang(): void {
    this._lang.set(this._lang() === 'fr' ? 'en' : 'fr')
  }
  readonly finalResult$ = new Subject<DictationResult>()

  dismissCorrection(): void {
    this._correctionRequestId++
    this._lastCorrection.set(null)
  }

  // Incremented each time a correction request is started or invalidated (send, dismiss, new
  // recording). The post_correct callback compares its captured ID against this value to discard
  // responses that arrived after the user already sent the message.
  private _correctionRequestId = 0
  private _session: RecordingSession | null = null
  // True while getUserMedia is in-flight; lets stopRecording() signal intent before the stream
  // (and thus a RecordingSession) exists.
  private _acquiring = false
  // Set when stop/cancel is requested while still acquiring — no MediaRecorder exists yet to
  // stop, so startRecording checks this once acquisition completes instead.
  private _stopRequestedWhileAcquiring = false

  constructor() {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.stopRecording()
        this._acquiring = false
      }
    })
  }

  async startRecording(): Promise<void> {
    // Guards against orphaning an already-active session: a caller (e.g. a delayed hold-timer)
    // may re-validate "not recording yet" only when it schedules the call, not when it actually
    // fires — by then a session can already be running. Silently reassigning `_session` here
    // would leave that session's recorder running with nothing left to stop it, still updating
    // the shared partialText/isRecording signals indefinitely.
    if (this._acquiring || this._session !== null) {
      return
    }
    this._acquiring = true
    this._stopRequestedWhileAcquiring = false
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
    if (this._stopRequestedWhileAcquiring) {
      stream!.getTracks().forEach((t) => t.stop())
      return
    }

    // Explicit mimeType, not the browser default: Firefox otherwise defaults audio-only capture
    // to Ogg, whose page-based structure isn't safely truncatable at an arbitrary chunk boundary
    // the way WebM/Matroska's cluster-based structure is — a real problem here since every
    // partial fire re-slices the growing chunk buffer from the start (see _firePartial below).
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : undefined
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    const session: RecordingSession = {
      recorder,
      stream,
      chunks: [],
      chunksAtLastPartial: 0,
      partialSub: null,
      stopping: false,
      finalPending: false,
    }
    this._session = session

    recorder.ondataavailable = (e) => {
      session.chunks.push(e.data)
      this._maybeFirePartial(session)
    }

    recorder.onstop = () => {
      if (this._session === session) {
        this._session = null
      }
      session.stream.getTracks().forEach((t) => t.stop())
      this._isRecording.set(false)
      session.stopping = true

      if (session.chunks.length < 2) {
        return
      }

      this._isTranscribing.set(true)

      if (session.partialSub !== null) {
        // A partial is already in-flight — mark it as final so it chains to /api/correct on completion.
        session.finalPending = true
      } else {
        // Nothing in-flight — fire one transcription now with all audio.
        this._firePartial(session, true)
      }
    }

    recorder.start(200)
    this._isRecording.set(true)
  }

  stopRecording(): void {
    if (this._acquiring) {
      this._stopRequestedWhileAcquiring = true
      return
    }
    if (!this._isRecording()) {
      return
    }
    this._session?.recorder.stop()
  }

  cancelRecording(): void {
    if (this._acquiring) {
      this._stopRequestedWhileAcquiring = true
      return
    }
    if (!this._isRecording()) {
      return
    }
    const session = this._session
    if (session === null) {
      return
    }
    // Override onstop so it releases the stream without transcribing.
    session.recorder.onstop = () => {
      if (this._session === session) {
        this._session = null
      }
      session.stream.getTracks().forEach((t) => t.stop())
      this._isRecording.set(false)
      this._isTranscribing.set(false)
      this._partialText.set('')
      session.partialSub?.unsubscribe()
      session.partialSub = null
    }
    session.recorder.stop()
  }

  private _maybeFirePartial(session: RecordingSession): void {
    if (session.stopping) {
      return
    }
    if (session.partialSub !== null) {
      return
    }
    if (session.chunks.length - session.chunksAtLastPartial < 3) {
      return
    }
    this._firePartial(session, false)
  }

  private _firePartial(session: RecordingSession, isFinal: boolean): void {
    session.chunksAtLastPartial = session.chunks.length
    const blob = new Blob(session.chunks, { type: 'audio/webm' })

    session.partialSub = this.api.post_transcribe(blob, this._lang()).subscribe({
      next: ({ text }) => {
        session.partialSub = null
        if (isFinal) {
          this._finalize(text)
          return
        }
        if (session.finalPending) {
          // This partial's blob predates the last ondataavailable chunks — re-fire with complete audio.
          session.finalPending = false
          this._firePartial(session, true)
          return
        }
        if (text) {
          this._partialText.set(text)
        }
        // Re-check immediately: more chunks may have arrived while Whisper was running.
        this._maybeFirePartial(session)
      },
      error: () => {
        session.partialSub = null
        if (isFinal || session.finalPending) {
          session.finalPending = false
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
    firstValueFrom(this.api.post_correct(raw, this._lang()))
      .then(({ text }) => {
        if (text && text !== raw && this._correctionRequestId === requestId) {
          this._lastCorrection.set({ raw, corrected: text })
        }
      })
      .catch(() => {})
  }
}
