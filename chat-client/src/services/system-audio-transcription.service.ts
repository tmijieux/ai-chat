import { Injectable, signal } from '@angular/core'
import { Subject } from 'rxjs'

function wsUrl(): string {
  return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/api/system-audio/ws'
}

export interface AudioSource {
  id: string
  label: string
  kind: 'output' | 'input'
}

export type SystemAudioState = 'idle' | 'connecting' | 'listening' | 'transcribing' | 'error'

interface DevicesEvent {
  type: 'devices'
  devices: AudioSource[]
}
interface StatusEvent {
  type: 'status'
  state: 'listening' | 'transcribing'
}
interface CommittedEvent {
  type: 'committed'
  text: string
}
interface ProvisionalEvent {
  type: 'provisional'
  text: string
}
interface ErrorEvent {
  type: 'error'
  message: string
}
type ServerEvent = DevicesEvent | StatusEvent | CommittedEvent | ProvisionalEvent | ErrorEvent

/**
 * Owns the WebSocket to /api/system-audio/ws and all System Audio Transcription display state.
 * Its own service (mirroring AgentService's transport/state split) rather than piled into
 * ChatService or VoiceDictationService — a distinct feature with a distinct source and lifecycle.
 */
@Injectable({ providedIn: 'root' })
export class SystemAudioTranscriptionService {
  readonly panelOpen = signal(false)
  readonly devices = signal<AudioSource[]>([])
  readonly selectedDeviceId = signal<string | null>(null)
  /** Whisper language code, or '' for auto-detect. */
  readonly language = signal<string>('')
  readonly translate = signal<boolean>(false)
  readonly state = signal<SystemAudioState>('idle')
  readonly committedText = signal<string>('')
  readonly provisionalText = signal<string>('')
  readonly errorMessage = signal<string>('')

  /** Emits the current transcript when the user clicks "Insert into message". */
  readonly insertRequested$ = new Subject<string>()

  private ws: WebSocket | null = null
  private _startWhenReady = false

  /** Open the drawer and connect the socket so the device list is available. */
  open(): void {
    this.panelOpen.set(true)
    if (this.ws === null) {
      this._connect()
    }
  }

  /** Close the drawer, stopping any active capture. */
  close(): void {
    this.stop()
    this.panelOpen.set(false)
  }

  /** Begin capturing the selected source. Connects first if the socket was closed after a stop. */
  start(): void {
    const deviceId = this.selectedDeviceId()
    if (deviceId === null) {
      return
    }
    this.errorMessage.set('')
    this.state.set('connecting')
    if (this.ws !== null && this.ws.readyState === WebSocket.OPEN) {
      this._sendStart()
    } else {
      this._startWhenReady = true
      this._connect()
    }
  }

  /** Stop capturing; the backend closes the socket in response. */
  stop(): void {
    if (this.ws !== null && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: 'stop' }))
    }
    this._startWhenReady = false
    this.provisionalText.set('')
    if (this.state() !== 'error') {
      this.state.set('idle')
    }
  }

  clearTranscript(): void {
    this.committedText.set('')
    this.provisionalText.set('')
  }

  insertIntoMessage(): void {
    const text = (this.committedText() + ' ' + this.provisionalText()).trim()
    if (text.length > 0) {
      this.insertRequested$.next(text)
    }
  }

  private _connect(): void {
    this.ws = new WebSocket(wsUrl())

    this.ws.onmessage = (ev) => {
      const event: ServerEvent = JSON.parse(ev.data)
      this._handle(event)
    }
    this.ws.onerror = () => {
      this.state.set('error')
      this.errorMessage.set('WebSocket error')
    }
    this.ws.onclose = () => {
      this.ws = null
      if (this.state() !== 'error') {
        this.state.set('idle')
      }
    }
  }

  private _sendStart(): void {
    this.ws!.send(
      JSON.stringify({
        action: 'start',
        device_id: this.selectedDeviceId(),
        kind: this.devices().find((d) => d.id === this.selectedDeviceId())?.kind ?? 'output',
        language: this.language(),
        translate: this.translate(),
      }),
    )
  }

  private _handle(event: ServerEvent): void {
    if (event.type === 'devices') {
      this.devices.set(event.devices)
      if (this.selectedDeviceId() === null && event.devices.length > 0) {
        this.selectedDeviceId.set(event.devices[0].id)
      }
      if (this._startWhenReady) {
        this._startWhenReady = false
        this._sendStart()
      }
    } else if (event.type === 'status') {
      if (this.state() !== 'error') {
        this.state.set(event.state)
      }
    } else if (event.type === 'committed') {
      if (event.text.length > 0) {
        this.committedText.update((c) => (c + ' ' + event.text).trim())
      }
    } else if (event.type === 'provisional') {
      this.provisionalText.set(event.text)
    } else if (event.type === 'error') {
      this.state.set('error')
      this.errorMessage.set(event.message)
    }
  }
}
