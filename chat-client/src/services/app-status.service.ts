import { inject, Injectable, signal } from '@angular/core'
import { catchError, interval, of, switchMap, takeWhile } from 'rxjs'
import { ApiService } from './api.service'

@Injectable({ providedIn: 'root' })
export class AppStatusService {
  private api = inject(ApiService)

  readonly llmReady = signal(false)
  readonly whisperReady = signal(false)

  startPolling(): void {
    interval(2000).pipe(
      switchMap(() => this.api.get_status().pipe(catchError(() => of(null)))),
      takeWhile(() => !this.llmReady() || !this.whisperReady(), true),
    ).subscribe((status) => {
      if (!status) {
        return
      }
      this.llmReady.set(status.llm)
      this.whisperReady.set(status.whisper)
    })
  }
}
