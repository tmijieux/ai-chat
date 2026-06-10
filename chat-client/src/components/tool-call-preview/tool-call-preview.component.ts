import { Component, effect, ElementRef, signal, untracked, ViewChild } from '@angular/core'
import { input } from '@angular/core'

@Component({
  standalone: true,
  selector: 'app-tool-call-preview',
  templateUrl: './tool-call-preview.component.html',
})
export class ToolCallPreviewComponent {
  readonly toolName = input<string>('')
  readonly args = input<string>('')
  readonly streaming = input(false)

  readonly collapsed = signal(false)

  @ViewChild('argsEl') private _argsEl!: ElementRef<HTMLElement>

  constructor() {
    effect(() => {
      const _args = this.args()
      untracked(() => {
        if (!this.collapsed()) {
          queueMicrotask(() => {
            const el = this._argsEl?.nativeElement
            if (el) {
              el.scrollTop = el.scrollHeight
            }
          })
        }
      })
    })
  }

  toggleCollapse(): void {
    this.collapsed.update((v) => !v)
  }
}
