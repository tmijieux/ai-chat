import { Component, input, linkedSignal } from '@angular/core'

@Component({
  selector: 'app-collapsible-bubble',
  standalone: true,
  template: `
    <div [class]="flat() ? 'w-full ' + wrapperClass() : 'rounded-xl overflow-hidden shadow-sm w-full ' + wrapperClass()">
      <div class="collapsible-header" (click)="opened.set(!opened())">
        <span class="collapsible-label"><ng-content select="[header]" /></span>
        @if (streaming()) {
          <span class="streaming-dot"></span>
        }
        <span class="collapsible-chevron">{{ opened() ? '▾' : '▸' }}</span>
      </div>
      @if (opened()) {
        <ng-content />
      }
    </div>
  `,
  styleUrls: ['./collapsible-bubble.component.scss'],
})
export class CollapsibleBubbleComponent {
  readonly streaming = input(false)
  readonly wrapperClass = input('bg-thinking-bubble')
  readonly flat = input(false)
  readonly initiallyOpen = input(false)

  // linkedSignal, not a plain signal seeded once: initiallyOpen can depend on @for-recycled
  // component state (e.g. msg().content only resolving after creation), so this must track it
  // reactively rather than read it once at construction. Still freely writable by the click
  // handler afterward, same as a plain signal.
  readonly opened = linkedSignal(() => this.initiallyOpen())
}
