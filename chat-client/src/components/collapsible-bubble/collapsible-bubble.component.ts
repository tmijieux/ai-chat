import { Component, input, signal } from '@angular/core'

@Component({
  selector: 'app-collapsible-bubble',
  standalone: true,
  template: `
    <div class="collapsible-wrap" [class]="wrapperClass()">
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

  readonly opened = signal(false)
}
