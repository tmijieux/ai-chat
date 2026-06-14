import { Component, input } from '@angular/core'

@Component({
  selector: 'app-tool-evaluating',
  standalone: true,
  templateUrl: './tool-evaluating.component.html',
  styleUrls: ['./tool-evaluating.component.scss'],
  host: { class: 'flex justify-center w-full' },
})
export class ToolEvaluatingComponent {
  readonly toolName = input.required<string>()
}
