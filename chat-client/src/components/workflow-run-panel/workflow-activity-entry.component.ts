import { Component, input } from '@angular/core'
import { CommonModule } from '@angular/common'
import { WorkflowActivityEntry } from '../../types/message-types'

/** One piece of a stage's transcript (thinking/content/tool_call/tool_result/error) — shared by
 * the live detail pane and the persisted-run drill view (ADR-0011) so both render it identically. */
@Component({
  selector: 'app-workflow-activity-entry',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './workflow-activity-entry.component.html',
})
export class WorkflowActivityEntryComponent {
  readonly entry = input.required<WorkflowActivityEntry>()
}
