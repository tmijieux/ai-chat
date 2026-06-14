import { Component, Input, Output, EventEmitter } from '@angular/core'

export type AgentForm = {
  name: string
  description: string
  system_prompt: string
  tools: string[]
  finish_tool: string
  max_iterations: number | null
  inject_turn_reminders: boolean
}

@Component({
  selector: 'app-agent-form',
  imports: [],
  templateUrl: './agent-form.component.html',
})
export class AgentFormComponent {
  @Input() form!: AgentForm
  @Input() availableTools: string[] = []
  @Input() availableFinishTools: string[] = []
  @Output() formChange = new EventEmitter<Partial<AgentForm>>()
  @Output() toolToggle = new EventEmitter<string>()

  update(partial: Partial<AgentForm>) {
    this.formChange.emit(partial)
  }

  toggle(toolName: string) {
    this.toolToggle.emit(toolName)
  }

  isToolSelected(toolName: string): boolean {
    return this.form.tools.includes(toolName)
  }
}
