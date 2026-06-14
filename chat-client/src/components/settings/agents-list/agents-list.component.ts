import { Component, inject, signal, OnInit } from '@angular/core'
import { ApiService } from '../../../services/api.service'
import { AgentDefinition } from '../../../types/message-types'
import { AgentFormComponent, AgentForm } from '../agent-form/agent-form.component'

const BLANK_FORM: AgentForm = {
  name: '',
  description: '',
  system_prompt: '',
  tools: [],
  finish_tool: 'finish_task',
  max_iterations: 5,
  inject_turn_reminders: false,
}

@Component({
  selector: 'app-agents-list',
  imports: [AgentFormComponent],
  templateUrl: './agents-list.component.html',
  host: { class: 'flex flex-col gap-4' },
})
export class AgentsListComponent implements OnInit {
  private api = inject(ApiService)

  readonly agents = signal<AgentDefinition[]>([])
  readonly availableTools = signal<string[]>([])
  readonly availableFinishTools = signal<string[]>([])
  readonly editingName = signal<string | null>(null)
  readonly showCreateForm = signal(false)
  readonly deleteConfirmName = signal<string | null>(null)
  readonly saving = signal(false)
  readonly createForm = signal<AgentForm>({ ...BLANK_FORM })
  readonly editForm = signal<AgentForm>({ ...BLANK_FORM })

  ngOnInit() {
    this.load()
    this.api.get_agent_tools().subscribe((r) => {
      this.availableTools.set(r.tools.map((t) => t.name))
    })
    this.api.get_finish_tools().subscribe((names) => this.availableFinishTools.set(names))
  }

  load() {
    this.api.get_agents().subscribe((a) => this.agents.set(a))
  }

  startCreate() {
    this.createForm.set({ ...BLANK_FORM })
    this.showCreateForm.set(true)
    this.editingName.set(null)
  }

  cancelCreate() {
    this.showCreateForm.set(false)
  }

  saveCreate() {
    const f = this.createForm()
    if (f.name.trim() === '' || f.system_prompt.trim() === '') { return }
    this.saving.set(true)
    this.api.create_agent(f).subscribe(() => {
      this.saving.set(false)
      this.showCreateForm.set(false)
      this.load()
    })
  }

  startEdit(a: AgentDefinition) {
    this.editForm.set({
      name: a.name,
      description: a.description,
      system_prompt: a.system_prompt,
      tools: [...a.tools],
      finish_tool: a.finish_tool,
      max_iterations: a.max_iterations,
      inject_turn_reminders: a.inject_turn_reminders,
    })
    this.editingName.set(a.name)
    this.showCreateForm.set(false)
  }

  cancelEdit() {
    this.editingName.set(null)
  }

  saveEdit(originalName: string) {
    const f = this.editForm()
    if (f.name.trim() === '' || f.system_prompt.trim() === '') { return }
    this.saving.set(true)
    this.api.update_agent(originalName, f).subscribe(() => {
      this.saving.set(false)
      this.editingName.set(null)
      this.load()
    })
  }

  confirmDelete(name: string) {
    this.deleteConfirmName.set(name)
  }

  cancelDelete() {
    this.deleteConfirmName.set(null)
  }

  doDelete(name: string) {
    this.api.delete_agent(name).subscribe(() => {
      this.deleteConfirmName.set(null)
      this.load()
    })
  }

  toggleTool(form: AgentForm, toolName: string): AgentForm {
    const tools = form.tools.includes(toolName)
      ? form.tools.filter((t) => t !== toolName)
      : [...form.tools, toolName]
    return { ...form, tools }
  }

  updateCreateForm(partial: Partial<AgentForm>) {
    this.createForm.update((f) => ({ ...f, ...partial }))
  }

  toggleCreateTool(toolName: string) {
    this.createForm.update((f) => this.toggleTool(f, toolName))
  }

  updateEditForm(partial: Partial<AgentForm>) {
    this.editForm.update((f) => ({ ...f, ...partial }))
  }

  toggleEditTool(toolName: string) {
    this.editForm.update((f) => this.toggleTool(f, toolName))
  }
}
