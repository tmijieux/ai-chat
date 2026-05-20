import { Location } from '@angular/common'
import { Component, inject, signal, computed, OnInit } from '@angular/core'
import { FormsModule } from '@angular/forms'
import { ApiService } from '../../services/api.service'
import { SystemPromptTemplate, SystemPromptCategory } from '../../types/message-types'

type PromptForm = { name: string; category: SystemPromptCategory; content: string; is_default: boolean }

const CATEGORIES: SystemPromptCategory[] = [
  'general',
  'code',
  'summarization',
  'context_compaction',
  'state_storage',
]

@Component({
  selector: 'app-settings',
  imports: [FormsModule],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.scss',
  host: {
    class: 'h-full w-full flex flex-col bg-panel-dark',
  },
})
export class SettingsComponent implements OnInit {
  location = inject(Location)
  private api = inject(ApiService)

  readonly categories = CATEGORIES

  prompts = signal<SystemPromptTemplate[]>([])
  editingId = signal<string | null>(null)
  showCreateForm = signal(false)
  deleteConfirmId = signal<string | null>(null)
  saving = signal(false)

  createForm = signal<PromptForm>({ name: '', category: 'general', content: '', is_default: false })
  editForm = signal<PromptForm>({ name: '', category: 'general', content: '', is_default: false })

  ngOnInit() {
    this.loadPrompts()
  }

  loadPrompts() {
    this.api.get_system_prompts().subscribe(p => this.prompts.set(p))
  }

  startCreate() {
    this.createForm.set({ name: '', category: 'general', content: '', is_default: false })
    this.showCreateForm.set(true)
    this.editingId.set(null)
  }

  cancelCreate() {
    this.showCreateForm.set(false)
  }

  saveCreate() {
    const f = this.createForm()
    if (!f.name.trim() || !f.content.trim()) return
    this.saving.set(true)
    this.api.create_system_prompt(f).subscribe(() => {
      this.saving.set(false)
      this.showCreateForm.set(false)
      this.loadPrompts()
    })
  }

  startEdit(p: SystemPromptTemplate) {
    this.editForm.set({ name: p.name, category: p.category, content: p.content, is_default: p.is_default === 1 })
    this.editingId.set(p.id)
    this.showCreateForm.set(false)
  }

  cancelEdit() {
    this.editingId.set(null)
  }

  saveEdit(id: string) {
    const f = this.editForm()
    if (!f.name.trim() || !f.content.trim()) return
    this.saving.set(true)
    this.api.update_system_prompt(id, f).subscribe(() => {
      this.saving.set(false)
      this.editingId.set(null)
      this.loadPrompts()
    })
  }

  confirmDelete(id: string) {
    this.deleteConfirmId.set(id)
  }

  cancelDelete() {
    this.deleteConfirmId.set(null)
  }

  doDelete(id: string) {
    this.api.delete_system_prompt(id).subscribe(() => {
      this.deleteConfirmId.set(null)
      this.loadPrompts()
    })
  }

  updateCreateForm(partial: Partial<PromptForm>) {
    this.createForm.update(f => ({ ...f, ...partial }))
  }

  updateEditForm(partial: Partial<PromptForm>) {
    this.editForm.update(f => ({ ...f, ...partial }))
  }
}
