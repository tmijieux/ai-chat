import { Component, inject, signal, OnInit } from '@angular/core'
import { ApiService } from '../../../services/api.service'
import { SystemPromptTemplate, SystemPromptCategory } from '../../../types/message-types'

type PromptForm = {
  name: string
  category: SystemPromptCategory
  content: string
  is_default: boolean
}

const CATEGORIES: SystemPromptCategory[] = [
  'general',
  'code',
  'summarization',
  'context_compaction',
  'state_storage',
]

const BLANK_FORM: PromptForm = { name: '', category: 'general', content: '', is_default: false }

@Component({
  selector: 'app-prompts-list',
  imports: [],
  templateUrl: './prompts-list.component.html',
  host: { class: 'flex flex-col gap-4' },
})
export class PromptsListComponent implements OnInit {
  private api = inject(ApiService)

  readonly categories = CATEGORIES
  readonly prompts = signal<SystemPromptTemplate[]>([])
  readonly editingId = signal<string | null>(null)
  readonly showCreateForm = signal(false)
  readonly deleteConfirmId = signal<string | null>(null)
  readonly saving = signal(false)
  readonly createForm = signal<PromptForm>({ ...BLANK_FORM })
  readonly editForm = signal<PromptForm>({ ...BLANK_FORM })

  ngOnInit() {
    this.load()
  }

  load() {
    this.api.get_system_prompts().subscribe((p) => this.prompts.set(p))
  }

  startCreate() {
    this.createForm.set({ ...BLANK_FORM })
    this.showCreateForm.set(true)
    this.editingId.set(null)
  }

  cancelCreate() {
    this.showCreateForm.set(false)
  }

  saveCreate() {
    const f = this.createForm()
    if (f.name.trim() === '' || f.content.trim() === '') { return }
    this.saving.set(true)
    this.api.create_system_prompt(f).subscribe(() => {
      this.saving.set(false)
      this.showCreateForm.set(false)
      this.load()
    })
  }

  startEdit(p: SystemPromptTemplate) {
    this.editForm.set({ name: p.name, category: p.category, content: p.content, is_default: p.is_default })
    this.editingId.set(p.id)
    this.showCreateForm.set(false)
  }

  cancelEdit() {
    this.editingId.set(null)
  }

  saveEdit(id: string) {
    const f = this.editForm()
    if (f.name.trim() === '' || f.content.trim() === '') { return }
    this.saving.set(true)
    this.api.update_system_prompt(id, f).subscribe(() => {
      this.saving.set(false)
      this.editingId.set(null)
      this.load()
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
      this.load()
    })
  }

  updateCreateForm(partial: Partial<PromptForm>) {
    this.createForm.update((f) => ({ ...f, ...partial }))
  }

  updateEditForm(partial: Partial<PromptForm>) {
    this.editForm.update((f) => ({ ...f, ...partial }))
  }
}
