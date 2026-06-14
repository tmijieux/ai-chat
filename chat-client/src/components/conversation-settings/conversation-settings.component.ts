import { Component, computed, inject, input, output, OnInit, signal } from '@angular/core'
import { FormsModule } from '@angular/forms'
import { ApiService } from '../../services/api.service'
import { ChatService } from '../../services/chat.service'
import {
  ConversationMode,
  ConversationSettings,
  SystemPromptTemplate,
} from '../../types/message-types'
import { DirectoryPickerComponent } from '../directory-picker/directory-picker.component'

@Component({
  selector: 'app-conversation-settings',
  imports: [FormsModule, DirectoryPickerComponent],
  templateUrl: './conversation-settings.component.html',
})
export class ConversationSettingsComponent implements OnInit {
  readonly conversationId = input<string | undefined>(undefined)
  readonly settings = input.required<ConversationSettings>()
  readonly closed = output<void>()
  readonly settingsChanged = output<ConversationSettings>()

  private api = inject(ApiService)
  private chatSvc = inject(ChatService)

  readonly prompts = signal<SystemPromptTemplate[]>([])
  readonly tools = this.chatSvc.allTools
  readonly enabledToolsTokenCount = computed(() => {
    const enabledNames = new Set(this.settings().active_tool_names)
    const enabledTools = this.chatSvc.allTools().filter((t) => enabledNames.has(t.name))
    const N = enabledTools.length
    if (N === 0) {
      return 0
    }
    const toolTokens = enabledTools.reduce((sum, t) => sum + t.token_count, 0)
    const stackingOverhead = this.chatSvc.stackingOverheadPerAdditionalTool() * (N - 1)
    return this.chatSvc.toolFrameworkOverhead() + toolTokens + stackingOverhead
  })
  readonly pickerOpen = signal(false)

  ngOnInit() {
    this.api.get_system_prompts().subscribe((p) => this.prompts.set(p))
  }

  isToolEnabled(toolName: string): boolean {
    return this.settings().active_tool_names.includes(toolName)
  }

  toggleTool(toolName: string, enabled: boolean) {
    const current = this.settings().active_tool_names
    const updated = enabled ? [...current, toolName] : current.filter((n) => n !== toolName)
    this.save({ active_tool_names: updated })
  }

  allToolsEnabled(): boolean {
    const names = this.tools().map(t => t.name)
    return names.length > 0 && names.every(n => this.settings().active_tool_names.includes(n))
  }

  toggleAllTools() {
    const all = this.tools().map(t => t.name)
    const updated = this.allToolsEnabled() ? [] : all
    this.save({ active_tool_names: updated })
  }

  setPrompt(promptId: string) {
    this.save({ active_prompt_id: promptId === '' ? null : promptId })
  }

  setMode(mode: ConversationMode) {
    this.save({ mode })
  }

  setWorkingDirectory(dir: string) {
    this.save({ working_directory: dir === '' ? null : dir })
  }

  onDirectorySelected(path: string) {
    this.pickerOpen.set(false)
    this.save({ working_directory: path })
  }

  private save(partial: Partial<ConversationSettings>) {
    const updated: ConversationSettings = { ...this.settings(), ...partial }
    this.settingsChanged.emit(updated)
  }

  activePromptLabel(): string {
    const id = this.settings().active_prompt_id
    if (id === null) return 'None'
    const p = this.prompts().find((p) => p.id === id)
    return p ? p.name : 'None'
  }
}
