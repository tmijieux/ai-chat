import { Location } from '@angular/common'
import { Component, inject, signal } from '@angular/core'
import { PromptsListComponent } from './prompts-list/prompts-list.component'
import { AgentsListComponent } from './agents-list/agents-list.component'

type Tab = 'prompts' | 'agents'

@Component({
  selector: 'app-settings',
  imports: [PromptsListComponent, AgentsListComponent],
  templateUrl: './settings.component.html',
  host: {
    class: 'h-full w-full flex flex-col bg-panel-dark',
  },
})
export class SettingsComponent {
  readonly location = inject(Location)
  readonly activeTab = signal<Tab>('prompts')

  setTab(tab: Tab) {
    this.activeTab.set(tab)
  }
}
