import { Component, OnInit, inject, signal } from '@angular/core'
import { FormsModule } from '@angular/forms'
import { RouterLink } from '@angular/router'
import { firstValueFrom } from 'rxjs'
import { ApiService, BASE_URL } from '../../services/api.service'
import {
  AgentToolMeta,
  SystemPromptTemplate,
  TokenVisualizerHistoryMessage,
  TokenVisualizerMessage,
  TokenVisualizerPiece,
  TokenVisualizerStreamEvent,
} from '../../types/message-types'

@Component({
  selector: 'app-token-visualizer',
  imports: [FormsModule, RouterLink],
  templateUrl: './token-visualizer.component.html',
  host: {
    class: 'h-full w-full flex flex-col bg-panel-dark',
  },
})
export class TokenVisualizerComponent implements OnInit {
  private readonly apiService = inject(ApiService)

  readonly messages = signal<TokenVisualizerMessage[]>([])
  readonly inputText = signal('')
  readonly loading = signal(false)

  readonly systemPrompts = signal<SystemPromptTemplate[]>([])
  readonly selectedSystemPromptId = signal<string>('')

  readonly agentTools = signal<AgentToolMeta[]>([])
  readonly selectedToolNames = signal<Set<string>>(new Set())

  readonly simulateToolName = signal<string>('')
  readonly simulateToolArguments = signal('{}')
  readonly simulateToolResult = signal('')
  readonly simulateToolError = signal<string | null>(null)

  ngOnInit() {
    this.apiService.get_system_prompts().subscribe((prompts) => this.systemPrompts.set(prompts))
    this.apiService.get_agent_tools().subscribe((response) => this.agentTools.set(response.tools))
  }

  private selectedSystemPromptContent(): string | null {
    const prompt = this.systemPrompts().find((p) => p.id === this.selectedSystemPromptId())
    return prompt !== undefined ? prompt.content : null
  }

  toggleTool(name: string) {
    this.selectedToolNames.update((current) => {
      const updated = new Set(current)
      if (updated.has(name)) {
        updated.delete(name)
      } else {
        updated.add(name)
      }
      return updated
    })
  }

  private historyForRequest(): TokenVisualizerHistoryMessage[] {
    return this.messages()
      .filter((m): m is TokenVisualizerMessage & { historyMessage: TokenVisualizerHistoryMessage } => m.historyMessage !== null)
      .map((m) => m.historyMessage)
  }

  async send() {
    const message = this.inputText().trim()
    if (message === '' || this.loading()) {
      return
    }

    const isFirstTurn = this.messages().length === 0
    const history = this.historyForRequest()
    const systemPromptContent = this.selectedSystemPromptContent()
    const toolNames = [...this.selectedToolNames()]

    this.loading.set(true)
    this.inputText.set('')

    this.messages.update((current) => {
      const additions: TokenVisualizerMessage[] = []
      if (isFirstTurn && (systemPromptContent !== null || toolNames.length > 0)) {
        additions.push({ role: 'system', displayLabel: 'system', historyMessage: null, content: systemPromptContent ?? '', tokens: [] })
      }
      additions.push({
        role: 'user',
        displayLabel: 'user',
        historyMessage: { role: 'user', content: message },
        content: message,
        tokens: [],
      })
      additions.push({
        role: 'assistant',
        displayLabel: 'assistant',
        historyMessage: { role: 'assistant', content: '' },
        content: '',
        tokens: [],
      })
      return [...current, ...additions]
    })

    try {
      const response = await fetch(`${BASE_URL}/token-visualizer/turn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history, message, system_prompt: systemPromptContent, tool_names: toolNames }),
      })
      if (response.body === null) {
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true })

        let newlineIndex: number
        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, newlineIndex)
          buffer = buffer.slice(newlineIndex + 1)
          if (line.trim() === '') {
            continue
          }
          this.handleEvent(JSON.parse(line) as TokenVisualizerStreamEvent)
        }
      }
    } finally {
      this.loading.set(false)
    }
  }

  private handleEvent(event: TokenVisualizerStreamEvent) {
    if (event.type === 'system_tokens') {
      this.messages.update((current) => {
        const updated = [...current]
        const systemMessageIndex = updated.length - 3
        updated[systemMessageIndex] = { ...updated[systemMessageIndex], tokens: event.tokens }
        return updated
      })
    } else if (event.type === 'user_tokens') {
      this.messages.update((current) => {
        const updated = [...current]
        const userMessageIndex = updated.length - 2
        updated[userMessageIndex] = { ...updated[userMessageIndex], tokens: event.tokens }
        return updated
      })
    } else if (event.type === 'assistant_preamble_tokens') {
      this.messages.update((current) => {
        const updated = [...current]
        const assistantMessageIndex = updated.length - 1
        const assistantMessage = updated[assistantMessageIndex]
        updated[assistantMessageIndex] = {
          ...assistantMessage,
          tokens: [...assistantMessage.tokens, ...event.tokens],
        }
        return updated
      })
    } else if (event.type === 'assistant_token') {
      this.messages.update((current) => {
        const updated = [...current]
        const assistantMessageIndex = updated.length - 1
        const assistantMessage = updated[assistantMessageIndex]
        const newContent = assistantMessage.content + event.piece
        updated[assistantMessageIndex] = {
          ...assistantMessage,
          content: newContent,
          historyMessage: { role: 'assistant', content: newContent },
          tokens: [...assistantMessage.tokens, { id: event.id, piece: event.piece, special: event.special }],
        }
        return updated
      })
    }
  }

  async insertSimulatedToolCall() {
    const toolName = this.simulateToolName()
    if (toolName === '' || this.loading()) {
      return
    }

    let parsedArguments: unknown
    try {
      parsedArguments = JSON.parse(this.simulateToolArguments())
    } catch {
      this.simulateToolError.set('Arguments must be valid JSON')
      return
    }
    this.simulateToolError.set(null)

    const isFirstTurn = this.messages().length === 0
    const baseHistory = this.historyForRequest()
    const systemPromptContent = this.selectedSystemPromptContent()
    const toolNames = [...this.selectedToolNames()]

    const toolCallId = `call_${Date.now()}`
    const assistantToolCallMessage: TokenVisualizerHistoryMessage = {
      role: 'assistant',
      content: null,
      tool_calls: [{ id: toolCallId, type: 'function', function: { name: toolName, arguments: JSON.stringify(parsedArguments) } }],
    }
    const toolResultMessage: TokenVisualizerHistoryMessage = {
      role: 'tool',
      tool_call_id: toolCallId,
      name: toolName,
      content: this.simulateToolResult(),
    }

    this.loading.set(true)
    try {
      const toolCallResponse = await firstValueFrom(
        this.apiService.post_token_visualizer_insert_messages(baseHistory, [assistantToolCallMessage], systemPromptContent, toolNames),
      )

      this.messages.update((current) => {
        const additions: TokenVisualizerMessage[] = []
        if (isFirstTurn && (systemPromptContent !== null || toolNames.length > 0)) {
          additions.push({
            role: 'system',
            displayLabel: 'system',
            historyMessage: null,
            content: systemPromptContent ?? '',
            tokens: toolCallResponse.system_tokens ?? [],
          })
        }
        additions.push({
          role: 'assistant',
          displayLabel: `assistant (tool_call: ${toolName})`,
          historyMessage: assistantToolCallMessage,
          content: '',
          tokens: toolCallResponse.tokens,
        })
        return [...current, ...additions]
      })

      const toolResultResponse = await firstValueFrom(
        this.apiService.post_token_visualizer_insert_messages(
          [...baseHistory, assistantToolCallMessage],
          [toolResultMessage],
          systemPromptContent,
          toolNames,
        ),
      )

      this.messages.update((current) => [
        ...current,
        {
          role: 'tool',
          displayLabel: `tool: ${toolName}`,
          historyMessage: toolResultMessage,
          content: this.simulateToolResult(),
          tokens: toolResultResponse.tokens,
        },
      ])

      this.simulateToolArguments.set('{}')
      this.simulateToolResult.set('')
    } finally {
      this.loading.set(false)
    }
  }

  onInputKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      this.send()
    }
  }

  tokenColorClass(index: number): string {
    return `token-color-${index % 8}`
  }

  tokenTooltipText(token: TokenVisualizerPiece): string {
    const lines = [`id: ${token.id}`]
    if (token.special) {
      lines.push('special control token')
    }
    return lines.join('\n')
  }

  tokenDisplayText(piece: string): string {
    return piece.replace(/\n/g, '⏎')
  }

  newlineBreaks(piece: string): number[] {
    const count = (piece.match(/\n/g) ?? []).length
    return Array.from({ length: count }, (_, i) => i)
  }
}
