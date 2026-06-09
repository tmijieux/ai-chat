import { Component, input, signal } from '@angular/core'
import { CommonModule } from '@angular/common'
import { CollapsibleBubbleComponent } from '../collapsible-bubble/collapsible-bubble.component'
import { DisplayMessage, TokenMeta } from '../../types/message-types'

export type ToolResultMessage = Extract<DisplayMessage, { kind: 'tool_result' }> & {
  token_meta?: TokenMeta
}

@Component({
  selector: 'app-tool-result',
  standalone: true,
  imports: [CommonModule, CollapsibleBubbleComponent],
  templateUrl: './tool-result.component.html',
  styleUrls: ['./tool-result.component.scss'],
})
export class ToolResultComponent {
  readonly CTX_LIMIT = 2 ** 15
  readonly msg = input.required<ToolResultMessage>()

  private readonly _tab = signal<'output' | 'summary'>('summary')

  get tab(): 'output' | 'summary' {
    return this._tab()
  }

  setTab(tab: 'output' | 'summary'): void {
    this._tab.set(tab)
  }

  grepHeaderSuffix(content: string, compressed: string | null | undefined): string {
    if (compressed) {
      return ''
    }
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'grep_files' || r.total == null) {
        return ''
      }
      const n: number = r.total
      return ` → ${n} ${n === 1 ? 'match' : 'matches'}`
    } catch {
      return ''
    }
  }

  parseGrepResult(content: string): {
    files: { name: string; lines: { no: number; text: string; match: boolean }[] }[]
    pattern: string
    total: number
    truncated: boolean
  } | null {
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'grep_files' || !Array.isArray(r.matches)) {
        return null
      }
      const map = new Map<string, { no: number; text: string; match: boolean }[]>()
      for (const m of r.matches) {
        if (!map.has(m.file)) {
          map.set(m.file, [])
        }
        map.get(m.file)!.push({ no: m.line, text: m.content, match: !!m.match })
      }
      return {
        files: [...map.entries()].map(([name, lines]) => ({ name, lines })),
        pattern: r.pattern ?? '',
        total: r.total ?? 0,
        truncated: !!r.truncated,
      }
    } catch {
      return null
    }
  }

  formatSimpleToolResult(content: string): { icon: string; text: string } | null {
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'write_file' && r.tool !== 'edit_file') {
        return null
      }
      if (r.status === 'rejected') {
        const reason = r.reason ? ` — ${r.reason}` : ''
        return { icon: '✗', text: `${r.tool}: rejected${reason}` }
      }
      const icon = r.status === 'success' ? '✓' : '✗'
      const msg = r.error?.message ? ` — ${r.error.message}` : r.message ? ` — ${r.message}` : ''
      return { icon, text: `${r.tool}: ${r.path ?? ''}${msg}` }
    } catch {
      return null
    }
  }

  formatShellResult(content: string): { icon: string; output: string } | null {
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'run_shell') {
        return null
      }
      if (r.status === 'rejected') {
        return { icon: '✗ rejected', output: r.reason ?? '' }
      }
      const ok = r.status === 'success'
      const icon = ok ? '✓ exit 0' : '✗ exit 1'
      const output = ok ? (r.output ?? '') : (r.error?.message ?? '')
      return { icon, output }
    } catch {
      return null
    }
  }

  formatReadFileResult(content: string): { path: string; fileContent: string } | null {
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'read_file') {
        return null
      }
      if (r.status !== 'success') {
        return null
      }
      return { path: r.path ?? '', fileContent: r.file_content ?? '' }
    } catch {
      return null
    }
  }

  formatListDirResult(content: string): { path: string; entries: string[] } | null {
    try {
      const r = JSON.parse(content)
      if (r.tool !== 'list_directory') {
        return null
      }
      if (r.status !== 'success') {
        return null
      }
      const entries = ((r.content as string) ?? '').split('\n').filter((line) => line !== '')
      return { path: r.path ?? '', entries }
    } catch {
      return null
    }
  }
}
