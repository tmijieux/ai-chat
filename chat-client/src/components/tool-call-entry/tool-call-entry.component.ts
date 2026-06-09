import { Component, computed, input } from '@angular/core'
import { DiffLine, ToolCallEntry } from '../../types/message-types'
import { DiffBlockComponent } from '../diff-block/diff-block.component'

@Component({
  selector: 'app-tool-call-entry',
  standalone: true,
  imports: [DiffBlockComponent],
  templateUrl: './tool-call-entry.component.html',
  styleUrls: ['./tool-call-entry.component.scss'],
})
export class ToolCallEntryComponent {
  readonly tc = input.required<ToolCallEntry>()

  readonly diffLines = computed<DiffLine[] | null>(() => {
    const entry = this.tc()
    if (entry.name !== 'edit_file') {
      return null
    }
    const oldString = entry.args['old_string']
    const newString = entry.args['new_string']
    if (typeof oldString !== 'string' || typeof newString !== 'string') {
      return null
    }
    return this._lcs(oldString, newString)
  })

  private _lcs(oldStr: string, newStr: string): DiffLine[] {
    const oldLines = oldStr.split('\n')
    const newLines = newStr.split('\n')
    const m = oldLines.length
    const n = newLines.length
    const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0))
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        dp[i][j] = oldLines[i - 1] === newLines[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
    type Op = ['eq' | 'rm' | 'add', number, number]
    const ops: Op[] = []
    let i = m, j = n
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && oldLines[i - 1] === newLines[j - 1]) {
        ops.unshift(['eq', i - 1, j - 1]); i--; j--
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        ops.unshift(['add', -1, j - 1]); j--
      } else {
        ops.unshift(['rm', i - 1, -1]); i--
      }
    }
    const result: DiffLine[] = []
    let oldLine = 1
    for (const [kind, oldIdx, newIdx] of ops) {
      if (kind === 'eq') {
        oldLine++
      } else if (kind === 'rm') {
        result.push({ type: 'removed', line: oldLine++, text: oldLines[oldIdx] })
      } else {
        result.push({ type: 'added', line: null, text: newLines[newIdx] })
      }
    }
    return result
  }

  formatArgs(args: Record<string, unknown>): string {
    return Object.entries(args)
      .map(([key, value]) => {
        const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
        const truncated = text.length > 300 ? text.slice(0, 300) + '…' : text
        return `${key}: ${truncated}`
      })
      .join('\n\n')
  }
}
