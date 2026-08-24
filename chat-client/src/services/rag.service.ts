import { Injectable, inject, signal } from '@angular/core'
import { firstValueFrom } from 'rxjs'
import { ApiService } from './api.service'
import { RagIndexService } from './rag-index.service'
import { RagActivity, RagSource, RagSpace } from '../types/message-types'

/** Sentinel rejection value from _indexWithProgress signalling the run was cancelled — not an
 * Error, since it isn't a failure the caller needs a stack trace for. */
const _RAG_CANCELLED = Symbol('rag-cancelled')

// Prism language ids actually loaded by main.ts — only extensions with a loaded grammar get a
// language hint; anything else still fences safely, just without syntax highlighting.
const _LANGUAGE_BY_EXTENSION: Record<string, string> = {
  '.py': 'python',
  '.ts': 'typescript',
  '.tsx': 'typescript',
  '.js': 'javascript',
  '.jsx': 'javascript',
  '.json': 'json',
  '.sql': 'sql',
  '.html': 'markup',
  '.htm': 'markup',
  '.xml': 'markup',
  '.css': 'css',
  '.scss': 'css',
  '.c': 'c',
  '.h': 'c',
  '.cpp': 'cpp',
  '.hpp': 'cpp',
  '.sh': 'bash',
  '.bash': 'bash',
}

function _languageFromPath(path: string | null): string {
  if (path === null) {
    return ''
  }
  const dot = path.lastIndexOf('.')
  return dot === -1 ? '' : (_LANGUAGE_BY_EXTENSION[path.slice(dot).toLowerCase()] ?? '')
}

/** A backtick fence guaranteed longer than any run of backticks already inside `text`, so a chunk
 * that's itself markdown containing its own code fences can't prematurely close ours. */
function _fenceFor(text: string): string {
  const longestRun = Math.max(0, ...(text.match(/`+/g) ?? []).map((run) => run.length))
  return '`'.repeat(Math.max(3, longestRun + 1))
}

/** Renders one search-result chunk as a fenced code block (language-hinted from its source file's
 * extension when known) rather than a markdown blockquote — quoting raw source as a blockquote let
 * the chunk's own markdown-special characters (underscores, asterisks, a leading '#', or its own
 * embedded backticks) get reinterpreted by the renderer instead of shown verbatim, which is what
 * made code results look broken. A fence treats the content as opaque text.
 *
 * Exported so the rag_search agent tool's result bubble (tool-result.component.ts) can render its
 * chunks identically to the /rag-search slash command's output. */
export function renderRagResultChunk(sourceTitle: string, score: number, originPath: string | null, text: string): string {
  const fence = _fenceFor(text)
  const language = _languageFromPath(originPath)
  return `**${sourceTitle}** (score: ${score.toFixed(3)})\n${fence}${language}\n${text}\n${fence}`
}

/**
 * RAG slash-command orchestration: resolving the one-space-per-workspace convention, running
 * ingestion (with live progress via RagIndexService) or a query, and formatting the result as a
 * markdown string. Owns `activity` (drives the chat area's progress banner) so ChatService and
 * its components don't need to — ChatService only calls in and posts the returned string as a
 * chat message, mirroring how AgentService owns the agent socket/running flag independently.
 */
@Injectable({ providedIn: 'root' })
export class RagService {
  private api = inject(ApiService)
  private indexSvc = inject(RagIndexService)

  private _activity = signal<RagActivity>(null)
  /** Drives the RAG activity banner; null when no /rag-index or /rag-search is running. */
  public readonly activity = this._activity.asReadonly()

  /** Indexes `arg` (a path relative to `workspace`; empty means the whole workspace) into the
   * RAG space bound to `workspace` (get-or-created), reporting live progress via `activity`.
   * Returns a formatted markdown summary suitable for posting as a chat message — never throws,
   * failure and cancellation are both reported in the returned text. */
  async runIndex(workspace: string, arg: string): Promise<string> {
    const space = await this._getOrCreateWorkspaceSpace(workspace)
    const path = arg.trim() === '' ? '.' : arg.trim()
    this._activity.set({ kind: 'rag-index', label: `Indexing \`${path}\` into RAG space **${space.name}**…` })
    try {
      const sources = await this._indexWithProgress(space.id, workspace, path)
      const lines = sources.slice(0, 30).map((s) => `- ${s.origin_path ?? s.title} (${s.status})`)
      const more = sources.length > 30 ? `\n…and ${sources.length - 30} more.` : ''
      return `Indexed **${sources.length}** file(s) from \`${path}\` into RAG space **${space.name}**:\n\n${lines.join('\n')}${more}`
    } catch (err) {
      return err === _RAG_CANCELLED ? 'Indexing cancelled.' : `Indexing failed: ${err}`
    } finally {
      this._activity.set(null)
    }
  }

  /** Searches the RAG space bound to `workspace` (get-or-created) for `query`, reporting a brief
   * "Searching…" activity. Returns a formatted markdown summary of the ranked results. */
  async runSearch(workspace: string, query: string): Promise<string> {
    if (query.trim() === '') {
      return 'Usage: `/rag-search <query>`'
    }
    const space = await this._getOrCreateWorkspaceSpace(workspace)
    this._activity.set({ kind: 'rag-search', label: `Searching RAG space **${space.name}**…` })
    try {
      const results = await firstValueFrom(this.api.post_rag_query(space.id, { query: query.trim(), top_k: 5 }))
      if (results.length === 0) {
        return `No results in RAG space **${space.name}** — has it been indexed yet? Try \`/rag-index\` first.`
      }
      return results
        .map((r) => renderRagResultChunk(r.source_title, r.score, r.origin_path, r.text))
        .join('\n\n')
    } finally {
      this._activity.set(null)
    }
  }

  /** Cancels the currently-running /rag-index, if any. */
  cancelIndex(): void {
    this.indexSvc.cancel()
  }

  private async _getOrCreateWorkspaceSpace(workspace: string): Promise<RagSpace> {
    const spaces = await firstValueFrom(this.api.get_rag_spaces())
    const existing = spaces.find((s) => s.workspace_path === workspace)
    if (existing) {
      return existing
    }
    const name = workspace.replace(/[/\\]+$/, '').split(/[/\\]/).pop() || workspace
    return firstValueFrom(this.api.post_rag_space({ name, workspace_path: workspace }))
  }

  /** Runs one workspace-path ingestion over the progress-reporting websocket, updating
   * `activity`'s progress field as events arrive. Resolves with the final sources, or rejects
   * with `_RAG_CANCELLED` if the run was cancelled, or the server's error message otherwise. */
  private _indexWithProgress(spaceId: string, workspace: string, path: string): Promise<RagSource[]> {
    return new Promise((resolve, reject) => {
      const subscription = this.indexSvc.events$.subscribe((event) => {
        if (event.type === 'progress') {
          this._activity.update((activity) =>
            activity ? { ...activity, progress: { current: event.current, total: event.total, filename: event.filename } } : activity,
          )
        } else if (event.type === 'done') {
          subscription.unsubscribe()
          resolve(event.sources)
        } else if (event.type === 'cancelled') {
          subscription.unsubscribe()
          reject(_RAG_CANCELLED)
        } else if (event.type === 'error') {
          subscription.unsubscribe()
          reject(event.message)
        }
      })
      this.indexSvc.start(spaceId, workspace, path)
    })
  }
}
