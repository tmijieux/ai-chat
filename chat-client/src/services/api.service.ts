import { inject, Injectable } from '@angular/core'
import {
  AgentToolMeta,
  AgentToolsResponse,
  Conversation,
  ConversationSettings,
  Message,
  MessageForQuery,
  SystemPromptTemplate,
} from '../types/message-types'
import { HttpClient } from '@angular/common/http'

export const BASE_URL = '/api'

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private http = inject(HttpClient)

  get_conversations() {
    return this.http.get<Conversation[]>(`${BASE_URL}/conversations`)
  }

  post_conversation(title: string) {
    return this.http.post<Conversation>(`${BASE_URL}/conversations`, { title })
  }

  put_conversation_settings(
    conversationId: string,
    conversationSettings: Partial<ConversationSettings>,
  ) {
    return this.http.put(
      `${BASE_URL}/conversations/${conversationId}/settings`,
      conversationSettings,
    )
  }

  post_message(conversationId: string, message: Message) {
    return this.http.post<{ id: string; parent_id: string | null }>(
      `${BASE_URL}/messages`,
      message,
      {
        params: { conversationId: conversationId },
      },
    )
  }

  delete_conversation(conversationId: string) {
    return this.http.delete(`${BASE_URL}/conversations/${conversationId}`)
  }

  get_conversation_messages(conversationId: string) {
    return this.http.get<Message[]>(`${BASE_URL}/conversations/${conversationId}/messages`)
  }

  generate_chat_response(messagesArray: MessageForQuery[]) {
    return this.http.post(
      `${BASE_URL}/chat`,
      { messages: messagesArray },
      {
        observe: 'events',
        responseType: 'text',
        reportProgress: true,
      },
    )
  }

  compute_conversation_token_count(id: string) {
    return this.http.post<{ token_count: number; message_id: string }>(
      `${BASE_URL}/conversations/${id}/count-tokens`,
      {},
    )
  }

  get_system_prompts() {
    return this.http.get<SystemPromptTemplate[]>(`${BASE_URL}/system-prompts`)
  }

  create_system_prompt(body: {
    name: string
    category: string
    content: string
    is_default: boolean
  }) {
    return this.http.post<SystemPromptTemplate>(`${BASE_URL}/system-prompts`, body)
  }

  update_system_prompt(
    id: string,
    body: Partial<{ name: string; category: string; content: string; is_default: boolean }>,
  ) {
    return this.http.put<SystemPromptTemplate>(`${BASE_URL}/system-prompts/${id}`, body)
  }

  delete_system_prompt(id: string) {
    return this.http.delete(`${BASE_URL}/system-prompts/${id}`)
  }

  get_agent_tools() {
    return this.http.get<AgentToolsResponse>(`${BASE_URL}/agent/tools`)
  }

  patch_message_token_count(msgId: string, tokenCount: number, tokenDelta?: number | null) {
    return this.http.patch(`${BASE_URL}/messages/${msgId}/token-count`, {
      token_count: tokenCount,
      ...(tokenDelta != null ? { token_delta: tokenDelta } : {}),
    })
  }

  branch_message(msgId: string, content: string) {
    return this.http.put<{ id: string; parent_id: string | null }>(
      `${BASE_URL}/messages/${msgId}/branch`,
      { content },
    )
  }

  delete_message(convId: string, msgId: string, subtree: boolean) {
    return this.http.delete<{ deleted: string[] }>(
      `${BASE_URL}/conversations/${convId}/messages/${msgId}`,
      { params: { subtree: String(subtree) } },
    )
  }

  set_active_branch(convId: string, messageId: string) {
    return this.http.put<{ active_message_id: string; messages: Message[] }>(
      `${BASE_URL}/conversations/${convId}/active-branch`,
      { message_id: messageId },
    )
  }

  browse_directory(path?: string | null) {
    return this.http.get<{ path: string; parent: string | null; entries: { name: string; path: string }[] }>(
      `${BASE_URL}/utils/browse-directory`,
      path ? { params: { path } } : undefined,
    )
  }
}
