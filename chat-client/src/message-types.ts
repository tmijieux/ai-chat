export type Role = 'user' | 'assistant' | 'system'

export type Message = {
  role: Role
  content: string
  thinking?: string
}

export type ApiDone =
  | { done: false }
  | {
      done: true
      done_reason: string
      total_duration: number
      load_duration: number
      prompt_eval_count: number
      prompt_eval_duration: number
      eval_count: number
      eval_duration: number
    }

export type ApiResponse = {
  model: string
  created_at: string
  message: Message
} & ApiDone

export type ConversationHistory = Message[]

export type Conversation = {
  id: string
  title: string
  history?: ConversationHistory
}
