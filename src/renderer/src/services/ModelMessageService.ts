import { Model } from '@renderer/types'
import { ChatCompletionMessageParam } from 'openai/resources'

export function processReqMessages(
  model: Model,
  reqMessages: ChatCompletionMessageParam[]
): ChatCompletionMessageParam[] {
  if (!needStrictlyInterleaveUserAndAssistantMessages(model)) {
    return reqMessages
  }

  return interleaveUserAndAssistantMessages(reqMessages)
}

function needStrictlyInterleaveUserAndAssistantMessages(model: Model) {
  return model.id === 'deepseek-reasoner'
}

/**
 * 对消息数组进行处理，确保用户和助手的消息严格交替出现。
 * 如果相邻消息的角色相同，则在它们之间插入一个空消息，其角色与当前消息角色相反。
 *
 * @param messages - 包含 ChatCompletionMessageParam 类型消息的数组。
 * @returns 处理后的消息数组，保证用户和助手消息严格交替。
 */
function interleaveUserAndAssistantMessages(messages: ChatCompletionMessageParam[]): ChatCompletionMessageParam[] {
  // 检查消息数组是否为空或未定义，如果是则直接返回空数组
  if (!messages || messages.length === 0) {
    return []
  }

  // 初始化一个空数组，用于存储处理后的消息
  const processedMessages: ChatCompletionMessageParam[] = []

  // 遍历消息数组
  for (let i = 0; i < messages.length; i++) {
    // 复制当前消息，避免修改原始数组中的对象
    const currentMessage = { ...messages[i] }

    // 检查当前消息是否不是第一条消息，并且当前消息的角色与前一条消息的角色相同
    if (i > 0 && currentMessage.role === messages[i - 1].role) {
      // 在相邻且角色相同的消息之间插入一个空消息，其角色与当前消息角色相反
      const emptyMessageRole = currentMessage.role === 'user' ? 'assistant' : 'user'
      processedMessages.push({
        role: emptyMessageRole,
        content: ''
      })
    }

    // 将当前消息添加到处理后的消息数组中
    processedMessages.push(currentMessage)
  }

  // 返回处理后的消息数组
  return processedMessages
}
