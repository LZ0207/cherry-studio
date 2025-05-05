import { Model } from '@renderer/types'
import type { Message } from '@renderer/types/newMessage'

import { findImageBlocks, getMainTextContent } from './messageUtils/find'

export function escapeDollarNumber(text: string) {
  let escapedText = ''

  for (let i = 0; i < text.length; i += 1) {
    let char = text[i]
    const nextChar = text[i + 1] || ' '

    if (char === '$' && nextChar >= '0' && nextChar <= '9') {
      char = '\\$'
    }

    escapedText += char
  }

  return escapedText
}

export function escapeBrackets(text: string) {
  const pattern = /(```[\s\S]*?```|`.*?`)|\\\[([\s\S]*?[^\\])\\\]|\\\((.*?)\\\)/g
  return text.replace(pattern, (match, codeBlock, squareBracket, roundBracket) => {
    if (codeBlock) {
      return codeBlock
    } else if (squareBracket) {
      return `
$$
${squareBracket}
$$
`
    } else if (roundBracket) {
      return `$${roundBracket}$`
    }
    return match
  })
}

export function extractTitle(html: string): string | null {
  // 处理标准闭合的标题标签
  const titleRegex = /<title>(.*?)<\/title>/i
  const match = html.match(titleRegex)

  if (match) {
    return match[1] ? match[1].trim() : ''
  }

  // 处理未闭合的标题标签
  const malformedTitleRegex = /<title>(.*?)($|<(?!\/title))/i
  const malformedMatch = html.match(malformedTitleRegex)

  if (malformedMatch) {
    return malformedMatch[1] ? malformedMatch[1].trim() : ''
  }

  return null
}

export function removeSvgEmptyLines(text: string): string {
  // 用正则表达式匹配 <svg> 标签内的内容
  const svgPattern = /(<svg[\s\S]*?<\/svg>)/g

  return text.replace(svgPattern, (svgMatch) => {
    // 将 SVG 内容按行分割,过滤掉空行,然后重新组合
    return svgMatch
      .split('\n')
      .filter((line) => line.trim() !== '')
      .join('\n')
  })
}

// export function withGeminiGrounding(block: MainTextMessageBlock | TranslationMessageBlock): string {
//   // TODO
//   // const citationBlock = findCitationBlockWithGrounding(block)
//   // const groundingSupports = citationBlock?.groundingMetadata?.groundingSupports

//   const content = block.content

//   // if (!groundingSupports || groundingSupports.length === 0) {
//   //   return content
//   // }

//   // groundingSupports.forEach((support) => {
//   //   const text = support?.segment?.text
//   //   const indices = support?.groundingChunkIndices

//   //   if (!text || !indices) return

//   //   const nodes = indices.reduce((acc, index) => {
//   //     acc.push(`<sup>${index + 1}</sup>`)
//   //     return acc
//   //   }, [] as string[])

//   //   content = content.replace(text, `${text} ${nodes.join(' ')}`)
//   // })

//   return content
// }

export interface ThoughtProcessor {
  canProcess: (content: string, model?: Model) => boolean
  process: (content: string) => { reasoning: string; content: string }
}

export const glmZeroPreviewProcessor: ThoughtProcessor = {
  canProcess: (content: string, model?: Model) => {
    if (!model) return false

    const modelId = model.id || ''
    const modelName = model.name || ''
    const isGLMZeroPreview =
      modelId.toLowerCase().includes('glm-zero-preview') || modelName.toLowerCase().includes('glm-zero-preview')

    return isGLMZeroPreview && content.includes('###Thinking')
  },
  process: (content: string) => {
    const parts = content.split('###')
    const thinkingMatch = parts.find((part) => part.trim().startsWith('Thinking'))
    const responseMatch = parts.find((part) => part.trim().startsWith('Response'))

    return {
      reasoning: thinkingMatch ? thinkingMatch.replace('Thinking', '').trim() : '',
      content: responseMatch ? responseMatch.replace('Response', '').trim() : ''
    }
  }
}

export const thinkTagProcessor: ThoughtProcessor = {
  canProcess: (content: string, model?: Model) => {
    if (!model) return false

    return content.startsWith('<think>') || content.includes('</think>')
  },
  process: (content: string) => {
    // 处理正常闭合的 think 标签
    const thinkPattern = /^<think>(.*?)<\/think>/s
    const matches = content.match(thinkPattern)
    if (matches) {
      return {
        reasoning: matches[1].trim(),
        content: content.replace(thinkPattern, '').trim()
      }
    }

    // 处理只有结束标签的情况
    if (content.includes('</think>') && !content.startsWith('<think>')) {
      const parts = content.split('</think>')
      return {
        reasoning: parts[0].trim(),
        content: parts.slice(1).join('</think>').trim()
      }
    }

    // 处理只有开始标签的情况
    if (content.startsWith('<think>')) {
      return {
        reasoning: content.slice(7).trim(), // 跳过 '<think>' 标签
        content: ''
      }
    }

    return {
      reasoning: '',
      content
    }
  }
}

export function withGenerateImage(message: Message): { content: string; images?: string[] } {
  const originalContent = getMainTextContent(message)
  const imagePattern = new RegExp(`!\\[[^\\]]*\\]\\((.*?)\\s*("(?:.*[^"])")?\\s*\\)`)
  const images: string[] = []
  let processedContent = originalContent

  processedContent = originalContent.replace(imagePattern, (_, url) => {
    if (url) {
      images.push(url)
    }
    return ''
  })

  processedContent = processedContent.replace(/\n\s*\n/g, '\n').trim()

  const downloadPattern = /\[[^\]]*\]\((.*?)\s*("(?:.*[^"])")?\s*\)/g
  processedContent = processedContent
    .replace(downloadPattern, '')
    .replace(/\n\s*\n/g, '\n')
    .trim()

  if (images.length > 0) {
    return { content: processedContent, images }
  }

  return { content: originalContent }
}

/**
 * 将生成的图片文件添加到消息内容中。
 * 该函数会查找最后一条助手消息，若其中包含生成图片的元数据，
 * 则将这些图片文件信息添加到该助手消息的 `images` 属性中。
 *
 * @param messages - 消息数组，包含多个消息对象。
 * @returns 更新后的消息数组，若有符合条件的图片文件，对应助手消息会添加 `images` 属性。
 */
export function addImageFileToContents(messages: Message[]) {
  // 查找消息数组中最后一条角色为 'assistant' 的消息
  const lastAssistantMessage = messages.findLast((m) => m.role === 'assistant')
  // 若未找到最后一条助手消息，直接返回原始消息数组
  if (!lastAssistantMessage) return messages
  // 从最后一条助手消息中查找图片块
  const blocks = findImageBlocks(lastAssistantMessage)
  // 若未找到图片块或图片块数量为 0，直接返回原始消息数组
  if (!blocks || blocks.length === 0) return messages
  // 检查所有图片块的元数据中是否都不包含生成图片信息，若是则直接返回原始消息数组
  if (blocks.every((v) => !v.metadata?.generateImage)) {
    return messages
  }

  // 从图片块的元数据中提取生成的图片文件信息，并将其展平为一维数组
  const imageFiles = blocks.map((v) => v.metadata?.generateImage?.images).flat()
  // 创建一个新的助手消息对象，复制最后一条助手消息的属性，并添加图片文件信息到 `images` 属性
  const updatedAssistantMessage = {
    ...lastAssistantMessage,
    images: imageFiles
  }

  // 遍历消息数组，将最后一条助手消息替换为更新后的助手消息，其他消息保持不变
  return messages.map((message) => (message.id === lastAssistantMessage.id ? updatedAssistantMessage : message))
}
