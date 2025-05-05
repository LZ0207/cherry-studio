import { DEFAULT_MAX_TOKENS } from '@renderer/config/constant'
import {
  findTokenLimit,
  getOpenAIWebSearchParams,
  isHunyuanSearchModel,
  isOpenAIWebSearch,
  isReasoningModel,
  isSupportedModel,
  isSupportedReasoningEffortGrokModel,
  isSupportedReasoningEffortModel,
  isSupportedReasoningEffortOpenAIModel,
  isSupportedThinkingTokenClaudeModel,
  isSupportedThinkingTokenModel,
  isSupportedThinkingTokenQwenModel,
  isVisionModel,
  isZhipuModel
} from '@renderer/config/models'
import { getStoreSetting } from '@renderer/hooks/useSettings'
import i18n from '@renderer/i18n'
import { getAssistantSettings, getDefaultModel, getTopNamingModel } from '@renderer/services/AssistantService'
import { EVENT_NAMES } from '@renderer/services/EventService'
import FileManager from '@renderer/services/FileManager'
import {
  filterContextMessages,
  filterEmptyMessages,
  filterUserRoleStartMessages
} from '@renderer/services/MessagesService'
import { processReqMessages } from '@renderer/services/ModelMessageService'
import store from '@renderer/store'
import {
  Assistant,
  EFFORT_RATIO,
  FileTypes,
  GenerateImageParams,
  MCPToolResponse,
  Model,
  Provider,
  Suggestion,
  Usage,
  WebSearchSource
} from '@renderer/types'
import { ChunkType, LLMWebSearchCompleteChunk } from '@renderer/types/chunk'
import { Message } from '@renderer/types/newMessage'
import { removeSpecialCharactersForTopicName } from '@renderer/utils'
import { addImageFileToContents } from '@renderer/utils/formats'
import {
  convertLinks,
  convertLinksToHunyuan,
  convertLinksToOpenRouter,
  convertLinksToZhipu
} from '@renderer/utils/linkConverter'
import { mcpToolCallResponseToOpenAIMessage, parseAndCallTools } from '@renderer/utils/mcp-tools'
import { findFileBlocks, findImageBlocks, getMainTextContent } from '@renderer/utils/messageUtils/find'
import { buildSystemPrompt } from '@renderer/utils/prompt'
import { isEmpty, takeRight } from 'lodash'
import OpenAI, { AzureOpenAI, toFile } from 'openai'
import {
  ChatCompletionContentPart,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam
} from 'openai/resources'
import { FileLike } from 'openai/uploads'

import { CompletionsParams } from '.'
import BaseProvider from './BaseProvider'

export default class OpenAIProvider extends BaseProvider {
  private sdk: OpenAI

  constructor(provider: Provider) {
    super(provider)

    if (provider.id === 'azure-openai' || provider.type === 'azure-openai') {
      this.sdk = new AzureOpenAI({
        dangerouslyAllowBrowser: true,
        apiKey: this.apiKey,
        apiVersion: provider.apiVersion,
        endpoint: provider.apiHost
      })
      return
    }

    this.sdk = new OpenAI({
      dangerouslyAllowBrowser: true,
      apiKey: this.apiKey,
      baseURL: this.getBaseURL(),
      defaultHeaders: {
        ...this.defaultHeaders(),
        ...(this.provider.id === 'copilot' ? { 'editor-version': 'vscode/1.97.2' } : {}),
        ...(this.provider.id === 'copilot' ? { 'copilot-vision-request': 'true' } : {})
      }
    })
  }

  /**
   * Check if the provider does not support files
   * @returns True if the provider does not support files, false otherwise
   */
  private get isNotSupportFiles() {
    if (this.provider?.isNotSupportArrayContent) {
      return true
    }

    const providers = ['deepseek', 'baichuan', 'minimax', 'xirang']

    return providers.includes(this.provider.id)
  }

  /**
   * Extract the file content from the message
   * @param message - The message
   * @returns The file content
   */
  private async extractFileContent(message: Message) {
    const fileBlocks = findFileBlocks(message)
    if (fileBlocks.length > 0) {
      const textFileBlocks = fileBlocks.filter(
        (fb) => fb.file && [FileTypes.TEXT, FileTypes.DOCUMENT].includes(fb.file.type)
      )

      if (textFileBlocks.length > 0) {
        let text = ''
        const divider = '\n\n---\n\n'

        for (const fileBlock of textFileBlocks) {
          const file = fileBlock.file
          const fileContent = (await window.api.file.read(file.id + file.ext)).trim()
          const fileNameRow = 'file: ' + file.origin_name + '\n\n'
          text = text + fileNameRow + fileContent + divider
        }

        return text
      }
    }

    return ''
  }

  /**
   * Get the message parameter
   * @param message - The message
   * @param model - The model
   * @returns The message parameter
   */
  private async getMessageParam(
    message: Message,
    model: Model
  ): Promise<OpenAI.Chat.Completions.ChatCompletionMessageParam> {
    const isVision = isVisionModel(model)
    const content = await this.getMessageContent(message)
    const fileBlocks = findFileBlocks(message)
    const imageBlocks = findImageBlocks(message)

    if (fileBlocks.length === 0 && imageBlocks.length === 0) {
      return {
        role: message.role === 'system' ? 'user' : message.role,
        content
      }
    }

    // If the model does not support files, extract the file content
    if (this.isNotSupportFiles) {
      const fileContent = await this.extractFileContent(message)

      return {
        role: message.role === 'system' ? 'user' : message.role,
        content: content + '\n\n---\n\n' + fileContent
      }
    }

    // If the model supports files, add the file content to the message
    const parts: ChatCompletionContentPart[] = []

    if (content) {
      parts.push({ type: 'text', text: content })
    }

    for (const imageBlock of imageBlocks) {
      if (isVision) {
        if (imageBlock.file) {
          const image = await window.api.file.base64Image(imageBlock.file.id + imageBlock.file.ext)
          parts.push({ type: 'image_url', image_url: { url: image.data } })
        } else if (imageBlock.url && imageBlock.url.startsWith('data:')) {
          parts.push({ type: 'image_url', image_url: { url: imageBlock.url } })
        }
      }
    }

    for (const fileBlock of fileBlocks) {
      const file = fileBlock.file
      if (!file) continue

      if ([FileTypes.TEXT, FileTypes.DOCUMENT].includes(file.type)) {
        const fileContent = await (await window.api.file.read(file.id + file.ext)).trim()
        parts.push({
          type: 'text',
          text: file.origin_name + '\n' + fileContent
        })
      }
    }

    return {
      role: message.role === 'system' ? 'user' : message.role,
      content: parts
    } as ChatCompletionMessageParam
  }

  /**
   * Get the temperature for the assistant
   * @param assistant - The assistant
   * @param model - The model
   * @returns The temperature
   */
  private getTemperature(assistant: Assistant, model: Model) {
    return isReasoningModel(model) || isOpenAIWebSearch(model) ? undefined : assistant?.settings?.temperature
  }

  /**
   * Get the provider specific parameters for the assistant
   * @param assistant - The assistant
   * @param model - The model
   * @returns The provider specific parameters
   */
  private getProviderSpecificParameters(assistant: Assistant, model: Model) {
    const { maxTokens } = getAssistantSettings(assistant)

    if (this.provider.id === 'openrouter') {
      if (model.id.includes('deepseek-r1')) {
        return {
          include_reasoning: true
        }
      }
    }

    if (this.isOpenAIReasoning(model)) {
      return {
        max_tokens: undefined,
        max_completion_tokens: maxTokens
      }
    }

    return {}
  }

  /**
   * Get the top P for the assistant
   * @param assistant - The assistant
   * @param model - The model
   * @returns The top P
   */
  private getTopP(assistant: Assistant, model: Model) {
    if (isReasoningModel(model) || isOpenAIWebSearch(model)) return undefined

    return assistant?.settings?.topP
  }

  /**
   * Get the reasoning effort for the assistant
   * @param assistant - The assistant
   * @param model - The model
   * @returns The reasoning effort
   */
  private getReasoningEffort(assistant: Assistant, model: Model) {
    if (this.provider.id === 'groq') {
      return {}
    }

    if (!isReasoningModel(model)) {
      return {}
    }
    const reasoningEffort = assistant?.settings?.reasoning_effort
    if (!reasoningEffort) {
      if (isSupportedThinkingTokenQwenModel(model)) {
        return { enable_thinking: false }
      }

      if (isSupportedThinkingTokenClaudeModel(model)) {
        return { thinking: { type: 'disabled' } }
      }

      return {}
    }
    const effortRatio = EFFORT_RATIO[reasoningEffort]
    const budgetTokens = Math.floor((findTokenLimit(model.id)?.max || 0) * effortRatio)
    // OpenRouter models
    if (model.provider === 'openrouter') {
      if (isSupportedReasoningEffortModel(model) || isSupportedThinkingTokenClaudeModel(model)) {
        return {
          reasoning: {
            effort: assistant?.settings?.reasoning_effort
          }
        }
      }
      if (isSupportedThinkingTokenModel(model)) {
        return {
          reasoning: {
            max_tokens: budgetTokens
          }
        }
      }
    }

    // Qwen models
    if (isSupportedThinkingTokenQwenModel(model)) {
      return {
        enable_thinking: true,
        thinking_budget: budgetTokens
      }
    }

    // Grok models
    if (isSupportedReasoningEffortGrokModel(model)) {
      return {
        reasoning_effort: assistant?.settings?.reasoning_effort
      }
    }

    // OpenAI models
    if (isSupportedReasoningEffortOpenAIModel(model)) {
      return {
        reasoning_effort: assistant?.settings?.reasoning_effort
      }
    }

    // Claude models
    const { maxTokens } = getAssistantSettings(assistant)
    if (isSupportedThinkingTokenClaudeModel(model)) {
      return {
        thinking: {
          type: 'enabled',
          budget_tokens: Math.floor(Math.max(Math.min(budgetTokens, maxTokens || DEFAULT_MAX_TOKENS), 1024))
        }
      }
    }

    // Default case: no special thinking settings
    return {}
  }

  /**
   * Check if the model is an OpenAI reasoning model
   * @param model - The model
   * @returns True if the model is an OpenAI reasoning model, false otherwise
   */
  private isOpenAIReasoning(model: Model) {
    return model.id.startsWith('o1') || model.id.startsWith('o3') || model.id.startsWith('o4')
  }

  /**
   * Generate completions for the assistant
   * @param messages - The messages
   * @param assistant - The assistant
   * @param mcpTools - The MCP tools
   * @param onChunk - The onChunk callback
   * @param onFilterMessages - The onFilterMessages callback
   * @returns The completions
   */
  /**
   * 生成对话完成结果，支持流式输出、思考内容处理、工具调用和网络搜索等功能。
   *
   * @param {CompletionsParams} params - 包含消息、助手、MCP工具、回调函数等参数的对象。
   * @returns {Promise<void>} 一个Promise，在完成处理后解析。
   */
  async completions({ messages, assistant, mcpTools, onChunk, onFilterMessages }: CompletionsParams): Promise<void> {
    // 如果助手启用了生成图片功能，则调用生成图片的方法
    if (assistant.enableGenerateImage) {
      await this.generateImageByChat({ messages, assistant, onChunk } as CompletionsParams)
      return
    }
    // 获取默认模型
    const defaultModel = getDefaultModel()
    // 若助手有指定模型则使用，否则使用默认模型
    const model = assistant.model || defaultModel
    // 从助手设置中获取上下文数量、最大令牌数和是否启用流式输出
    const { contextCount, maxTokens, streamOutput } = getAssistantSettings(assistant)
    // 检查是否启用了网络搜索
    const isEnabledWebSearch = assistant.enableWebSearch || !!assistant.webSearchProviderId
    // 将图片文件添加到消息内容中
    messages = addImageFileToContents(messages)
    // 初始化系统消息
    let systemMessage = { role: 'system', content: assistant.prompt || '' }
    // 如果是支持推理的OpenAI模型，修改系统消息的角色和内容
    if (isSupportedReasoningEffortOpenAIModel(model)) {
      systemMessage = {
        role: 'developer',
        content: `Formatting re-enabled${systemMessage ? '\n' + systemMessage.content : ''}`
      }
    }
    // 如果存在MCP工具，构建系统提示
    if (mcpTools && mcpTools.length > 0) {
      systemMessage.content = buildSystemPrompt(systemMessage.content || '', mcpTools)
    }

    // 初始化用户消息数组
    const userMessages: ChatCompletionMessageParam[] = []
    // 过滤消息，包括空消息、上下文消息和以用户角色开头的消息
    const _messages = filterUserRoleStartMessages(
      filterEmptyMessages(filterContextMessages(takeRight(messages, contextCount + 1)))
    )

    // 调用过滤消息的回调函数
    onFilterMessages(_messages)

    // 将每条消息转换为符合OpenAI API要求的消息参数
    for (const message of _messages) {
      userMessages.push(await this.getMessageParam(message, model))
    }

    // 检查是否支持流式输出
    const isSupportStreamOutput = () => {
      return streamOutput
    }

    // 标记是否有推理内容
    let hasReasoningContent = false
    // 存储上一个chunk的内容
    let lastChunk = ''
    /**
     * 检查推理是否结束
     *
     * @param {OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta & { reasoning_content?: string; reasoning?: string; thinking?: string }} delta - 响应的增量内容
     * @returns {boolean} 如果推理结束返回true，否则返回false
     */
    const isReasoningJustDone = (
      delta: OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta & {
        reasoning_content?: string
        reasoning?: string
        thinking?: string
      }
    ) => {
      if (!delta?.content) return false

      // 检查当前chunk和上一个chunk的组合是否形成###Response标记
      const combinedChunks = lastChunk + delta.content
      lastChunk = delta.content

      // 检测思考结束
      if (combinedChunks.includes('###Response') || delta.content === '</think>') {
        return true
      }

      // 如果有reasoning_content或reasoning，说明是在思考中
      if (delta?.reasoning_content || delta?.reasoning || delta?.thinking) {
        hasReasoningContent = true
      }

      // 如果之前有reasoning_content或reasoning，现在有普通content，说明思考结束
      if (hasReasoningContent && delta.content) {
        return true
      }

      return false
    }

    // 记录第一个token的时间
    let time_first_token_millsec = 0
    // 记录第一个token的时间差
    let time_first_token_millsec_delta = 0
    // 记录第一个内容的时间
    let time_first_content_millsec = 0
    // 记录开始时间
    const start_time_millsec = new Date().getTime()
    console.log(
      `completions start_time_millsec ${new Date(start_time_millsec).toLocaleString(undefined, {
        year: 'numeric',
        month: 'numeric',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        second: 'numeric',
        fractionalSecondDigits: 3
      })}`
    )
    // 找到最后一条用户消息
    const lastUserMessage = _messages.findLast((m) => m.role === 'user')
    // 创建中止控制器
    const { abortController, cleanup, signalPromise } = this.createAbortController(lastUserMessage?.id, true)
    // 获取信号
    const { signal } = abortController
    // 检查是否为Copilot并更新API密钥
    await this.checkIsCopilot()

    // 当 systemMessage 内容为空时不发送 systemMessage
    let reqMessages: ChatCompletionMessageParam[]
    if (!systemMessage.content) {
      reqMessages = [...userMessages]
    } else {
      reqMessages = [systemMessage, ...userMessages].filter(Boolean) as ChatCompletionMessageParam[]
    }

    // 存储MCP工具响应
    const toolResponses: MCPToolResponse[] = []

    /**
     * 处理工具调用
     *
     * @param {string} content - 消息内容
     * @param {number} idx - 调用索引
     * @returns {Promise<void>} 一个Promise，在处理完成后解析。
     */
    const processToolUses = async (content: string, idx: number) => {
      // 解析并调用工具
      const toolResults = await parseAndCallTools(
        content,
        toolResponses,
        onChunk,
        idx,
        mcpToolCallResponseToOpenAIMessage,
        mcpTools,
        isVisionModel(model)
      )

      if (toolResults.length > 0) {
        // 将助手消息添加到请求消息中
        reqMessages.push({
          role: 'assistant',
          content: content
        } as ChatCompletionMessageParam)
        // 将工具调用结果添加到请求消息中
        toolResults.forEach((ts) => reqMessages.push(ts as ChatCompletionMessageParam))

        console.debug('[tool] reqMessages before processing', model.id, reqMessages)
        // 处理请求消息
        reqMessages = processReqMessages(model, reqMessages)
        console.debug('[tool] reqMessages', model.id, reqMessages)
        // 发起新的流式请求
        const newStream = await this.sdk.chat.completions
          // @ts-ignore key is not typed
          .create(
            {
              model: model.id,
              messages: reqMessages,
              temperature: this.getTemperature(assistant, model),
              top_p: this.getTopP(assistant, model),
              max_tokens: maxTokens,
              keep_alive: this.keepAliveTime,
              stream: isSupportStreamOutput(),
              // tools: tools,
              ...getOpenAIWebSearchParams(assistant, model),
              ...this.getReasoningEffort(assistant, model),
              ...this.getProviderSpecificParameters(assistant, model),
              ...this.getCustomParameters(assistant)
            },
            {
              signal
            }
          )
        // 处理新的流
        await processStream(newStream, idx + 1)
      }
    }

    /**
     * 处理流式响应
     *
     * @param {any} stream - 流式响应
     * @param {number} idx - 调用索引
     * @returns {Promise<void>} 一个Promise，在处理完成后解析。
     */
    const processStream = async (stream: any, idx: number) => {
      // 处理非流式情况
      if (!isSupportStreamOutput()) {
        const time_completion_millsec = new Date().getTime() - start_time_millsec
        // 计算最终指标
        const finalMetrics = {
          completion_tokens: stream.usage?.completion_tokens,
          time_completion_millsec,
          time_first_token_millsec: 0 // 非流式，第一个token时间无关紧要
        }

        // 创建一个合成的usage对象，如果stream.usage未定义
        const finalUsage = stream.usage
        // 分别调用onChunk处理文本和usage/metrics
        if (stream.choices[0].message?.content) {
          onChunk({ type: ChunkType.TEXT_COMPLETE, text: stream.choices[0].message.content })
        }

        // 始终发送usage和metrics数据
        onChunk({ type: ChunkType.BLOCK_COMPLETE, response: { usage: finalUsage, metrics: finalMetrics } })
        return
      }

      // 累积内容用于工具处理
      let content = ''
      // 累积思考内容
      let thinkingContent = ''
      // 记录最终的完成时间差
      let final_time_completion_millsec_delta = 0
      // 记录最终的思考时间差
      let final_time_thinking_millsec_delta = 0
      // 存储最后收到的usage对象
      let lastUsage: Usage | undefined = undefined
      // 标记是否为第一个chunk
      let isFirstChunk = true
      // 标记是否为第一个思考chunk
      let isFirstThinkingChunk = true
      // 遍历流式响应
      for await (const chunk of stream) {
        // 如果聊天完成被暂停，则退出循环
        if (window.keyv.get(EVENT_NAMES.CHAT_COMPLETION_PAUSED)) {
          break
        }

        // 获取响应的增量内容
        const delta = chunk.choices[0]?.delta
        // 获取完成原因
        const finishReason = chunk.choices[0]?.finish_reason

        // --- 增量调用onChunk ---

        // 1. 推理内容
        const reasoningContent = delta?.reasoning_content || delta?.reasoning
        // 获取当前时间
        const currentTime = new Date().getTime()

        if (time_first_token_millsec === 0 && isFirstThinkingChunk && reasoningContent) {
          // 记录第一个token的时间
          time_first_token_millsec = currentTime
          // 记录第一个token的时间差
          time_first_token_millsec_delta = currentTime - start_time_millsec
          console.log(
            `completions time_first_token_millsec ${new Date(currentTime).toLocaleString(undefined, {
              year: 'numeric',
              month: 'numeric',
              day: 'numeric',
              hour: 'numeric',
              minute: 'numeric',
              second: 'numeric',
              fractionalSecondDigits: 3
            })}`
          )
          isFirstThinkingChunk = false
        }
        if (reasoningContent) {
          // 累积思考内容
          thinkingContent += reasoningContent
          // 标记有推理内容
          hasReasoningContent = true

          // 计算思考时间
          const thinking_time = currentTime - time_first_token_millsec
          // 发送思考增量内容
          onChunk({ type: ChunkType.THINKING_DELTA, text: reasoningContent, thinking_millsec: thinking_time })
        }

        if (isReasoningJustDone(delta)) {
          if (time_first_content_millsec === 0) {
            time_first_content_millsec = currentTime
            final_time_thinking_millsec_delta = time_first_content_millsec - time_first_token_millsec
            // 发送思考完成内容
            onChunk({
              type: ChunkType.THINKING_COMPLETE,
              text: thinkingContent,
              thinking_millsec: final_time_thinking_millsec_delta
            })

            thinkingContent = ''
            isFirstThinkingChunk = true
            hasReasoningContent = false
          }
        }

        // 2. 文本内容
        if (delta?.content) {
          if (assistant.enableWebSearch) {
            if (delta?.annotations) {
              // 转换链接
              delta.content = convertLinks(delta.content || '', isFirstChunk)
            } else if (assistant.model?.provider === 'openrouter') {
              // 转换OpenRouter链接
              delta.content = convertLinksToOpenRouter(delta.content || '', isFirstChunk)
            } else if (isZhipuModel(assistant.model)) {
              // 转换智谱链接
              delta.content = convertLinksToZhipu(delta.content || '', isFirstChunk)
            } else if (isHunyuanSearchModel(assistant.model)) {
              // 转换混元搜索链接
              delta.content = convertLinksToHunyuan(
                delta.content || '',
                chunk.search_info.search_results || [],
                isFirstChunk
              )
            }
          }
          // 说明前面没有思考内容
          if (isFirstChunk && time_first_token_millsec === 0 && time_first_token_millsec_delta === 0) {
            isFirstChunk = false
            time_first_token_millsec = currentTime
            time_first_token_millsec_delta = time_first_token_millsec - start_time_millsec
          }
          // 累积内容用于工具处理
          content += delta.content

          // 发送文本增量内容
          onChunk({ type: ChunkType.TEXT_DELTA, text: delta.content })
        }
        // 当完成原因不为空时
        if (!isEmpty(finishReason)) {
          // 发送文本完成内容
          onChunk({ type: ChunkType.TEXT_COMPLETE, text: content })
          final_time_completion_millsec_delta = currentTime - start_time_millsec
          console.log(
            `completions final_time_completion_millsec ${new Date(currentTime).toLocaleString(undefined, {
              year: 'numeric',
              month: 'numeric',
              day: 'numeric',
              hour: 'numeric',
              minute: 'numeric',
              second: 'numeric',
              fractionalSecondDigits: 3
            })}`
          )
          // 6. Usage (如果每个chunk提供) - 捕获最后已知的usage
          if (chunk.usage) {
            // 更新最后已知的usage信息
            lastUsage = chunk.usage
          }

          // 3. 网络搜索
          if (delta?.annotations) {
            // 发送OpenAI网络搜索完成内容
            onChunk({
              type: ChunkType.LLM_WEB_SEARCH_COMPLETE,
              llm_web_search: {
                results: delta.annotations,
                source: WebSearchSource.OPENAI
              }
            } as LLMWebSearchCompleteChunk)
          }

          if (assistant.model?.provider === 'perplexity') {
            const citations = chunk.citations
            if (citations) {
              // 发送Perplexity网络搜索完成内容
              onChunk({
                type: ChunkType.LLM_WEB_SEARCH_COMPLETE,
                llm_web_search: {
                  results: citations,
                  source: WebSearchSource.PERPLEXITY
                }
              } as LLMWebSearchCompleteChunk)
            }
          }
          if (isEnabledWebSearch && isZhipuModel(model) && finishReason === 'stop' && chunk?.web_search) {
            // 发送智谱网络搜索完成内容
            onChunk({
              type: ChunkType.LLM_WEB_SEARCH_COMPLETE,
              llm_web_search: {
                results: chunk.web_search,
                source: WebSearchSource.ZHIPU
              }
            } as LLMWebSearchCompleteChunk)
          }
          if (isEnabledWebSearch && isHunyuanSearchModel(model) && chunk?.search_info?.search_results) {
            // 发送混元网络搜索完成内容
            onChunk({
              type: ChunkType.LLM_WEB_SEARCH_COMPLETE,
              llm_web_search: {
                results: chunk.search_info.search_results,
                source: WebSearchSource.HUNYUAN
              }
            } as LLMWebSearchCompleteChunk)
          }
        }

        // --- 增量调用onChunk结束 ---
      } // 结束for await循环

      // 在主流内容处理完成后调用processToolUses
      await processToolUses(content, idx)

      // 发送最终的block_complete chunk
      onChunk({
        type: ChunkType.BLOCK_COMPLETE,
        response: {
          // 使用增强的usage对象
          usage: lastUsage,
          metrics: {
            // 从最后一个usage对象获取完成令牌数
            completion_tokens: lastUsage?.completion_tokens,
            time_completion_millsec: final_time_completion_millsec_delta,
            time_first_token_millsec: time_first_token_millsec_delta,
            time_thinking_millsec: final_time_thinking_millsec_delta
          }
        }
      })

      // FIXME: 临时方案，重置时间戳和思考内容
      time_first_token_millsec = 0
      time_first_content_millsec = 0
    }

    console.debug('[completions] reqMessages before processing', model.id, reqMessages)
    // 处理请求消息
    reqMessages = processReqMessages(model, reqMessages)
    console.debug('[completions] reqMessages', model.id, reqMessages)
    // 等待接口返回流
    onChunk({ type: ChunkType.LLM_RESPONSE_CREATED })
    // 发起流式请求
    const stream = await this.sdk.chat.completions
      // @ts-ignore key is not typed
      .create(
        {
          model: model.id,
          messages: reqMessages,
          temperature: this.getTemperature(assistant, model),
          top_p: this.getTopP(assistant, model),
          max_tokens: maxTokens,
          keep_alive: this.keepAliveTime,
          stream: isSupportStreamOutput(),
          // tools: tools,
          ...getOpenAIWebSearchParams(assistant, model),
          ...this.getReasoningEffort(assistant, model),
          ...this.getProviderSpecificParameters(assistant, model),
          ...this.getCustomParameters(assistant)
        },
        {
          signal
        }
      )

    // 处理流式响应并在完成后执行清理操作
    await processStream(stream, 0).finally(cleanup)

    // 捕获signal的错误
    await signalPromise?.promise?.catch((error) => {
      throw error
    })
  }

  /**
   * Translate a message
   * @param message - The message
   * @param assistant - The assistant
   * @param onResponse - The onResponse callback
   * @returns The translated message
   */
  async translate(content: string, assistant: Assistant, onResponse?: (text: string, isComplete: boolean) => void) {
    const defaultModel = getDefaultModel()
    const model = assistant.model || defaultModel
    const messagesForApi = content
      ? [
          { role: 'system', content: assistant.prompt },
          { role: 'user', content }
        ]
      : [{ role: 'user', content: assistant.prompt }]

    const isOpenAIReasoning = this.isOpenAIReasoning(model)

    const isSupportedStreamOutput = () => {
      if (!onResponse) {
        return false
      }
      if (isOpenAIReasoning) {
        return false
      }
      return true
    }

    const stream = isSupportedStreamOutput()

    await this.checkIsCopilot()

    // console.debug('[translate] reqMessages', model.id, message)
    // @ts-ignore key is not typed
    const response = await this.sdk.chat.completions.create({
      model: model.id,
      messages: messagesForApi as ChatCompletionMessageParam[],
      stream,
      keep_alive: this.keepAliveTime,
      temperature: assistant?.settings?.temperature
    })

    if (!stream) {
      return response.choices[0].message?.content || ''
    }

    let text = ''
    let isThinking = false
    const isReasoning = isReasoningModel(model)

    for await (const chunk of response) {
      const deltaContent = chunk.choices[0]?.delta?.content || ''

      if (isReasoning) {
        if (deltaContent.includes('<think>')) {
          isThinking = true
        }

        if (!isThinking) {
          text += deltaContent
          onResponse?.(text, false)
        }

        if (deltaContent.includes('</think>')) {
          isThinking = false
        }
      } else {
        text += deltaContent
        onResponse?.(text, false)
      }
    }

    onResponse?.(text, true)

    return text
  }

  /**
   * Summarize a message
   * @param messages - The messages
   * @param assistant - The assistant
   * @returns The summary
   */
  public async summaries(messages: Message[], assistant: Assistant): Promise<string> {
    const model = getTopNamingModel() || assistant.model || getDefaultModel()

    const userMessages = takeRight(messages, 5)
      .filter((message) => !message.isPreset)
      .map((message) => ({
        role: message.role,
        content: getMainTextContent(message)
      }))

    const userMessageContent = userMessages.reduce((prev, curr) => {
      const content = curr.role === 'user' ? `User: ${curr.content}` : `Assistant: ${curr.content}`
      return prev + (prev ? '\n' : '') + content
    }, '')

    const systemMessage = {
      role: 'system',
      content: getStoreSetting('topicNamingPrompt') || i18n.t('prompts.title')
    }

    const userMessage = {
      role: 'user',
      content: userMessageContent
    }

    await this.checkIsCopilot()

    console.debug('[summaries] reqMessages', model.id, [systemMessage, userMessage])
    // @ts-ignore key is not typed
    const response = await this.sdk.chat.completions.create({
      model: model.id,
      messages: [systemMessage, userMessage] as ChatCompletionMessageParam[],
      stream: false,
      keep_alive: this.keepAliveTime,
      max_tokens: 1000
    })

    // 针对思考类模型的返回，总结仅截取</think>之后的内容
    let content = response.choices[0].message?.content || ''
    content = content.replace(/^<think>(.*?)<\/think>/s, '')

    return removeSpecialCharactersForTopicName(content.substring(0, 50))
  }

  /**
   * Summarize a message for search
   * @param messages - The messages
   * @param assistant - The assistant
   * @returns The summary
   */
  public async summaryForSearch(messages: Message[], assistant: Assistant): Promise<string | null> {
    const model = assistant.model || getDefaultModel()

    const systemMessage = {
      role: 'system',
      content: assistant.prompt
    }

    const messageContents = messages.map((m) => getMainTextContent(m))
    const userMessageContent = messageContents.join('\n')

    const userMessage = {
      role: 'user',
      content: userMessageContent
    }
    console.debug('[summaryForSearch] reqMessages', model.id, [systemMessage, userMessage])

    const lastUserMessage = messages[messages.length - 1]
    console.log('lastUserMessage?.id', lastUserMessage?.id)
    const { abortController, cleanup } = this.createAbortController(lastUserMessage?.id)
    const { signal } = abortController

    const response = await this.sdk.chat.completions
      // @ts-ignore key is not typed
      .create(
        {
          model: model.id,
          messages: [systemMessage, userMessage] as ChatCompletionMessageParam[],
          stream: false,
          keep_alive: this.keepAliveTime,
          max_tokens: 1000
        },
        {
          timeout: 20 * 1000,
          signal: signal
        }
      )
      .finally(cleanup)

    // 针对思考类模型的返回，总结仅截取</think>之后的内容
    let content = response.choices[0].message?.content || ''
    content = content.replace(/^<think>(.*?)<\/think>/s, '')

    return content
  }

  /**
   * Generate text
   * @param prompt - The prompt
   * @param content - The content
   * @returns The generated text
   */
  public async generateText({ prompt, content }: { prompt: string; content: string }): Promise<string> {
    const model = getDefaultModel()

    await this.checkIsCopilot()

    const response = await this.sdk.chat.completions.create({
      model: model.id,
      stream: false,
      messages: [
        { role: 'system', content: prompt },
        { role: 'user', content }
      ]
    })

    return response.choices[0].message?.content || ''
  }

  /**
   * Generate suggestions
   * @param messages - The messages
   * @param assistant - The assistant
   * @returns The suggestions
   */
  async suggestions(messages: Message[], assistant: Assistant): Promise<Suggestion[]> {
    const model = assistant.model

    if (!model) {
      return []
    }

    await this.checkIsCopilot()

    const userMessagesForApi = messages
      .filter((m) => m.role === 'user')
      .map((m) => ({
        role: m.role,
        content: getMainTextContent(m)
      }))

    const response: any = await this.sdk.request({
      method: 'post',
      path: '/advice_questions',
      body: {
        messages: userMessagesForApi,
        model: model.id,
        max_tokens: 0,
        temperature: 0,
        n: 0
      }
    })

    return response?.questions?.filter(Boolean)?.map((q: any) => ({ content: q })) || []
  }

  /**
   * Check if the model is valid
   * @param model - The model
   * @param stream - Whether to use streaming interface
   * @returns The validity of the model
   */
  public async check(model: Model, stream: boolean = false): Promise<{ valid: boolean; error: Error | null }> {
    if (!model) {
      return { valid: false, error: new Error('No model found') }
    }
    const body = {
      model: model.id,
      messages: [{ role: 'user', content: 'hi' }],
      stream
    }

    try {
      await this.checkIsCopilot()
      console.debug('[checkModel] body', model.id, body)
      if (!stream) {
        const response = await this.sdk.chat.completions.create(body as ChatCompletionCreateParamsNonStreaming)
        if (!response?.choices[0].message) {
          throw new Error('Empty response')
        }
        return { valid: true, error: null }
      } else {
        const response: any = await this.sdk.chat.completions.create(body as any)
        // 等待整个流式响应结束
        let hasContent = false
        for await (const chunk of response) {
          if (chunk.choices?.[0]?.delta?.content) {
            hasContent = true
          }
        }
        if (hasContent) {
          return { valid: true, error: null }
        }
        throw new Error('Empty streaming response')
      }
    } catch (error: any) {
      return {
        valid: false,
        error
      }
    }
  }

  /**
   * Get the models
   * @returns The models
   */
  public async models(): Promise<OpenAI.Models.Model[]> {
    try {
      await this.checkIsCopilot()

      const response = await this.sdk.models.list()

      if (this.provider.id === 'github') {
        // @ts-ignore key is not typed
        return response.body
          .map((model) => ({
            id: model.name,
            description: model.summary,
            object: 'model',
            owned_by: model.publisher
          }))
          .filter(isSupportedModel)
      }

      if (this.provider.id === 'together') {
        // @ts-ignore key is not typed
        return response?.body
          .map((model: any) => ({
            id: model.id,
            description: model.display_name,
            object: 'model',
            owned_by: model.organization
          }))
          .filter(isSupportedModel)
      }

      const models = response?.data || []

      return models.filter(isSupportedModel)
    } catch (error) {
      return []
    }
  }

  /**
   * Generate an image
   * @param params - The parameters
   * @returns The generated image
   */
  public async generateImage({
    model,
    prompt,
    negativePrompt,
    imageSize,
    batchSize,
    seed,
    numInferenceSteps,
    guidanceScale,
    signal,
    promptEnhancement
  }: GenerateImageParams): Promise<string[]> {
    const response = (await this.sdk.request({
      method: 'post',
      path: '/images/generations',
      signal,
      body: {
        model,
        prompt,
        negative_prompt: negativePrompt,
        image_size: imageSize,
        batch_size: batchSize,
        seed: seed ? parseInt(seed) : undefined,
        num_inference_steps: numInferenceSteps,
        guidance_scale: guidanceScale,
        prompt_enhancement: promptEnhancement
      }
    })) as { data: Array<{ url: string }> }

    return response.data.map((item) => item.url)
  }

  /**
   * Get the embedding dimensions
   * @param model - The model
   * @returns The embedding dimensions
   */
  public async getEmbeddingDimensions(model: Model): Promise<number> {
    await this.checkIsCopilot()

    const data = await this.sdk.embeddings.create({
      model: model.id,
      input: model?.provider === 'baidu-cloud' ? ['hi'] : 'hi'
    })
    return data.data[0].embedding.length
  }

  public async checkIsCopilot() {
    if (this.provider.id !== 'copilot') return
    const defaultHeaders = store.getState().copilot.defaultHeaders
    // copilot每次请求前需要重新获取token，因为token中附带时间戳
    const { token } = await window.api.copilot.getToken(defaultHeaders)
    this.sdk.apiKey = token
  }

  public async generateImageByChat({ messages, assistant, onChunk }: CompletionsParams): Promise<void> {
    const defaultModel = getDefaultModel()
    const model = assistant.model || defaultModel
    // save image data from the last assistant message
    messages = addImageFileToContents(messages)
    const lastUserMessage = messages.findLast((m) => m.role === 'user')
    const lastAssistantMessage = messages.findLast((m) => m.role === 'assistant')
    if (!lastUserMessage) {
      return
    }

    const { abortController } = this.createAbortController(lastUserMessage?.id, true)
    const { signal } = abortController
    const content = getMainTextContent(lastUserMessage!)
    let response: OpenAI.Images.ImagesResponse | null = null
    let images: FileLike[] = []

    try {
      if (lastUserMessage) {
        const UserFiles = findImageBlocks(lastUserMessage)
        const validUserFiles = UserFiles.filter((f) => f.file) // Filter out files that are undefined first
        const userImages = await Promise.all(
          validUserFiles.map(async (f) => {
            // f.file is guaranteed to exist here due to the filter above
            const fileInfo = f.file!
            const binaryData = await FileManager.readBinaryImage(fileInfo)
            const file = await toFile(binaryData, fileInfo.origin_name || 'image.png', {
              type: 'image/png'
            })
            return file
          })
        )
        images = images.concat(userImages)
      }

      if (lastAssistantMessage) {
        const assistantFiles = findImageBlocks(lastAssistantMessage)
        const assistantImages = await Promise.all(
          assistantFiles.filter(Boolean).map(async (f) => {
            const base64Data = f?.url?.replace(/^data:image\/\w+;base64,/, '')
            if (!base64Data) return null
            const binary = atob(base64Data)
            const bytes = new Uint8Array(binary.length)
            for (let i = 0; i < binary.length; i++) {
              bytes[i] = binary.charCodeAt(i)
            }
            const file = await toFile(bytes, 'assistant_image.png', {
              type: 'image/png'
            })
            return file
          })
        )
        images = images.concat(assistantImages.filter(Boolean) as FileLike[])
      }
      onChunk({
        type: ChunkType.IMAGE_CREATED
      })

      const start_time_millsec = new Date().getTime()

      if (images.length > 0) {
        response = await this.sdk.images.edit(
          {
            model: model.id,
            image: images,
            prompt: content || ''
          },
          {
            signal,
            timeout: 300_000
          }
        )
      } else {
        response = await this.sdk.images.generate(
          {
            model: model.id,
            prompt: content || '',
            response_format: model.id.includes('gpt-image-1') ? undefined : 'b64_json'
          },
          {
            signal,
            timeout: 300_000
          }
        )
      }

      onChunk({
        type: ChunkType.IMAGE_COMPLETE,
        image: {
          type: 'base64',
          images: response?.data?.map((item) => `data:image/png;base64,${item.b64_json}`) || []
        }
      })

      onChunk({
        type: ChunkType.BLOCK_COMPLETE,
        response: {
          usage: {
            completion_tokens: response.usage?.output_tokens || 0,
            prompt_tokens: response.usage?.input_tokens || 0,
            total_tokens: response.usage?.total_tokens || 0
          },
          metrics: {
            completion_tokens: response.usage?.output_tokens || 0,
            time_first_token_millsec: 0, // Non-streaming, first token time is not relevant
            time_completion_millsec: new Date().getTime() - start_time_millsec
          }
        }
      })
    } catch (error: any) {
      console.error('[generateImageByChat] error', error)
      onChunk({
        type: ChunkType.ERROR,
        error
      })
    }
  }
}
