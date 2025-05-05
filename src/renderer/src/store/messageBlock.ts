// 导入 Google GenAI 的 GroundingMetadata 类型
import type { GroundingMetadata } from '@google/genai'
// 从 @reduxjs/toolkit 导入创建实体适配器、选择器和切片所需的函数及类型
import { createEntityAdapter, createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit'
// 导入引用列表中的 Citation 类型
import type { Citation } from '@renderer/pages/home/Messages/CitationsList'
// 导入 Web 搜索提供者响应和源类型
import { WebSearchProviderResponse, WebSearchSource } from '@renderer/types'
// 导入引用消息块和消息块类型
import type { CitationMessageBlock, MessageBlock } from '@renderer/types/newMessage'
// 导入消息块类型枚举
import { MessageBlockType } from '@renderer/types/newMessage'
// 导入 OpenAI 类型
import type OpenAI from 'openai'

// 导入根状态类型，确认从 store/index.ts 导出
import type { RootState } from './index'

// 1. 创建实体适配器 (Entity Adapter)
// 我们使用块的 `id` 作为唯一标识符。
/**
 * 消息块实体适配器，用于管理消息块的状态。
 * 使用消息块的 `id` 作为唯一标识符。
 */
const messageBlocksAdapter = createEntityAdapter<MessageBlock>()

// 2. 使用适配器定义初始状态 (Initial State)
// 如果需要，可以在规范化实体的旁边添加其他状态属性。
/**
 * 消息块的初始状态，包含加载状态和错误信息。
 */
const initialState = messageBlocksAdapter.getInitialState({
  loadingState: 'idle' as 'idle' | 'loading' | 'succeeded' | 'failed',
  error: null as string | null
})

// 3. 创建 Slice
/**
 * 消息块状态切片，包含消息块的 CRUD 操作和状态管理。
 */
const messageBlocksSlice = createSlice({
  name: 'messageBlocks',
  initialState,
  reducers: {
    // 使用适配器的 reducer 助手进行 CRUD 操作。
    // 这些 reducer 会自动处理规范化的状态结构。

    /** 添加或更新单个块 (Upsert)。 */
    /**
     * 添加或更新单个消息块。
     * @param payload - 要添加或更新的消息块。
     */
    upsertOneBlock: messageBlocksAdapter.upsertOne, // 期望 MessageBlock 作为 payload

    /** 添加或更新多个块。用于加载消息。 */
    /**
     * 添加或更新多个消息块，通常用于加载消息。
     * @param payload - 要添加或更新的消息块数组。
     */
    upsertManyBlocks: messageBlocksAdapter.upsertMany, // 期望 MessageBlock[] 作为 payload

    /** 根据 ID 移除单个块。 */
    /**
     * 根据消息块的 ID 移除单个消息块。
     * @param payload - 要移除的消息块的 ID。
     */
    removeOneBlock: messageBlocksAdapter.removeOne, // 期望 EntityId (string) 作为 payload

    /** 根据 ID 列表移除多个块。用于清理话题。 */
    /**
     * 根据消息块的 ID 列表移除多个消息块，通常用于清理话题。
     * @param payload - 要移除的消息块的 ID 数组。
     */
    removeManyBlocks: messageBlocksAdapter.removeMany, // 期望 EntityId[] (string[]) 作为 payload

    /** 移除所有块。用于完全重置。 */
    /**
     * 移除所有消息块，用于完全重置状态。
     */
    removeAllBlocks: messageBlocksAdapter.removeAll,

    // 你可以为其他状态属性（如加载/错误）添加自定义 reducer
    /**
     * 设置消息块的加载状态。
     * @param payload - 加载状态，可选值为 'idle' 或 'loading'。
     */
    setMessageBlocksLoading: (state, action: PayloadAction<'idle' | 'loading'>) => {
      state.loadingState = action.payload
      state.error = null
    },
    /**
     * 设置消息块的错误信息。
     * @param payload - 错误信息字符串。
     */
    setMessageBlocksError: (state, action: PayloadAction<string>) => {
      state.loadingState = 'failed'
      state.error = action.payload
    },
    // 注意：如果只想更新现有块，也可以使用 `updateOne`
    /**
     * 更新单个消息块的部分内容。
     * @param payload - 包含要更新的消息块 ID 和更新内容的对象。
     */
    updateOneBlock: messageBlocksAdapter.updateOne // 期望 { id: EntityId, changes: Partial<MessageBlock> }
  }
  // 如果需要处理其他 slice 的 action，可以在这里添加 extraReducers。
})

// 4. 导出 Actions 和 Reducer
/**
 * 添加或更新单个消息块的 action。
 */
export const {
  upsertOneBlock,
  upsertManyBlocks,
  removeOneBlock,
  removeManyBlocks,
  removeAllBlocks,
  setMessageBlocksLoading,
  setMessageBlocksError,
  updateOneBlock
} = messageBlocksSlice.actions

/**
 * 消息块的选择器集合，用于从状态中获取消息块相关数据。
 */
export const messageBlocksSelectors = messageBlocksAdapter.getSelectors<RootState>(
  (state) => state.messageBlocks // Ensure this matches the key in the root reducer
)

// --- Selector Integration --- START

/**
 * 根据消息块 ID 获取原始消息块实体。
 * @param state - 根状态对象。
 * @param blockId - 消息块的 ID，可选。
 * @returns 对应的消息块实体，若 ID 为空则返回 undefined。
 */
const selectBlockEntityById = (state: RootState, blockId: string | undefined) =>
  blockId ? messageBlocksSelectors.selectById(state, blockId) : undefined // Use adapter selector

// --- Centralized Citation Formatting Logic ---
/**
 * 从引用消息块中格式化引用信息。
 * @param block - 引用消息块，可选。
 * @returns 格式化后的引用信息数组。
 */
const formatCitationsFromBlock = (block: CitationMessageBlock | undefined): Citation[] => {
  if (!block) return []

  let formattedCitations: Citation[] = []
  // 1. Handle Web Search Responses (Non-Gemini)
  if (block.response) {
    switch (block.response.source) {
      case WebSearchSource.GEMINI:
        formattedCitations =
          (block.response?.results as GroundingMetadata)?.groundingChunks?.map((chunk, index) => ({
            number: index + 1,
            url: chunk?.web?.uri || '',
            title: chunk?.web?.title,
            showFavicon: false,
            type: 'websearch'
          })) || []
        break
      case WebSearchSource.OPENAI:
        formattedCitations =
          (block.response.results as OpenAI.Chat.Completions.ChatCompletionMessage.Annotation[])?.map((url, index) => {
            const urlCitation = url.url_citation
            let hostname: string | undefined
            try {
              hostname = urlCitation.title ? undefined : new URL(urlCitation.url).hostname
            } catch {
              hostname = urlCitation.url
            }
            return {
              number: index + 1,
              url: urlCitation.url,
              title: urlCitation.title,
              hostname: hostname,
              showFavicon: true,
              type: 'websearch'
            }
          }) || []
        break
      case WebSearchSource.OPENROUTER:
      case WebSearchSource.PERPLEXITY:
        formattedCitations =
          (block.response.results as any[])?.map((url, index) => {
            try {
              const hostname = new URL(url).hostname
              return {
                number: index + 1,
                url,
                hostname,
                showFavicon: true,
                type: 'websearch'
              }
            } catch {
              return {
                number: index + 1,
                url,
                hostname: url,
                showFavicon: true,
                type: 'websearch'
              }
            }
          }) || []
        break
      case WebSearchSource.ZHIPU:
      case WebSearchSource.HUNYUAN:
        formattedCitations =
          (block.response.results as any[])?.map((result, index) => ({
            number: index + 1,
            url: result.link || result.url,
            title: result.title,
            showFavicon: true,
            type: 'websearch'
          })) || []
        break
      case WebSearchSource.WEBSEARCH:
        formattedCitations =
          (block.response.results as WebSearchProviderResponse)?.results?.map((result, index) => ({
            number: index + 1,
            url: result.url,
            title: result.title,
            content: result.content,
            showFavicon: true,
            type: 'websearch'
          })) || []
        break
    }
  }
  // 3. Handle Knowledge Base References
  if (block.knowledge && block.knowledge.length > 0) {
    formattedCitations.push(
      ...block.knowledge.map((result, index) => ({
        number: index + 1,
        url: result.sourceUrl,
        title: result.sourceUrl,
        content: result.content,
        showFavicon: true,
        type: 'knowledge'
      }))
    )
  }
  // 4. Deduplicate by URL and Renumber Sequentially
  const urlSet = new Set<string>()
  return formattedCitations
    .filter((citation) => {
      if (!citation.url || urlSet.has(citation.url)) return false
      urlSet.add(citation.url)
      return true
    })
    .map((citation, index) => ({
      ...citation,
      number: index + 1
    }))
}
// --- End of Centralized Logic ---

/**
 * 记忆化选择器，根据消息块 ID 返回格式化后的引用信息。
 * @param state - 根状态对象。
 * @param blockId - 消息块的 ID。
 * @returns 格式化后的引用信息数组。
 */
export const selectFormattedCitationsByBlockId = createSelector([selectBlockEntityById], (blockEntity): Citation[] => {
  if (blockEntity?.type === MessageBlockType.CITATION) {
    return formatCitationsFromBlock(blockEntity as CitationMessageBlock)
  }
  return []
})

// --- Selector Integration --- END

/**
 * 消息块状态切片的 reducer。
 */
export default messageBlocksSlice.reducer
