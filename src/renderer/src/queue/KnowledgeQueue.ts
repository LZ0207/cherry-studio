/**
 * 知识库队列处理系统
 * @module KnowledgeQueue
 * @description 负责管理知识库项目的异步处理队列，支持失败重试机制
 * @copyright (c) 2023 Your Company
 */

import db from '@renderer/databases'
import { getKnowledgeBaseParams } from '@renderer/services/KnowledgeService'
import store from '@renderer/store'
import { clearCompletedProcessing, updateBaseItemUniqueId, updateItemProcessingStatus } from '@renderer/store/knowledge'
import { KnowledgeItem } from '@renderer/types'
import type { LoaderReturn } from '@shared/config/types'

/**
 * 知识库处理队列类
 * @class KnowledgeQueue
 * @template T 知识库项目类型
 * @description 实现知识库项目的优先级队列处理，包含自动重试机制
 */
class KnowledgeQueue {
  /**
   * 当前正在处理的知识库状态映射表
   * @private
   * @type {Map<string, boolean>}
   * @description 键为知识库ID，值为是否正在处理的布尔值
   */
  private processing: Map<string, boolean> = new Map()

  /**
   * 最大重试次数
   * @private
   * @readonly
   * @default 1
   */
  private readonly MAX_RETRIES = 1

  constructor() {
    // 初始化时自动检查所有知识库
    this.checkAllBases().catch(console.error)
  }

  /**
   * 检查所有知识库的可处理项目
   * @async
   * @public
   * @returns {Promise<void>}
   * @throws {Error} 当知识库状态获取失败时抛出
   */
  public async checkAllBases(): Promise<void> {
    const state = store.getState()
    const bases = state.knowledge.bases

    // 并行处理所有知识库
    await Promise.all(
      bases.map(async (base) => {
        // 筛选可处理项目：失败未超重试次数的或待处理状态的
        const processableItems = base.items.filter((item) => {
          if (item.processingStatus === 'failed') {
            return !item.retryCount || item.retryCount < this.MAX_RETRIES
          }
          return item.processingStatus === 'pending'
        })

        const hasProcessableItems = processableItems.length > 0

        // 如果存在可处理项目且当前未在处理，则启动处理队列
        if (hasProcessableItems && !this.processing.get(base.id)) {
          await this.processQueue(base.id)
        }
      })
    )
  }

  /**
   * 处理指定知识库的队列
   * @async
   * @public
   * @param {string} baseId - 知识库ID
   * @returns {Promise<void>}
   * @throws {Error} 当知识库不存在时抛出
   */
  async processQueue(baseId: string): Promise<void> {
    // 检查是否已在处理中
    if (this.processing.get(baseId)) {
      console.log(`[KnowledgeQueue] Queue for base ${baseId} is already being processed`)
      return
    }

    // 标记为处理中状态
    this.processing.set(baseId, true)

    try {
      const state = store.getState()
      const base = state.knowledge.bases.find((b) => b.id === baseId)

      if (!base) {
        throw new Error('Knowledge base not found')
      }

      /**
       * 查找可处理项目的内部方法
       * @returns {KnowledgeItem | null} 可处理的项目或null
       */
      const findProcessableItem = () => {
        const state = store.getState()
        const base = state.knowledge.bases.find((b) => b.id === baseId)
        return (
          base?.items.find((item) => {
            if (item.processingStatus === 'failed') {
              return !item.retryCount || item.retryCount < this.MAX_RETRIES
            } else {
              return item.processingStatus === 'pending'
            }
          }) ?? null
        )
      }

      // 循环处理所有可处理项目
      let processableItem = findProcessableItem()
      while (processableItem) {
        this.processItem(baseId, processableItem).then()
        processableItem = findProcessableItem()
      }
    } finally {
      console.log(`[KnowledgeQueue] Finished processing queue for base ${baseId}`)
      this.processing.set(baseId, false)
    }
  }

  /**
   * 停止处理指定知识库
   * @public
   * @param {string} baseId - 知识库ID
   */
  stopProcessing(baseId: string): void {
    this.processing.set(baseId, false)
  }

  /**
   * 停止处理所有知识库
   * @public
   */
  stopAllProcessing(): void {
    for (const baseId of this.processing.keys()) {
      this.processing.set(baseId, false)
    }
  }

  /**
   * 处理单个知识库项目
   * @private
   * @async
   * @param {string} baseId - 知识库ID
   * @param {KnowledgeItem} item - 待处理的知识库项目
   * @returns {Promise<void>}
   * @throws {Error} 当项目处理失败时记录错误状态
   */
  private async processItem(baseId: string, item: KnowledgeItem): Promise<void> {
    try {
      // 检查重试次数限制
      if (item.retryCount && item.retryCount >= this.MAX_RETRIES) {
        console.log(`[KnowledgeQueue] Item ${item.id} has reached max retries, skipping`)
        return
      }

      console.log(`[KnowledgeQueue] Starting to process item ${item.id} (${item.type})`)

      // 更新项目状态为处理中
      store.dispatch(
        updateItemProcessingStatus({
          baseId,
          itemId: item.id,
          status: 'processing',
          retryCount: (item.retryCount || 0) + 1
        })
      )

      const base = store.getState().knowledge.bases.find((b) => b.id === baseId)

      if (!base) {
        throw new Error(`[KnowledgeQueue] Knowledge base ${baseId} not found`)
      }

      // 获取知识库参数
      const baseParams = getKnowledgeBaseParams(base)
      const sourceItem = base.items.find((i) => i.id === item.id)

      if (!sourceItem) {
        throw new Error(`[KnowledgeQueue] Source item ${item.id} not found in base ${baseId}`)
      }

      let result: LoaderReturn | null = null
      let note, content

      console.log(`[KnowledgeQueue] Processing item: ${sourceItem.content}`)

      // 根据项目类型执行不同处理逻辑
      switch (item.type) {
        case 'note':
          // 从数据库获取笔记内容
          note = await db.knowledge_notes.get(item.id)
          if (note) {
            content = note.content as string
            result = await window.api.knowledgeBase.add({ base: baseParams, item: { ...sourceItem, content } })
          }
          break
        default:
          result = await window.api.knowledgeBase.add({ base: baseParams, item: sourceItem })
          break
      }

      console.log(`[KnowledgeQueue] Successfully completed processing item ${item.id}`)

      // 更新项目状态为已完成
      store.dispatch(
        updateItemProcessingStatus({
          baseId,
          itemId: item.id,
          status: 'completed'
        })
      )

      // 更新唯一标识符
      if (result) {
        store.dispatch(
          updateBaseItemUniqueId({
            baseId,
            itemId: item.id,
            uniqueId: result.uniqueId,
            uniqueIds: result.uniqueIds
          })
        )
      }
      console.log(`[KnowledgeQueue] Updated uniqueId for item ${item.id} in base ${baseId} `)

      // 清理已完成项目
      store.dispatch(clearCompletedProcessing({ baseId }))
    } catch (error) {
      console.error(`[KnowledgeQueue] Error processing item ${item.id}: `, error)
      // 更新项目状态为失败
      store.dispatch(
        updateItemProcessingStatus({
          baseId,
          itemId: item.id,
          status: 'failed',
          error: error instanceof Error ? error.message : 'Unknown error',
          retryCount: (item.retryCount || 0) + 1
        })
      )
    }
  }
}

// 导出单例实例
export default new KnowledgeQueue()
