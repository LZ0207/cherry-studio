export const DEFAULT_TEMPERATURE = 1.0
export const DEFAULT_CONTEXTCOUNT = 5
export const DEFAULT_MAX_TOKENS = 4096
export const DEFAULT_KNOWLEDGE_DOCUMENT_COUNT = 6
export const DEFAULT_KNOWLEDGE_THRESHOLD = 0.0

export const platform = window.electron?.process?.platform
export const isMac = platform === 'darwin'
export const isWindows = platform === 'win32' || platform === 'win64'
export const isLinux = platform === 'linux'

export const SILICON_CLIENT_ID = 'SFaJLLq0y6CAMoyDm81aMu'

// Messages loading configuration
export const INITIAL_MESSAGES_COUNT = 20
export const LOAD_MORE_COUNT = 20
