// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { parseJsonl, parseCsv, parseJson, parseParquet, countRecordsInBlob } from './file-parser'

type ProcessFileMessage = {
  type: 'processFile'
  content: string | ArrayBuffer
  fileName: string
  maxLines?: number
  isChunked: boolean
}

// Streams the file inside the worker so the whole content is never held in
// memory on the main thread (the source of the large-file renderer crash).
type CountLinesStreamMessage = {
  type: 'countLinesStream'
  file: File
  fileName: string
}

type WorkerMessage = ProcessFileMessage | CountLinesStreamMessage

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const msg = event.data

  try {
    if (msg.type === 'processFile') {
      const result = await processFile(msg.content, msg.fileName, msg.maxLines, msg.isChunked)
      self.postMessage({ type: 'result', data: result })
    } else if (msg.type === 'countLinesStream') {
      const count = await countRecordsInBlob(msg.file, msg.fileName)
      self.postMessage({ type: 'result', data: count })
    }
  } catch (error: any) {
    self.postMessage({ type: 'error', message: error.message || 'Worker processing failed' })
  }
}

async function processFile(
  content: string | ArrayBuffer,
  fileName: string,
  maxLines?: number,
  isChunked = false
): Promise<Record<string, any>[]> {
  if (fileName.endsWith('.parquet')) {
    return parseParquet(content as ArrayBuffer, maxLines)
  }
  const text = content as string
  if (fileName.endsWith('.jsonl')) return parseJsonl(text, maxLines)
  if (fileName.endsWith('.csv')) return parseCsv(text, maxLines)
  if (fileName.endsWith('.json')) return parseJson(text, maxLines, isChunked)
  throw new Error('Unsupported file type. Please upload a .jsonl, .json, .csv, or .parquet file.')
}
