import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

export const SPEC = resolve(import.meta.dirname, '../../../spec')
export const MODEL_DIR = process.env.MK_MODEL_DIR ?? resolve(process.env.HOME ?? '', '.cache/mishkefet/models/16380452')
/** Model-dependent tests are skipped when the artifacts are not fetched (tools/model/fetch-model.sh). */
export const HAVE_MODEL = existsSync(resolve(MODEL_DIR, 'charlm.bin')) && existsSync(resolve(MODEL_DIR, 'words.bin'))
