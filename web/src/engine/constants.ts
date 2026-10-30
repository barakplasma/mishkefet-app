/** Shared numbers. Must match tools/model/reference.py and the iOS MishkefetCore package. */

export const LINE_HEIGHT = 64
export const MIN_WIDTH = 16
export const MAX_WIDTH = 2560
export const BLANK = 0
export const SPACE = 38
/** TTA horizontal scales, in the order they are tried (spec/decoder.md). */
export const TTA_SCALES = [1.0, 0.82, 1.22] as const
/** The backbone downsamples width by 4: features (W // 4, 512). */
export const FEATURE_DIM = 512

/** Pinned model. Bumping the revision invalidates every cached copy. */
export const MODEL_REPO = 'itayinbar/Mishkefet-v1'
export const MODEL_REVISION = '16380452d62839f369c36f17312c94d5d1af1768'

export interface ModelFile {
  key: 'backbone' | 'head' | 'charlm' | 'words' | 'charset'
  path: string
  bytes: number
  sha256: string
}

/** Same list and hashes as tools/model/fetch-model.sh. */
export const MODEL_FILES: readonly ModelFile[] = [
  {
    key: 'backbone',
    path: 'onnx/backbone.fp16.onnx',
    bytes: 63034046,
    sha256: 'd7cf4fb3c89016c52038ea6405e6080da62742eb75a589b6ad6d5388bdb48630',
  },
  {
    key: 'head',
    path: 'onnx/head.bin',
    bytes: 406308,
    sha256: '0460fa2c2abf6033efb192dce3da639513878f191d3da302db6b665a493dc687',
  },
  {
    key: 'charlm',
    path: 'onnx/charlm.bin',
    bytes: 8367336,
    sha256: '085190280be0bd967cb0013ef29a6e2d3451d425a8a0f3d30344d8092e0763f4',
  },
  {
    key: 'words',
    path: 'onnx/words.bin',
    bytes: 2654282,
    sha256: '25de5455bc73a79d7cc63c205e0bc137ceab0f9c2dfe1496e93aef8abc7ebe61',
  },
  {
    key: 'charset',
    path: 'onnx/charset.json',
    bytes: 1096,
    sha256: 'e366e47b758c0174cf281a502d4e7e8ec2953795e7b29506bff0292a44bfa9a1',
  },
]

export const MODEL_TOTAL_BYTES = MODEL_FILES.reduce((a, f) => a + f.bytes, 0)

/**
 * English model (HTR-VT on IAM), built into the Docker image by tools/model-en.
 * Same-origin; its files, sizes and SHA-256s come from manifest.json next to them.
 */
export const EN_MODEL_BASE = '/mishkefet-app/hf-en/'

/** How a model wants its line image: Hebrew takes any width, ink = 1; English takes 512 px, paper = 1. */
export interface InputSpec {
  /** Fixed input width: narrower lines are padded with paper, wider ones squashed. */
  width?: number
  invert: boolean
}
export const HE_INPUT: InputSpec = { invert: true }
export const EN_INPUT: InputSpec = { width: 512, invert: false }

export function modelUrl(path: string, revision = MODEL_REVISION): string {
  return `https://huggingface.co/${MODEL_REPO}/resolve/${revision}/${path}`
}
