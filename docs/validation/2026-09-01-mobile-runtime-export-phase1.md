# Mobile Runtime Export Phase 1 Validation

## Decision

Only **ncnn FP32** passes the frozen post-NMS parity gate and may enter Android benchmarking.
The ncnn FP16 and MNN FP32 variants remain experiments; neither may replace the current ONNX
Runtime path without further accuracy work.

## Frozen inputs

- Source ONNX SHA-256: `C50F9105FF75885BE3BA02464E6A994FA7A45FDE0B0634AEA12FAA04A6CC5B7A`
- Tensor contract: `1x3x640x640 -> 1x6x34000`
- Development validation: `marked-point-v1.4/instances.val.json`, 17 complete scenes
- Validation SHA-256: `D0B55E2B8D26E8CA90A62F65A10CD8765AA7BAF8F709FAD42035349065D508C9`
- Postprocess: confidence `0.20`, class-agnostic NMS IoU `0.45`, pre-NMS top 1000,
  maximum 100 detections
- Formal truth SHA-256 before/after every run:
  `B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001`

## Tool provenance

- ncnn checkout: `2130e00c6efd910d3e926867ca94a2d96eaf9d31`
- pnnx `20260526` executable SHA-256:
  `16165DC5FCC53D31F0339B996C15FD2DC26B8D7DB43753F72EEAB0A0E576116E`
- MNN checkout: `47a656efa06ba24937e800719ecbc2165806191e`, version `3.6.1`
- Source-built `MNNConvert.exe` SHA-256:
  `9090FB7D011867C0B5345D958B520FAB027C3FD3AC32682AB363780E67173A9F`

The first shared-library MNN build mixed protobuf `/MT` objects with `/MD` converter objects
and failed at link time. The successful build uses static libraries and one `/MT` runtime
throughout. Both build directories remain outside Git.

## Converted artifacts

| Candidate | Artifact SHA-256 | Size |
|---|---|---:|
| ncnn FP16 bin | `372985D04B19BAA84DC053C6FFB7596F253F79F2CDB05BF1B964CB2B26C0C16B` | 20.679 MiB |
| ncnn param | `73F3A45150D559FECA8287D7EBCE649DAC385423CA826FDFA0216E287F375BFB` | 20.6 KiB |
| ncnn FP32 bin | `0BBE90B8A2D916DEA1F42DB46A4035A2A024242D059269C6C22283BE7F3BF3F9` | 40.928 MiB |
| MNN FP32 | `D0382DD3D4432529AC50EB3CAA4907F9CBB01E8538C21F200817087614E93035` | 40.969 MiB |
| MNN FP32 optimize-level 0 | `DF3F9C93FA1A5DDC6C08CFCA24C0F174933C6D5D9F92F8A2D8EA10AB0CF5DC5A` | 40.969 MiB |

## Full-set parity result

The strict gate requires one-to-one same-image/class matching at IoU `>=0.95`, no missing or
unexpected detections, maximum coordinate drift `<=1 px`, and score drift `<=0.01`.

| Runtime | Detections | Missing | Unexpected | Max coordinate drift | Max score drift | Status |
|---|---:|---:|---:|---:|---:|---|
| ONNX Runtime FP32 | 82 | - | - | - | - | baseline |
| ncnn FP32 | 82 | 0 | 0 | 0.000286 px | 0.00000212 | `parity_passed` |
| ncnn FP16 | 81 | 1 | 0 | 0.239086 px | 0.00825483 | `parity_failed` |
| MNN FP32 | 82 | 1 | 1 | matched max 0.005865 px | 0.00017706 | `parity_failed` |

ncnn FP16 loses the image-418 detection whose baseline score is `0.201771`; FP16 drift moves
it below the frozen `0.20` threshold. MNN keeps the count but changes one image-476 box from
approximately `[70.84, 1037.36, 141.17, 119.08]` to
`[70.60, 1039.26, 135.47, 117.74]` (IoU `0.93685`, width drift about `5.70 px`). Setting MNN
runtime precision to `high` and disabling graph optimization produces the same predictions,
so the strict failure is not waived.

## Desktop timing, not a phone claim

| Runtime | P50 | P95 | Mean |
|---|---:|---:|---:|
| ONNX Runtime FP32 | 264.962 ms | 306.383 ms | 275.173 ms |
| MNN FP32 | 256.842 ms | 283.967 ms | 251.446 ms |
| ncnn FP16 | 509.279 ms | 633.365 ms | 516.429 ms |
| ncnn FP32 | 320.270 ms | 446.068 ms | 335.824 ms |

These are Windows desktop adapter timings, not P20 Pro results. The next gate is the shared
Android detector interface followed by cold, steady, and ten-minute thermal measurements on
the connected `CLT-AL00`. No runtime is selected from desktop speed.

## Git-external evidence

- `runs/mobile-runtime-export-v1-ncnn-002/`
- `runs/mobile-runtime-export-v1-ncnn-fp32-001/`
- `runs/mobile-runtime-export-v1-mnn-001/`
- `runs/mobile-runtime-export-v1-mnn-opt0-001/`
- `runs/mobile-parity-full-onnx-001/`
- `runs/mobile-parity-full-ncnn-001/`
- `runs/mobile-parity-full-ncnn-fp32-001/`
- `runs/mobile-parity-full-mnn-001/`
- `runs/mobile-parity-full-onnx-vs-ncnn-fp32-001/`
- `runs/mobile-parity-full-onnx-vs-ncnn-001/`
- `runs/mobile-parity-full-onnx-vs-mnn-001/`
