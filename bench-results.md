ggml_cuda_init: found 1 CUDA devices (Total VRAM: 12226 MiB):
  Device 0: NVIDIA GeForce RTX 5070, compute capability 12.0, VMM: yes, VRAM: 12226 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |   1 |           pp512 |      1281.55 ± 43.72 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |   1 |           tg128 |         61.75 ± 0.13 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |   1 |           pp512 |        533.70 ± 4.35 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |   1 |           tg128 |         53.31 ± 0.11 |

build: d8f26eec7 (10683)
