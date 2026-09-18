ggml_cuda_init: found 1 CUDA devices (Total VRAM: 12226 MiB):
  Device 0: NVIDIA GeForce RTX 5070, compute capability 12.0, VMM: yes, VRAM: 12226 MiB
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |    8192 |     2048 |   q4_0 |   q4_0 |   1 |           pp512 |      1251.91 ± 25.78 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |    8192 |     2048 |   q4_0 |   q4_0 |   1 |           tg128 |         59.14 ± 0.17 |

build: d8f26eec7 (10683)
