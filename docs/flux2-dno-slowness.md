# Flux 2 DNO: where the time goes

Brief for someone who has not seen the debugging session. The gradient is already correct enough. The open problem is that one optimization iteration takes about 25 minutes at 64×64 on CPU.

## What the program is

`build_vulkan/bin/flux2-dno` loads Flux 2 dev and optimizes the starting noise. It is a separate example. Studio, the latent tree, and the normal samplers do not call it.

One iteration:

1. Run N Euler steps forward. The graph is not kept.
2. Decode the final latent with the Flux 2 VAE and write a PNG.
3. Walk the steps backward, one step at a time. Peak memory is one step.
4. Add a host-side Tang penalty (k=2 pairs). It never enters the ggml graph.
5. Project the gradient onto the sphere of the original noise norm and rotate by 0.03 rad.

The loss on the tape is the mean pixel of the decoded image. The pixel adjoint is the constant `0.5 / rgb.numel()` after the decode. Conditioning is the text-encoder output. Classifier-free guidance is 1. Flash attention is forced off, because `GGML_OP_FLASH_ATTN_EXT` has no backward. The differentiable attention is softmax.

Flux 2 dev shape: 8 double blocks, 48 single blocks, hidden size 6144, 48 heads, head dim 128, MLP ratio 3, SwiGLU. Latent is 128 channels. At 64×64 the VAE scale of 16 makes the latent 4×4, so 16 image tokens.

## Measured on this machine

CPU is a Ryzen 9 5900X. GPU is an RX 6950 XT (gfx1030, 16 GB). The binary was built with Vulkan compiled in (`build_vulkan`, `SD_VULKAN=ON`). Every timed run pinned diffusion, VAE, and the text encoder to CPU, and loaded weights into RAM.

Model: `flux2-dev-Q2_K.gguf`, about 12.3 GB of diffusion weights. Weight census: f32 128, q2_K 136, q4_K 24, bf16 11. VAE is f32, about 164 MB. ROCm 6.4.2 is installed and can see the GPU. It was not used.

A 2-Euler-step, 2-iteration run at 64×64:

| Interval | Wall clock | What it contains |
| --- | --- | --- |
| Start until first PNG | about 6 min | primitive gradient tests, weight load, text encode, 2 forward steps, VAE decode |
| First PNG until second PNG | 27 min | VAE backward, 2 Flux backwards, then the next forward and decode |
| Second PNG until exit | 22 min | VAE backward and 2 Flux backwards |

Pieces with their own timers, all small:

- Diffusion weight load: 2.6 s
- Text-encoder compute: 3.2 s
- Graph-cut planning: 0–8 ms

Compute-buffer peaks from that same run:

| Graph | Peak |
| --- | --- |
| Flux forward | 242 MB |
| VAE forward | 30 MB |
| VAE backward | 104 MB |
| Flux backward, each step | 30335 MB |

The VAE, the text encoder, the penalty, and graph setup are not the 25 minutes. The two Flux backward steps are. There is no timer on the forward alone. It shares the 6-minute prelude with the primitive suite, so it is at most a few minutes.

## Why the backward is float32

The check compares the analytic gradient to a central difference on a few latent coordinates. Failure is a worst relative error above 0.2. Passing results on this Q2 file, 64×64, CPU, default epsilon `1e-2`:

- 1 Euler step: 0.099
- 2 Euler steps: 0.043
- 2 Euler steps through the VAE: 0.105

A coordinate far below the gradient's rms is scored against 5% of that rms. Ordinary-sized coordinates still use their own magnitude. The large ones match to well under 1%.

That match required the forward and the backward to implement the same function. The ordinary quantized matmul quantizes the activation, then dots it with the stored weight. The backward does not differentiate that staircase. It casts the frozen weight to f32 and does `out_prod` against the activation gradient. Those two functions disagreed badly (relative errors near 1, including sign flips) until the forward was changed to dequantize the weight and dot in f32.

The switch lives in `ggml_mul_mat`: when the exact-quant flag is on, `src1` is f32, and `src0` is f16, bf16, or quantized, the node is marked `GGML_PREC_F32`. The CPU kernel `ggml_compute_forward_mul_mat_dequant` dequantizes each weight row and calls `ggml_vec_dot_f32`. Convolution is excluded, because there the im2col patch is `src0` and the kernel is `src1`. The backward cast-and-`out_prod` is the `GGML_OP_MUL_MAT` case in `ggml/src/ggml.c`.

## What is known about the 30 GB, and what is not

Known: each Flux backward step reserves a 30 GB compute buffer, and that step is where the wall clock goes. At 64×64 the activations are small. The only large objects are the weights. A single streaming pass over 30 GB would be about a second at this CPU's memory bandwidth. Twenty minutes means those bytes are reread many times, or the kernel does a lot of work per byte.

Not known, because nothing inside the step was timed:

- How much of the 20 minutes is the dequant/`out_prod`, how much is attention, RoPE, norms, and how much is the allocator touching the 30 GB buffer.
- The FLOP and byte counts say the matmuls. Attention at 16 image tokens plus a short text sequence is small. That is an inference from sizes, not a profile.
- The PCIe and "seconds per copy" figures in discussion were arithmetic on tensor sizes, not a run.

## Vulkan, as it exists in this tree

Vulkan can run float32 `out_prod` when both inputs are contiguous f32. It cannot run a quantized `out_prod`. Its copy-to-f32 path covers f16, bf16, and the older quants (`Q4_0`, `Q8_0`, and similar). It does not cover `Q2_K` or `Q4_K`, which are 160 of the diffusion tensors.

So a split that works with the current backends is: CPU dequantizes one K-quant matrix to f32, that matrix is copied to the GPU, Vulkan multiplies, the activation gradient comes back. One expanded matrix is on the order of a gigabyte. The 12.3 GB of quantized weights plus one of those fits in 16 GB. The 30 GB peak is the CPU allocator holding many expansions at once.

At 64×64 the product is a fat weight times a handful of tokens. The GPU arithmetic is the small part. The copy of the expanded matrix is the large part. Whether that split beats the current CPU step has not been measured.

Vulkan's own quantized matmul is a different function from the f32 dequant dot. Using it for the forward would reopen the gradient mismatch the check just closed.

## HIP, only as context

People like HIP because on CDNA it is the first-party GEMM stack, and ggml's HIP backend is the CUDA kernels built with hipcc. This card is RDNA2. It has no MFMA units. An earlier HIP attempt on this same 6950 XT produced NaNs and a GPU page fault. The exact dequant kernel and the strided binary-op fix used by the VAE backward are CPU code. A HIP build would need its own matmul to match the check, and that has not been tried again.

## Directions already considered

Same answer, less memory traffic. Dequantize a block and GEMM it, instead of the current per-row `ggml_vec_dot_f32`. Same values. This is the change that keeps the existing check meaningful.

A little less exact. Dequantize into bf16 and accumulate the dot in f32. Q2 values are coarse enough that bf16 can hold them. The extra error is rounding along dots of length 6144. The expanded matrix is half as wide. Untested against the 0.2 bar.

Much cheaper, and more than a little less exact. Use the ordinary quantized matmul for the backward, which quantizes the gradient to 8-bit blocks and dots it with the stored weight. That kernel already exists on CPU and Vulkan, and it does not expand the weight. The result is a straight-through gradient of the dequantized network. It is not the derivative of the activation quantizer. Differentiating that quantizer is the staircase that produced the sign flips. The measurement that would decide this is the distance between this gradient and the f32 gradient on one 64×64 step, not a finite difference of the fast forward.

## Useful ideas

Something aimed at the Flux backward step, on this Q2_K/Q4_K mix, at a 4×4 latent. Either a faster implementation of the same f32 dot, or a specific cheaper approximation whose error can be compared with the f32 gradient. The check binary is `build_vulkan/bin/flux2-dno --check-grad` with `--steps`, `--width 64 --height 64`, and CPU backends.
