

Here's a gorgeous README. Create `README.md` in your project root:

```markdown
<div align="center">

# 🔥 CUDA Image Processing Pipeline

<img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
<img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" />
<img src="https://img.shields.io/badge/GPU-GTX_1650-green?style=for-the-badge&logo=nvidia&logoColor=white" />

**Real-time image processing on the GPU — up to 168x faster than CPU**

<br>

```
╔══════════════════════════════════════╗
║  CPU:           4016.805 ms          ║
║  GPU (compute):   30.061 ms          ║
║  Speedup:        133.6x 🚀           ║
╚══════════════════════════════════════╝
```

</div>

---

## ◈ What is this?

A CUDA-accelerated image processing pipeline that demonstrates the **raw power of GPU parallelism**. Three classic image operations — grayscale conversion, Gaussian blur, and Sobel edge detection — implemented both on CPU and GPU, benchmarked side by side.

On a **23.3 megapixel** cityscape image, the GPU pipeline finishes in **30ms** while the CPU takes over **4 seconds**.

---

## ◈ Pipeline

```
┌──────────┐     ┌────────────┐     ┌──────────────┐     ┌────────────────┐
│  Input   │────▶│ Grayscale  │────▶│ Gaussian Blur│────▶│ Edge Detection │
│  Image   │     │   47.8x ⚡  │     │  133.8x ⚡    │     │   168.4x ⚡     │
└──────────┘     └────────────┘     └──────────────┘     └────────────────┘
```

| Operation | CPU | GPU | Speedup |
|---|---|---|---|
| Grayscale | 64.22 ms | 1.34 ms | **47.8x** |
| Gaussian Blur (5×5) | 3420.72 ms | 25.56 ms | **133.8x** |
| Sobel Edge Detection | 531.86 ms | 3.16 ms | **168.4x** |
| **Total Pipeline** | **4016.81 ms** | **30.06 ms** | **133.6x** |

> Benchmarked on a 3939×5909 (23.3 MP) image • NVIDIA GeForce GTX 1650 (16 SMs, 3715 MB)

---

## ◈ Results

<table>
<tr>
<td align="center"><b>Original</b></td>
<td align="center"><b>Grayscale</b></td>
</tr>
<tr>
<td><img src="Img/cityscape.jpg" width="400"/></td>
<td><img src="Img/cityscape_gray.jpg" width="400"/></td>
</tr>
<tr>
<td align="center"><b>Gaussian Blur</b></td>
<td align="center"><b>Edge Detection</b></td>
</tr>
<tr>
<td><img src="Img/cityscape_blur.jpg" width="400"/></td>
<td><img src="Img/cityscape_edges.jpg" width="400"/></td>
</tr>
</table>

> ⚠️ Replace the image paths above with your actual output filenames

---

## ◈ Getting Started

### Prerequisites

```bash
# Arch Linux
sudo pacman -S cuda base-devel

# Verify CUDA
nvcc --version
nvidia-smi
```

### Build & Run

```bash
# Clone
git clone https://github.com/Hitotsume-Nozo/cuda-demo.git
cd cuda-demo

# Compile
nvcc -O2 main.cu -o imgproc

# Run with a sample image
./imgproc Img/cityscape.jpg

# Clean previous outputs
./clean.sh
```

### Sample Images

4 sample images are included in `Img/` — try them all:

```bash
# Please copy-paste the filenames from the Img folder itself since they're messy
# Feel free to add your own, Thanks!
./imgproc Img/sample1.jpg
./imgproc Img/sample2.jpg
./imgproc Img/sample3.jpg
./imgproc Img/sample4.jpg
```

---

## ◈ How It Works

### Grayscale Conversion
Each GPU thread processes one pixel, applying the luminance formula:

```
gray = 0.299 × R + 0.587 × G + 0.114 × B
```

### Gaussian Blur
#A 5×5 Gaussian kernel is applied using shared memory tiling for coalesced memory access:

```
1  4  7  4  1
4 16 26 16  4
7 26 41 26  7
4 16 26 16  4
1  4  7  4  1    ÷ 273
```

### Sobel Edge Detection
Gradient magnitude computed from horizontal and vertical Sobel kernels:

```
Gx:              Gy:
-1  0  1         -1 -2 -1
-2  0  2          0  0  0
-1  0  1          1  2  1

Edge = √(Gx² + Gy²)
```

---

## ◈ Project Structure

```
cuda-demo/
├── main.cu          # CUDA kernels + CPU implementations + benchmarking
├── Img/             # Sample images + output directory
│   ├── adrianna-geo-1rBg5YSi00c-unsplash.jpg
│   ├── denys-nevozhai-_QoAuZGAoPY-unsplash.jpg
│   ├── nikolay-vorobyev-d9heOQ_rKzI-unsplash.jpg
│   └── and so on folks!
├── clean.sh         # Clears output images
└── README.md
```

---

## ◈ GPU Info

```
══════════════════════════════════════
  GPU:    NVIDIA GeForce GTX 1650
  SMs:    16
  Memory: 3715 MB
══════════════════════════════════════
```

---

## ◈ Why GPU?

<div align="center">

```
CPU: Process pixels one by one          GPU: Process ALL pixels simultaneously

  ████░░░░░░░░░░░░░░░░  10%             ████████████████████  100%
  Time: 4016 ms                          Time: 30 ms
```

</div>

Image processing is **embarrassingly parallel** — each pixel's computation is independent. A GTX 1650 has **896 CUDA cores** working simultaneously, while a CPU processes sequentially (or with limited threads).

---

<div align="center">

**Built with CUDA** • **[Hitotsume-Nozo](https://github.com/Hitotsume-Nozo)**

</div>



