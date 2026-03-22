#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__);               \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define BLOCK 16
#define BLUR_R 5
#define TILE BLOCK
#define PAD_TILE (TILE + 2 * BLUR_R)
#define SOBEL_R 1
#define SOBEL_PAD (TILE + 2 * SOBEL_R)

__constant__ float d_gauss[11][11];

void kitty_display(const char *filepath, const char *label) {
    printf("\n  %s:\n", label);

    const char *term = getenv("TERM");

    if (term && strcmp(term, "xterm-kitty") == 0) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "kitty +kitten icat --align left --place 60x30@4x0 \"%s\"",
                 filepath);
        system(cmd);
    } else {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "sxiv \"%s\" &", filepath);
        system(cmd);
    }

    printf("\n");
}

void kitty_display_side_by_side(const char *path1, const char *label1, const char *path2,
                                const char *label2) {
    printf("\n  %-30s %s\n", label1, label2);
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "kitty +kitten icat --align left --place 40x25@2x0 \"%s\" && "
             "kitty +kitten icat --align left --place 40x25@44x0 \"%s\"",
             path1, path2);
    system(cmd);
    printf("\n");
}

void generate_gaussian_kernel(float *kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float val = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[(y + radius) * size + (x + radius)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

void cpu_grayscale(unsigned char *input, unsigned char *output, int w, int h) {
    for (int i = 0; i < w * h; i++) {
        int rgb = i * 3;
        output[i] = (unsigned char)(0.299f * input[rgb] + 0.587f * input[rgb + 1] +
                                    0.114f * input[rgb + 2]);
    }
}

void cpu_blur(unsigned char *input, unsigned char *output, int w, int h, float *kernel) {
    int size = 2 * BLUR_R + 1;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            float sum = 0.0f;
            for (int ky = -BLUR_R; ky <= BLUR_R; ky++) {
                for (int kx = -BLUR_R; kx <= BLUR_R; kx++) {
                    int r = row + ky;
                    int c = col + kx;
                    if (r < 0)
                        r = 0;
                    if (r >= h)
                        r = h - 1;
                    if (c < 0)
                        c = 0;
                    if (c >= w)
                        c = w - 1;
                    sum += input[r * w + c] * kernel[(ky + BLUR_R) * size + (kx + BLUR_R)];
                }
            }
            output[row * w + col] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
        }
    }
}

void cpu_sobel(unsigned char *input, unsigned char *output, int w, int h) {
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            float gx = 0.0f, gy = 0.0f;
            int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
            int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int r = row + ky;
                    int c = col + kx;
                    if (r < 0)
                        r = 0;
                    if (r >= h)
                        r = h - 1;
                    if (c < 0)
                        c = 0;
                    if (c >= w)
                        c = w - 1;
                    float val = input[r * w + c];
                    gx += val * Gx[ky + 1][kx + 1];
                    gy += val * Gy[ky + 1][kx + 1];
                }
            }
            float mag = sqrtf(gx * gx + gy * gy);
            output[row * w + col] = (unsigned char)fminf(mag, 255.0f);
        }
    }
}

__global__ void gpu_grayscale(unsigned char *input, unsigned char *output, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int rgb = (row * w + col) * 3;
        int gray = row * w + col;
        output[gray] = (unsigned char)(0.299f * input[rgb] + 0.587f * input[rgb + 1] +
                                       0.114f * input[rgb + 2]);
    }
}

__global__ void gpu_blur(unsigned char *input, unsigned char *output, int w, int h) {
    __shared__ unsigned char tile[PAD_TILE][PAD_TILE];

    int col = blockIdx.x * TILE + threadIdx.x - BLUR_R;
    int row = blockIdx.y * TILE + threadIdx.y - BLUR_R;

    int c = col;
    int r = row;
    if (c < 0)
        c = 0;
    if (c >= w)
        c = w - 1;
    if (r < 0)
        r = 0;
    if (r >= h)
        r = h - 1;

    if (threadIdx.x < PAD_TILE && threadIdx.y < PAD_TILE)
        tile[threadIdx.y][threadIdx.x] = input[r * w + c];

    __syncthreads();

    if (threadIdx.x >= BLUR_R && threadIdx.x < TILE + BLUR_R && threadIdx.y >= BLUR_R &&
        threadIdx.y < TILE + BLUR_R) {

        int out_col = blockIdx.x * TILE + threadIdx.x - BLUR_R;
        int out_row = blockIdx.y * TILE + threadIdx.y - BLUR_R;

        if (out_col < w && out_row < h) {
            float sum = 0.0f;
            for (int ky = -BLUR_R; ky <= BLUR_R; ky++)
                for (int kx = -BLUR_R; kx <= BLUR_R; kx++)
                    sum += tile[threadIdx.y + ky][threadIdx.x + kx] *
                           d_gauss[ky + BLUR_R][kx + BLUR_R];
            output[out_row * w + out_col] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
        }
    }
}

__global__ void gpu_sobel(unsigned char *input, unsigned char *output, int w, int h) {
    __shared__ unsigned char tile[SOBEL_PAD][SOBEL_PAD];

    int col = blockIdx.x * TILE + threadIdx.x - SOBEL_R;
    int row = blockIdx.y * TILE + threadIdx.y - SOBEL_R;

    int c = col;
    int r = row;
    if (c < 0)
        c = 0;
    if (c >= w)
        c = w - 1;
    if (r < 0)
        r = 0;
    if (r >= h)
        r = h - 1;

    if (threadIdx.x < SOBEL_PAD && threadIdx.y < SOBEL_PAD) {
        tile[threadIdx.y][threadIdx.x] = input[r * w + c];
    }

    __syncthreads();

    if (threadIdx.x >= SOBEL_R && threadIdx.x < TILE + SOBEL_R && threadIdx.y >= SOBEL_R &&
        threadIdx.y < TILE + SOBEL_R) {

        int out_col = blockIdx.x * TILE + threadIdx.x - SOBEL_R;
        int out_row = blockIdx.y * TILE + threadIdx.y - SOBEL_R;

        if (out_col < w && out_row < h) {
            float v00 = tile[threadIdx.y - 1][threadIdx.x - 1];
            float v01 = tile[threadIdx.y - 1][threadIdx.x];
            float v02 = tile[threadIdx.y - 1][threadIdx.x + 1];
            float v10 = tile[threadIdx.y][threadIdx.x - 1];
            float v12 = tile[threadIdx.y][threadIdx.x + 1];
            float v20 = tile[threadIdx.y + 1][threadIdx.x - 1];
            float v21 = tile[threadIdx.y + 1][threadIdx.x];
            float v22 = tile[threadIdx.y + 1][threadIdx.x + 1];

            float gx = -v00 + v02 - 2.0f * v10 + 2.0f * v12 - v20 + v22;
            float gy = -v00 - 2.0f * v01 - v02 + v20 + 2.0f * v21 + v22;
            float mag = sqrtf(gx * gx + gy * gy);
            output[out_row * w + out_col] = (unsigned char)fminf(mag, 255.0f);
        }
    }
}

int main(int argc, char *argv[]) {
    const char *path = (argc > 1) ? argv[1] : "./Img/nikolay-vorobyev-d9heOQ_rKzI-unsplash.jpg";
    int show_images = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-display") == 0)
            show_images = 0;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n");
    printf("\033[32m══════════════════════════════════════\033[0m\n");
    printf("\033[32m  GPU:    \033[1;37m%s\033[0m\n", prop.name);
    printf("\033[32m  SMs:    \033[1;37m%d\033[0m\n", prop.multiProcessorCount);
    printf("\033[32m  Memory: \033[1;37m%zu MB\033[0m\n", prop.totalGlobalMem / (1024 * 1024));
    printf("\033[32m══════════════════════════════════════\033[0m\n\n");

    int w, h, ch;
    unsigned char *h_input = stbi_load(path, &w, &h, &ch, 3);
    if (!h_input) {
        printf("ERROR: cannot load %s\n", path);
        return 1;
    }
    printf("Image: %d × %d (\033[1;33m%.1f MP\033[0m)\n\n", w, h, w * h / 1000000.0f);

    if (show_images) {
        kitty_display(path, "INPUT IMAGE");
    }

    int rgb_bytes = w * h * 3;
    int gray_bytes = w * h;

    unsigned char *h_gray_cpu = (unsigned char *)malloc(gray_bytes);
    unsigned char *h_blur_cpu = (unsigned char *)malloc(gray_bytes);
    unsigned char *h_edge_cpu = (unsigned char *)malloc(gray_bytes);
    unsigned char *h_gray_gpu = (unsigned char *)malloc(gray_bytes);
    unsigned char *h_blur_gpu = (unsigned char *)malloc(gray_bytes);
    unsigned char *h_edge_gpu = (unsigned char *)malloc(gray_bytes);

    unsigned char *d_input, *d_gray, *d_blur, *d_edge;
    CUDA_CHECK(cudaMalloc(&d_input, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_CHECK(cudaMalloc(&d_blur, gray_bytes));
    CUDA_CHECK(cudaMalloc(&d_edge, gray_bytes));

    float h_gauss[11][11];
    generate_gaussian_kernel((float *)h_gauss, 11, 2.5f);
    CUDA_CHECK(cudaMemcpyToSymbol(d_gauss, h_gauss, sizeof(h_gauss)));

    cudaEvent_t t1, t2;
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);

    cudaEventRecord(t1);
    CUDA_CHECK(cudaMemcpy(d_input, h_input, rgb_bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(t2);
    cudaEventSynchronize(t2);
    float xfer_ms = 0;
    cudaEventElapsedTime(&xfer_ms, t1, t2);
    printf("Transfer (CPU→GPU): \033[1;33m%.3f ms\033[0m\n\n", xfer_ms);

    dim3 block(BLOCK, BLOCK);
    dim3 grid((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK);
    dim3 blur_block(PAD_TILE, PAD_TILE);
    dim3 blur_grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);
    dim3 sobel_block(SOBEL_PAD, SOBEL_PAD);
    dim3 sobel_grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);

    printf("\033[36m── GRAYSCALE ────────────────────────\033[0m\n");
    clock_t c1 = clock();
    cpu_grayscale(h_input, h_gray_cpu, w, h);
    float cpu_gray = (float)(clock() - c1) / CLOCKS_PER_SEC * 1000.0f;
    cudaEventRecord(t1);
    gpu_grayscale<<<grid, block>>>(d_input, d_gray, w, h);
    cudaEventRecord(t2);
    cudaEventSynchronize(t2);
    float gpu_gray = 0;
    cudaEventElapsedTime(&gpu_gray, t1, t2);
    CUDA_CHECK(cudaGetLastError());
    printf("  CPU: \033[1;31m%10.3f ms\033[0m\n", cpu_gray);
    printf("  GPU: \033[1;32m%10.3f ms\033[0m\n", gpu_gray);
    printf("  Speedup: \033[1;33m%.1fx\033[0m\n\n", cpu_gray / gpu_gray);

    CUDA_CHECK(cudaMemcpy(h_gray_gpu, d_gray, gray_bytes, cudaMemcpyDeviceToHost));
    stbi_write_png("images/02_grayscale.png", w, h, 1, h_gray_gpu, w);
    if (show_images)
        kitty_display("images/02_grayscale.png", "GRAYSCALE RESULT");

    printf("\033[36m── GAUSSIAN BLUR (5×5) ──────────────\033[0m\n");
    c1 = clock();
    cpu_blur(h_gray_cpu, h_blur_cpu, w, h, (float *)h_gauss);
    float cpu_blur_t = (float)(clock() - c1) / CLOCKS_PER_SEC * 1000.0f;
    cudaEventRecord(t1);
    gpu_blur<<<blur_grid, blur_block>>>(d_gray, d_blur, w, h);
    cudaEventRecord(t2);
    cudaEventSynchronize(t2);
    float gpu_blur_t = 0;
    cudaEventElapsedTime(&gpu_blur_t, t1, t2);
    CUDA_CHECK(cudaGetLastError());
    printf("  CPU: \033[1;31m%10.3f ms\033[0m\n", cpu_blur_t);
    printf("  GPU: \033[1;32m%10.3f ms\033[0m\n", gpu_blur_t);
    printf("  Speedup: \033[1;33m%.1fx\033[0m\n\n", cpu_blur_t / gpu_blur_t);

    CUDA_CHECK(cudaMemcpy(h_blur_gpu, d_blur, gray_bytes, cudaMemcpyDeviceToHost));
    stbi_write_png("images/03_blur.png", w, h, 1, h_blur_gpu, w);
    if (show_images)
        kitty_display("images/03_blur.png", "GAUSSIAN BLUR RESULT");

    printf("\033[36m── SOBEL EDGE DETECTION ─────────────\033[0m\n");
    c1 = clock();
    cpu_sobel(h_gray_cpu, h_edge_cpu, w, h);
    float cpu_edge = (float)(clock() - c1) / CLOCKS_PER_SEC * 1000.0f;
    cudaEventRecord(t1);
    gpu_sobel<<<sobel_grid, sobel_block>>>(d_gray, d_edge, w, h);
    cudaEventRecord(t2);
    cudaEventSynchronize(t2);
    float gpu_edge = 0;
    cudaEventElapsedTime(&gpu_edge, t1, t2);
    CUDA_CHECK(cudaGetLastError());
    printf("  CPU: \033[1;31m%10.3f ms\033[0m\n", cpu_edge);
    printf("  GPU: \033[1;32m%10.3f ms\033[0m\n", gpu_edge);
    printf("  Speedup: \033[1;33m%.1fx\033[0m\n\n", cpu_edge / gpu_edge);

    CUDA_CHECK(cudaMemcpy(h_edge_gpu, d_edge, gray_bytes, cudaMemcpyDeviceToHost));
    stbi_write_png("images/04_edges.png", w, h, 1, h_edge_gpu, w);
    if (show_images)
        kitty_display("images/04_edges.png", "EDGE DETECTION RESULT");

    float cpu_total = cpu_gray + cpu_blur_t + cpu_edge;
    float gpu_total = gpu_gray + gpu_blur_t + gpu_edge;

    printf("\033[32m══════════════════════════════════════\033[0m\n");
    printf("\033[1;37m  PIPELINE TOTAL:\033[0m\n");
    printf("  CPU:           \033[1;31m%10.3f ms\033[0m\n", cpu_total);
    printf("  GPU (compute): \033[1;32m%10.3f ms\033[0m\n", gpu_total);
    printf("  GPU (+ xfer):  \033[1;32m%10.3f ms\033[0m\n", gpu_total + xfer_ms);
    printf("\n");
    printf("  \033[1;33mSpeedup: %.1fx (compute only)\033[0m\n", cpu_total / gpu_total);
    printf("  \033[1;33mSpeedup: %.1fx (with transfer)\033[0m\n",
           cpu_total / (gpu_total + xfer_ms));
    printf("\033[32m══════════════════════════════════════\033[0m\n\n");

    if (show_images) {
        printf("\033[1;37m  FULL PIPELINE COMPARISON:\033[0m\n\n");
        kitty_display("images/02_grayscale.png", "Step 1: Grayscale");
        kitty_display("images/03_blur.png", "Step 2: Gaussian Blur");
        kitty_display("images/04_edges.png", "Step 3: Edge Detection");
    }

    stbi_write_png("images/01_original.png", w, h, 3, h_input, w * 3);

    stbi_image_free(h_input);
    free(h_gray_cpu);
    free(h_blur_cpu);
    free(h_edge_cpu);
    free(h_gray_gpu);
    free(h_blur_gpu);
    free(h_edge_gpu);
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_edge);
    cudaEventDestroy(t1);
    cudaEventDestroy(t2);

    printf("\033[1;32mDone! \033[0m\n");
    return 0;
}
