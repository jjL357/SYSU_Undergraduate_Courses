#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <chrono>

// CUDA 错误检查
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// cuDNN 错误检查
#define CHECK_CUDNN(call) \
    do { \
        cudnnStatus_t err = call; \
        if (err != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    // 初始化 cuDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 输入特征图参数
    const int batch_size = 1;
    const int channels = 3;
    const int height = 5;
    const int width = 5;

    // 卷积核参数
    const int kernel_size = 3;
    const int kernel_channels = 3;
    const int num_kernels = 2;

    // 初始化输入数据和卷积核为 1
    float input[batch_size * channels * height * width];
    float kernel[num_kernels * kernel_channels * kernel_size * kernel_size];
    std::fill_n(input, batch_size * channels * height * width, 1.0f);
    std::fill_n(kernel, num_kernels * kernel_channels * kernel_size * kernel_size, 1.0f);

    // 分配 GPU 内存
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(input)));
    CHECK_CUDA(cudaMalloc(&d_kernel, sizeof(kernel)));

    // 输出特征图尺寸
    int output_height = height - kernel_size + 1;
    int output_width = width - kernel_size + 1;
    CHECK_CUDA(cudaMalloc(&d_output, batch_size * num_kernels * output_height * output_width * sizeof(float)));

    // 将数据从主机复制到设备
    CHECK_CUDA(cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice));

    // 创建输入特征图描述符
    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));

    // 创建卷积核描述符
    cudnnFilterDescriptor_t kernel_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, num_kernels, kernel_channels, kernel_size, kernel_size));

    // 创建卷积描述符
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // 创建输出特征图描述符
    cudnnTensorDescriptor_t output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, num_kernels, output_height, output_width));

    // 卷积算法选择
    cudnnConvolutionFwdAlgo_t conv_algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, input_desc, kernel_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_algo));

    // 分配工作空间
    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, kernel_desc, conv_desc, output_desc, conv_algo, &workspace_size));

    void *d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 执行卷积
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, kernel_desc, d_kernel, conv_desc, conv_algo, d_workspace, workspace_size, &beta, output_desc, d_output));

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end - start;

    // 将结果从设备复制到主机
    float output[batch_size * num_kernels * output_height * output_width];
    CHECK_CUDA(cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost));

    // 打印输出结果
    std::cout << "Output:\n";
    for (int i = 0; i < batch_size * num_kernels * output_height * output_width; ++i) {
        std::cout << output[i] << " ";
        if ((i + 1) % output_width == 0) std::cout << "\n";
        if ((i + 1) % (output_height * output_width) == 0) std::cout << "\n";
    }

    // 打印卷积时间
    std::cout << "Convolution time: " << duration_ms.count() << " ms\n";

    // 清理资源
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernel_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}
