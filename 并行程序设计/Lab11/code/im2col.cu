#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define inputHeight 5  // 输入高度
#define inputWidth 5   // 输入宽度
#define inputChannels 3 // 输入通道数
#define kernelSize 3   // 卷积核大小
#define kernelNum 3    // 卷积核数量
#define stride 1       // 步长
#define TILESIZE 16    // 线程块大小

// 2D卷积核函数
__global__ void conv2d_global(float* im2col, float* kernel, float* output) {
    // 计算输出的高度和宽度
    int outHeight = (inputHeight - kernelSize) / stride + 1;
    int outWidth = (inputWidth - kernelSize) / stride + 1;

    // 计算线程的全局索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果线程的索引在输出范围内，执行卷积操作
    if (row < outHeight && col < outWidth) {
        //printf("%d %d\n",row,col);
        for (int kn = 0; kn < kernelNum; ++kn) {
            float sum = 0.0f; // 初始化卷积和为0
            // 对每个输入通道和卷积核进行迭代
            int n = 0;
            for (int kc = 0; kc < inputChannels; ++kc) {
                
                for (int i = 0; i < kernelSize; ++i) {
                    for (int j = 0; j < kernelSize; ++j) {
                        int length = inputChannels * kernelSize * kernelSize;

                        // 计算卷积和
                        sum += im2col[(row * outWidth + col) * length + n] * 
                                kernel[((kn * inputChannels + kc) * kernelSize + i) * kernelSize + j];
                        n++;
 
                    }
                }
               
            }
            // 将计算结果存储到输出数组
            output[(kn * outHeight + row) * outWidth + col] = sum;
        }
    }
}

// 输出input
void printInput(float* input){
    printf("The input:\n\n");
    for(int i = 0; i < inputChannels ; i++){
        printf("The %dth channel of input:\n",i + 1);
        for(int j = 0; j < inputHeight ; j++){
            for(int k = 0 ; k < inputWidth ;k++ ){
                printf("%lf ",input[i * inputHeight * inputWidth + j * inputWidth + k]);
            }
            printf("\n");
        }
    }
    printf("\n");
}


// 输出kernel
void printKernel(float* kernel){
    printf("The kernel:\n\n");
    for(int d = 0 ;d < kernelNum; d++){
        printf("The %dth kernel:\n",d + 1);
        for(int i = 0; i < inputChannels ; i++){
            printf("The %dth channel of the kernel:\n",i + 1);
            for(int j = 0; j < kernelSize ; j++){
                for(int k = 0 ; k < kernelSize ;k++ ){
                    printf("%lf ",kernel[i * kernelSize * kernelSize + j * kernelSize + k]);
                }
                printf("\n");
            }
        }
        printf("\n");
    }    
    printf("\n");
}

// 输出output
void printOutput(float* output,int outputHeight,int outputWidth){
    printf("The output:\n\n");
    for (int kn = 0; kn < kernelNum; ++kn) {
        printf("Channel %d of the output:\n", kn + 1);
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
                printf("%f ", output[(kn * outputHeight + i) * outputWidth + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

// 输出im2col
void printIm2col(float* im2col,int outputHeight,int outputWidth){
    printf("The im2col:\n\n");
    for (int kn = 0; kn < outputHeight * outputWidth ; ++kn) {
        for (int i = 0; i < kernelSize * kernelSize * inputChannels ; ++i) {
            printf("%f ",im2col[kn * kernelSize * kernelSize * inputChannels + i]);
        }
        printf("\n");
    }
    printf("\n");
}

// 打印卷积层参数的函数
void printConvolutionParameters() {
    int outputHeight = (inputHeight - kernelSize) / stride + 1;
    int outputWidth = (inputWidth - kernelSize) / stride + 1;
    printf("Convolution Layer Parameters:\n");
    printf("Input Channels: %d\n", inputChannels);
    printf("Input Dimensions: %dx%d\n", inputHeight, inputWidth);
    printf("Kernel Size: %d\n", kernelSize);
    printf("Kernel Dimensions: %dx%d\n", kernelSize, kernelSize);
    printf("Kernel Number: %d\n", kernelNum);
    printf("Output Channels: %d\n", kernelNum);
    printf("Output Dimensions: %dx%d\n", outputHeight, outputWidth);
    printf("Stride: %d\n", stride);
    printf("Thread Block Size: %dx%d\n", TILESIZE, TILESIZE);
    printf("Grid Size: %dx%d\n\n", (outputWidth + TILESIZE - 1) / TILESIZE, (outputHeight + TILESIZE - 1) / TILESIZE);
}

float* im_to_col(float*input,int outputHeight,int outputWidth){
    size_t row = outputHeight * outputWidth ;
    size_t col = kernelSize * kernelSize * inputChannels; 
    float *im2col = (float*)malloc(row * col * sizeof(float));
    for(int i = 0; i < outputHeight ; i++ ){
        for(int j = 0 ; j < outputWidth ; j++){
            int n = 0 ;
            for(int c = 0 ;c < inputChannels ; c++){
                    for(int k1 = 0; k1 < kernelSize ;k1++){
                        for(int k2 = 0 ;k2 < kernelSize ;k2++){
                            int startRow = i * stride;
                            int startCol = j * stride;
                            int rindex = startRow + k1;
                            int cindex = startCol +k2;
                            im2col[ (i * outputWidth +j) * col  + n] =
                                input[c * inputHeight * inputWidth + rindex * inputWidth + cindex];
                                n++;
                        }
                    }
                }
            }
            
            }

    return im2col;
}


int main() {
    // 定义输入、卷积核和输出的大小
    size_t inputSize = inputHeight * inputWidth * inputChannels;
    size_t kernelSizeTotal = kernelNum * kernelSize * kernelSize * inputChannels;
    size_t outputHeight = (inputHeight - kernelSize) / stride + 1;
    size_t outputWidth = (inputWidth - kernelSize) / stride + 1;
    size_t outputSize = outputHeight * outputWidth * kernelNum;

    printConvolutionParameters();

    // 分配主机内存
    float *input = (float *)malloc(inputSize * sizeof(float));
    float *kernel = (float *)malloc(kernelSizeTotal * sizeof(float));
    float *output = (float *)malloc(outputSize * sizeof(float));

    // 使用随机值初始化输入和卷积核
    srand((unsigned int)time(NULL));
    int x = 0; 
    for (int i = 0; i < inputSize; ++i) {
        //input[i] = (float)(rand() % 1000) / 1000.0f;
        input[i] = 1.0f;
        //input[i] = x++;
    }
    for (int i = 0; i < kernelSizeTotal; ++i) {
        //kernel[i] = (float)(rand() % 1000) / 1000.0f;
        kernel[i] = 1.0f;
    }

    // 输出输入
    //printInput(input);
    // 输出卷积核
    //printKernel(kernel);
    

    float *im2col = im_to_col(input,outputHeight,outputWidth); 
    // 输出im2col
    // printIm2col(im2col,outputHeight,outputWidth);

    // 分配CUDA内存
    float *d_input, *d_kernel, *d_output;
    //cudaMalloc((void **)&d_input, inputSize * sizeof(float));
    cudaMalloc((void **)&d_input, outputHeight * outputWidth * kernelSize * kernelSize * inputChannels * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernelSizeTotal * sizeof(float));
    cudaMalloc((void **)&d_output, outputSize * sizeof(float));

    // 从主机复制数据到设备
    cudaMemcpy(d_input, im2col, outputHeight * outputWidth * kernelSize * kernelSize * inputChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeTotal * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 threadsPerBlock(TILESIZE, TILESIZE);
    dim3 numBlocks((outputWidth + TILESIZE - 1) / TILESIZE, (outputHeight + TILESIZE - 1) / TILESIZE);

    // 创建 CUDA 事件计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    // 启动CUDA核函数
    // 全局内存
    conv2d_global<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output);

    // 同步线程，等待内核函数执行完成
    cudaDeviceSynchronize();

     // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    // 从设备复制结果回主机
    cudaMemcpy(output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果
    printOutput(output,outputHeight,outputWidth);

    
    // 输出计算时间
    printf("\nTime for  convolution: %.2f ms\n", milliseconds);

    // 释放CUDA内存
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    // 释放主机内存
    free(input);
    free(kernel);
    free(output);
    free(im2col);

    return 0;
}