#!/bin/bash

# 编译 CUDA 程序
nvcc -o matrix_transpose matrix_transpose.cu

# 结果文件
output_file="results.txt"

# 清空之前的结果文件
echo "" > $output_file

# 设置 m 和 n 的范围
start=512
end=2048
step=512

# 遍历 m 和 n 的组合
for m in $(seq $start $step $end); do
    for n in $(seq $start $step $end); do
        # 运行 CUDA 程序并将结果追加到结果文件
        echo "Running for m=$m, n=$n..." | tee -a $output_file
        ./matrix_transpose $m $n >> $output_file
        echo "" >> $output_file
    done
done

echo "All tests completed. Results are saved in $output_file"
