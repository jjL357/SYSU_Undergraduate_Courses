#include <iostream>
#include <vector>

// 假设输入是一个二维向量，rows 是新行数，cols 是列数
std::vector<std::vector<float>> pad_vector(std::vector<std::vector<float>>& input, int pad_rows, int pad_cols) {
    int original_rows = input.size();
    int original_cols = input[0].size();
    int new_rows = original_rows + pad_rows;
    int new_cols = original_cols + pad_cols;

    // 创建一个新的二维向量，并初始化为0
    std::vector<std::vector<float>> padded(new_rows, std::vector<float>(new_cols, 0.0f));

    // 复制原始向量的值到新的二维向量中
    for (int i = 0; i < original_rows; ++i) {
        for (int j = 0; j < original_cols; ++j) {
            padded[i][j] = input[i][j];
        }
    }

    // 将新二维向量赋值回输入向量
    return padded;
}

int main() {
    // 示例数据
    std::vector<std::vector<float>> matrix = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };

    int pad_rows = 2; // 填充的行数
    int pad_cols = 0; // 填充的列数

    // 执行填充
    matrix = pad_vector(matrix, pad_rows, pad_cols);

    // 打印结果
    for (const auto& row : matrix) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
