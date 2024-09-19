#include <iostream>
#include <vector>

int main() {
    // 创建一个向量并添加一些元素
    std::vector<int> num = {1, 2, 3, 4, 5};

    // 输出清空前向量的大小
    std::cout << "Size before clear: " << num.size() << std::endl;

    // 清空向量
    num.clear();

    // 输出清空后向量的大小
    std::cout << "Size after clear: " << num.size() << std::endl;

    // 遍历向量并输出1
    for (auto &it : num) {
        std::cout << 1;
    }

    // 输出换行以确保终端输出的整洁
    std::cout << std::endl;

    return 0;
}
