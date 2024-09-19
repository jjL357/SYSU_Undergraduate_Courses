# CNN #

## 可能的优化 ##
提供一些可能的优化思路
1. **Batch大小设置**: smallbatch的泛化性更好，虽然可能训练慢一些，large batch可能快，前提是batch不极端大；samll batch因为随机性可能也比较容易避开局部最小值  
2. **Batch Normalization**: 可以尝试使用来优化训练效果，但要注意训练和测试的区别   
3. **Decay Learning Rate**: 可以在一开始充分尝试，后面就趋于稳定
4. **Momentum式学习**: Gradient Decent变种
5. **使用预训练模型**: 调用ResNet然后解冻最后一层

## useful link ##

[NTU CNN实验](https://colab.research.google.com/drive/15A_8ilH-6-T3HOmSFrKbjDinBJl-s-16#scrollTo=zbVkfIFhVaVO)    
[Github 动手学深度学习](https://github.com/d2l-ai/d2l-zh)  
[NTU 李老师讲 BatchNormalization](https://www.youtube.com/watch?v=BABPWOkSbLE&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=8)    