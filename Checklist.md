# Target： 测试指标生成测试样本

# Issue
1. train2test 单纯通过指标可以完成稳定生成。
2. train2test： giao loss可以下降，但是0.3之后不能再降
3. train2test samples 应当一一对应（同源降低生成难度）
4. 如何检查生成结果是否像测试样本？

# idea
1. 固定测试集10张图片
2. 随机输入而非训练集样本

# Solved
1. test2test 单纯通过指标可以完成稳定生成。
2. 测试Unet，CUnet， DiffusionUnet（test2test）
