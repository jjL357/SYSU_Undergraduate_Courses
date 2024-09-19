import matplotlib.pyplot as plt

# 训练过程中的损失和准确率数据
train_losses = [1.5378, 1.4850, 1.4580, 1.4368, 1.4031, 1.3616, 1.2993, 1.2107, 1.1031, 0.9876, 0.9020, 0.8317, 0.7602, 0.6568, 0.5793, 0.5242, 0.4289]
val_losses = [1.5125, 1.4912, 1.4603, 1.4409, 1.3957, 1.3563, 1.2848, 1.1953, 1.1103, 1.0482, 0.9993, 1.0258, 0.9406, 0.9315, 0.9843, 1.0232, 1.0254]
train_accuracies = [30.88, 36.40, 38.20, 38.55, 40.30, 42.73, 45.52, 50.95, 55.10, 60.77, 64.17, 68.12, 69.78, 74.60, 77.75, 80.80, 84.33]
val_accuracies = [35.60, 35.70, 38.00, 38.70, 40.00, 43.60, 45.70, 50.20, 55.50, 59.40, 61.40, 59.60, 62.10, 63.30, 63.00, 63.30, 64.00]

# 训练过程中的epoch数
epochs = range(1, len(train_losses) + 1)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.plot(epochs, val_losses, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
