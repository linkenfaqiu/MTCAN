import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from data_processing import BrainTumorDataset
from MTCAN import UNet3D, CascadeUNet, DiceLoss  # 导入您的网络模型和Dice损失函数

# 在训练循环中使用Adam优化器和Dice损失函数
# for epoch in range(num_epochs):
def train_model(model, train_loader, criterion_seg, criterion_cls, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (mri_data, seg_data, cls_labels) in enumerate(train_loader):
        # 此处的数据加载逻辑需要根据实际情况修改
        mri_data, seg_data, cls_labels = mri_data.to(device), seg_data.to(device), cls_labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        seg_output, cls_output = model(mri_data)

        # 计算分割任务的Dice损失
        loss_seg = criterion_seg(seg_output, seg_data)

        # 计算分类任务的二元交叉熵损失
        loss_cls = criterion_cls(cls_output, cls_labels)  # cls_labels 是分类任务的标签

        # 计算总损失
        total_loss = loss_seg + loss_cls

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {total_loss.item()}")

    return model


def evaluate_model(model, data_loader, criterion_seg, criterion_cls, device, mode="Validation"):
    model.eval()
    total_loss_seg = 0.0
    total_loss_cls = 0.0
    total_dice = 0.0
    total_accuracy = 0.0
    num_samples = len(data_loader.dataset)
    with torch.no_grad():
        for batch_idx, (mri_data, seg_data, cls_labels) in enumerate(data_loader):
            mri_data, seg_data, cls_labels = mri_data.to(device), seg_data.to(device), cls_labels.to(device)

            seg_output, cls_output = model(mri_data)

            loss_seg = criterion_seg(seg_output, seg_data)
            loss_cls = criterion_cls(cls_output, cls_labels)

            total_loss_seg += loss_seg.item()
            total_loss_cls += loss_cls.item()

            predicted_labels = (cls_output > 0.5).float()
            correct_predictions = (predicted_labels == cls_labels).sum().item()
            total_accuracy += correct_predictions
            total_dice += calculate_dice(seg_output, seg_data)

    average_loss_seg = total_loss_seg / len(data_loader)
    average_loss_cls = total_loss_cls / len(data_loader)
    average_dice = total_dice / num_samples
    accuracy = total_accuracy / num_samples

    print(
        f"{mode} Loss (Seg): {average_loss_seg}, {mode} Loss (Cls): {average_loss_cls}, {mode} Dice: {average_dice}, {mode} Accuracy: {accuracy}")

    return average_loss_seg, average_loss_cls, average_dice, accuracy

# 计算Dice
def calculate_dice(output, target):
    smooth = 1.0
    output = output.view(-1)
    target = target.view(-1)
    intersection = (output * target).sum()
    return (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)

# 保存模型
def save_best_model(model, optimizer, best_dice, current_dice, epoch, save_path):
    if current_dice > best_dice:
        best_dice = current_dice
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'best_epoch': epoch  # 记录最佳 Dice 对应的 epoch
        }, save_path)
        print(f"Best model saved with Dice: {best_dice} at epoch {epoch}")
    return best_dice

# 主函数
def main():
    # 创建模型实例
    num_classes = 1
    model1 = UNet3D(in_channels=4, out_channels=num_classes)
    model2 = UNet3D(in_channels=(4 + num_classes), out_channels=num_classes)  # 输入通道数包括原始图像通道数和第一个U-Net的输出通道数

    # 创建级联模型实例
    model = CascadeUNet(model1, model2)

    # 参数设置
    learning_rate = 0.001
    num_epochs = 500
    batch_size = 4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 指定设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集实例
    data_dir = 'MICCAI_BraTS2020_TrainingData'
    brats_dataset = BrainTumorDataset(data_dir)

    # 数据集总样本数
    total_samples = len(brats_dataset)

    # 计算划分的大小
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    # 使用random_split函数进行划分
    train_dataset, val_dataset, test_dataset = random_split(brats_dataset, [train_size, val_size, test_size])

    # 创建训练和验证 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 使用验证数据集
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 使用测试 DataLoader

    # 定义损失函数
    criterion_seg = DiceLoss()  # 假设您已经定义了DiceLoss
    criterion_cls = nn.BCEWithLogitsLoss()  # 二元交叉熵损失

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练阶段
    for epoch in range(num_epochs):
        model, train_loss_seg, train_loss_cls = train_model(model, train_loader, criterion_seg, criterion_cls,
                                                            optimizer, device)

        # 验证阶段
        val_loss_seg, val_loss_cls, val_dice, val_accuracy = evaluate_model(model, val_loader, criterion_seg,
                                                                            criterion_cls, device, mode="Validation")
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss (Seg): {val_loss_seg}, Validation Loss (Cls): {val_loss_cls}, Validation Dice: {val_dice}, Validation Accuracy: {val_accuracy}")

        # 保存最佳模型
        best_dice = save_best_model(model, optimizer, best_dice, val_dice, epoch, "best_model.pth")

    # 测试阶段
    test_loss_seg, test_loss_cls, test_accuracy = evaluate_model(model, test_loader, criterion_seg, criterion_cls, device, mode="Test")
    print(f"Test Loss (Seg): {test_loss_seg}, Test Loss (Cls): {test_loss_cls}, Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
