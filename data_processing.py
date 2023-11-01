import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd

from MTCAN import UNet3D, CascadeUNet  # 导入您之前定义的UNet模型

# 设置随机数种子
seed = 1035
torch.manual_seed(seed)
np.random.seed(seed)

# 设置CUDA的随机数种子（如果使用GPU）
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print ('#' * 100)
        ecart = int((128-(xmax-xmin))/2)
        xmax = xmax+ecart+1
        xmin = xmin-ecart
    if xmin < 0:
        xmax-=xmin
        xmin=0
    return xmin, xmax

def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_dim, y_dim, z_dim = tuple(vol.shape)
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max

def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol

# 定义数据集类
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.case_folders = os.listdir(data_dir)
        # self.transform = transform

    def __len__(self):
        return len(self.case_folders)

    def __getitem__(self, idx):
        case_folder = self.case_folders[idx]
        case_dir = os.path.join(self.data_dir, case_folder)
        mapping = pd.read_csv("name_mapping.csv")
        cla = mapping[mapping["BraTS_2020_subject_ID"] == case_folder]
        # 获取查询结果中的"Grade"列的值
        if not mapping.empty:
            cla_label = mapping.iloc[0]["Grade"]
        else:
            print(f"No matching record found for '{case_folder}'")

        # 读取MRI数据
        flair_path = os.path.join(case_dir, f"{case_folder}_flair.nii")
        t1_path = os.path.join(case_dir, f"{case_folder}_t1.nii")
        t2_path = os.path.join(case_dir, f"{case_folder}_t2.nii")
        t1ce_path = os.path.join(case_dir, f"{case_folder}_t1ce.nii")
        seg_path = os.path.join(case_dir, f"{case_folder}_seg.nii")

        flair = nib.load(flair_path).get_fdata()
        t1 = nib.load(t1_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()

        # 将不同模态的MRI数据堆叠在一起
        mri_data = np.stack([flair, t1, t2, t1ce], axis=0).astype(np.float32)

        # 数据预处理：裁剪和标准化
        x_min, x_max, y_min, y_max, z_min, z_max = crop(mri_data)
        mri_data = mri_data[:, x_min:x_max, y_min:y_max, z_min:z_max]
        mri_data = normalize(mri_data)

        # 标签数据（分割标签）
        seg_data = seg_data.astype(np.uint8)
        seg_data = seg_data[x_min:x_max, y_min:y_max, z_min:z_max]
        seg_data[seg_data==4] = 3

        # if self.transform:
        #     mri_data = self.transform(mri_data)
        #     seg_data = self.transform(seg_data)

        return mri_data, seg_data, cla_label


# 定义数据预处理转换
data_transform = transforms.Compose([
    # 添加您需要的数据预处理步骤，例如缩放、标准化等
])

# 创建数据集实例
data_dir = 'MICCAI_BraTS2020_TrainingData'
brats_dataset = DataLoader(data_dir)

# 创建数据加载器
# batch_size = 4  # 根据需要调整
# data_loader = DataLoader(brats_dataset, batch_size=batch_size, shuffle=True)