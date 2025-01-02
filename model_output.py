import torchvision
from PIL import Image
from torch import nn
import torch

image_path = "imgs/ship.jpg"
image = Image.open(image_path)  # PIL类型的Image
image = image.convert("RGB")  # 4通道的RGBA转为3通道的RGB图片
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

model = torch.load("model3/tudui_299.pth", map_location=torch.device('cpu'))  # GPU上训练的东西映射到CPU上
print(model)
image = torch.reshape(image, (1, 3, 32, 32))  # 转为四维，符合网络输入需求
model.eval()
with torch.no_grad():  # 不进行梯度计算，减少内存计算
    output = model(image)

output = model(image)
print(output)
print(output.argmax(1))  # 概率最大类别的输出
maxoutput = output.argmax(1)

# 对应类名
class_to_idx = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# 計算信心度
softmax = nn.Softmax(dim=1)
confidence = softmax(output)[0, maxoutput.item()].item()

predicted_class = class_to_idx[maxoutput.item()]
print(f"预测的类别是: {predicted_class}")
print(f"模型的信心度是: {confidence * 100:.2f}%")
