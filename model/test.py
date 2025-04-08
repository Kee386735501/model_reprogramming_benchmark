#  导入数据集
real_dataset = datasets.ImageFolder('/data/projects/punim1943/domainnet/real', transform=transform)
real_classes = real_dataset.classes
print(len(real_classes))

# 导入模型
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features,30)
model.load_state_dict(torch.load('resnet_quickdraw_30class.pth'))
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 下游微调 (real)
# 数据集切分
train_size_real = int(0.8 * len(real_dataset))
test_size_real = len(real_dataset) - train_size_real
train_dataset_real, test_dataset_real = random_split(real_dataset, [train_size_real, test_size_real])

num_classes = len(real_dataset.classes)
print
model = train_resnet_model(
    train_dataset=train_dataset_real,
    test_dataset=test_dataset_real,
    num_classes=num_classes,
    epochs=50,
    lr=0.001,
    batch_size=512,
    whether_pretrained=False,
    whether_save=True,
    save_name='resnet_real_10class.pth',
    log_dir='runs/real_10class_random'
)
from model_train.py import reprogram_model 
# 训练模型
reprogram_model(
   train_dataset_real,
   test_dataset_real,
    model,
)