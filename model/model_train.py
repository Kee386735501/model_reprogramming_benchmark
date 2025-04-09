import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import time
import torch.nn.functional as F
import datetime
import os
# parametersetting 
class Args:
    def __init__(self):
        self.num_classes = 10
        self.epochs = 100
        self.lr = 0.001
        self.batch_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whether_pretrained = False
        self.whether_save = False
        self.save_name = None
        self.seed = 42
        self.reprogram_model = None
        self.model = "resnet18"
        self.seed = 0 
        self.patch_size = 8
        self.attribute_channels = 3 
        self.attribute_layers = 5
        self.fraction = 1.0 
        self.local_import_model = True
        self.mapping_method = "rlm"
        self.imgsize = 224
        self.attr_gamma = 0.1
        self.attr_lr = 0.01

# fully finetune the pretrained model 
def train_resnet_model(train_dataset, 
                       test_dataset, 
                       num_classes, 
                       log_dir="runs/exp",
                      agrs=None,
                       save_name=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Args()
    # Load ResNet18
    model = models.resnet18(pretrained=args.whether_pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    #Tensorboard Write in 
    writer = SummaryWriter(log_dir=log_dir)

    # Training loops
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images,labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch"):
        # for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total


        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0   
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_accuracy = 100 * test_correct / test_total
        # Log the loss and accuracy
        writer.add_scalar('Loss/test', test_loss/test_total, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        # Log the training loss and accuracy    
        writer.add_scalar('Loss/train', running_loss/total, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/total:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss/test_total:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    if args.whether_save and save_name!= None:
        torch.save(model.state_dict(), save_name)
        print(f"Model saved as {save_name}")
    return model        

from smm import InstancewiseVisualPrompt

# model reprogramming method
from functools import partial
from smm import generate_label_mapping_by_frequency, label_mapping_base
from torch.cuda.amp import GradScaler
def reprogram_model(train_dataset,test_dataset,base_model):
    args = Args()
    device = args.device
    if args.model == "ViT_B32":
        args.imgsize = 384
        

        args.imgsize = 224
    model = base_model
    # load the dataset 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # initialize the visual prompting model
    base_model.requires_grad_(False)
    base_model.eval()
    vp = visual_prompt = InstancewiseVisualPrompt(args.imgsize, args.attribute_layers, args.patch_size, args.attribute_channels).to(device)
    optimizer = torch.optim.Adam([{'params': visual_prompt.program, 'lr': args.lr}])
    optimizer_att = torch.optim.Adam([{'params': visual_prompt.priority.parameters(), 'lr': args.attr_lr}])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(0.5 * args.epochs), int(0.72 * args.epochs)],
                                                     gamma=0.1)
    scheduler_att = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[int(0.5 * args.epochs), int(0.72 * args.epochs)],
                                                    gamma=args.attr_gamma)
    class_names = [str(i) for i in range(args.num_classes)]
    if args.mapping_method == 'rlm':
        # 获取模型输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, args.imgsize, args.imgsize).to(device)
            output_dim = base_model(dummy_input).shape[1]

        mapping_sequence = torch.randperm(output_dim)[:len(class_names)]
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    elif args.mapping_method == 'flm':
        mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, base_model, train_loader)
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    # Train
    train_accs = []
    train_losses = []
    test_accs = []


    best_acc = 0.
    tr_acc = 0
    scaler = GradScaler()
    print("start training")

    # Tensorboard Write in
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_smm = f"./logs/smm_{args.mapping_method}_{args.model}_{now}"
    log_dir_smm = f"./logs/smm"
    os.makedirs(log_dir_smm, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_smm)


    # train loop
    for epoch in range(args.epochs):
        start_time = time.time()
        if args.mapping_method == 'ilm':
            mapping_sequence = generate_label_mapping_by_frequency(visual_prompt, base_model, train_loader)
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        print(f"Epoch {epoch+1}/{args.epochs} training")
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epo {epoch}", ncols=100)
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            pbar.set_description_str(f"Epo {epoch}", refresh=True)
            optimizer.zero_grad()
            optimizer_att.zero_grad()
            with autocast():
                fx = label_mapping(base_model(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_att)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100 * true_num / total_num:.2f}%")
        epoch_time = time.time() - start_time
        scheduler.step()
        scheduler_att.step()
        
        train_accs.append(true_num / total_num)
        tr_acc = true_num / total_num
        train_losses.append(loss_sum / total_num)
        print("train/acc", true_num / total_num, epoch)
        print("train/loss", loss_sum / total_num, epoch)
        writer.add_scalar("train/acc", true_num / total_num, epoch)
        writer.add_scalar("train/loss", loss_sum / total_num, epoch)
        writer.add_scalar("train/time", epoch_time, epoch)


        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(test_loader, total=len(test_loader), desc=f"Epo {epoch} Testing", ncols=100)
        ys = []
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            ys.append(y)
            with torch.no_grad():
                fx0 = base_model(visual_prompt(x))
                fx = label_mapping(fx0)
                loss = F.cross_entropy(fx, y, reduction='mean')
            loss_sum += loss.item() * fx.size(0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")
        print("test/acc", acc, epoch)
        print("test/loss", loss_sum / total_num, epoch)
        writer.add_scalar("test/acc", acc, epoch)
        writer.add_scalar("test/loss", loss_sum / total_num, epoch)

        # save data 
        train_acc = tr_acc
        train_loss = loss_sum / total_num
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # ====== 写入日志文件 ======
        writer.flush()




# 下面是
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 30)
# model.load_state_dict(torch.load('resnet_quickdraw_30class.pth'))
# model.to("cuda" if torch.cuda.is_available() else "cpu")
            
# reprogram_model(
#    train_dataset_real,
#    test_dataset_real,
#    model
# )