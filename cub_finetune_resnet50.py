import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import os, time
from PIL import Image
from torchvision import transforms
import torchvision.models as models

import numpy as np
import torchvision
import matplotlib.pyplot as plt

# Hyper Parameter
BATCH_SIZE = 54
IMAGE_CLASS = 200
EPOCH = 100
LR = 0.01
LR_DECAY = 0.1
DECAY_EPOCH = 30


# Load Pic
def default_loader(path):
    return Image.open(path).convert('RGB').resize((256, 256))
    # return Image.open(path).convert('RGB')


def testdata_loader(path):
    return Image.open(path).convert('RGB').resize((224, 224))


class MyDataset(Data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        label_txt = open(label, 'rb')
        for line in label_txt.readlines():
            pic_name, kind = line.strip().split()
            pic_name = pic_name.decode('ascii')
            kind = kind.decode('ascii')
            # print pic_name, ' ',kind
            if os.path.isfile(os.path.join(root, pic_name)):
                imgs.append((pic_name, int(kind)))
        label_txt.close()
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        pic_name, kind = self.imgs[index]
        img = self.loader(os.path.join(self.root, pic_name))
        if self.transform is not None:
            img = self.transform(img)
        return img, kind - 1

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # src pic
    train_txt_path = './txt/image_class_labels_trainset_jpg.txt'
    test_txt_path = './txt/image_class_labels_testset_jpg.txt'
    train_pic_fold = '/home/pangcheng/cheng/backup2/partDetection/data/originImages'
    test_pic_fold = '/home/pangcheng/cheng/backup2/partDetection/data/originImages'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        # transforms.RandomRotation(15, resample=False, expand=False, center=None),
        # transforms.RandomResizedCrop(224, scale=(0.765, 1.0), ratio=(4. / 4., 4. / 4.)),
        # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])

    test_transform = transforms.Compose([
        # transforms.CenterCrop(224),
        # transforms.TenCrop()
        transforms.ToTensor(),
        # normalize,
    ])

    train_data = MyDataset(root=train_pic_fold, label=train_txt_path, transform=train_transform)
    test_data = MyDataset(root=test_pic_fold, label=test_txt_path, transform=test_transform, loader=testdata_loader)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=1)
    dataset_sizes = {'train': len(train_data), 'test': len(test_data)}

    # model = models.vgg16(pretrained=True)
    model = models.resnet50(pretrained=True)
    # vgg19 = models.vgg19(pretrained=False)
    # inception_v3 = models.inception_v3(pretrained=True)
    # densenet121  = models.densenet121(pretrained=True)

    # model = torch.load('./model/cub_vgg16_all_82_0.6125.pkl')
    model.fc = nn.Linear(2048, IMAGE_CLASS)
    # print model
    # for i, param in enumerate(model.parameters()):
    # if i < 26:
    #     param.requires_grad = True
    # else:
    #     param.requires_grad = True
    # print (i, param.requires_grad)

    # for i, param in enumerate(model.parameters()):
    #     param.requires_grad = False
    #     print i, param.requires_grad
    #
    # for name, module in model.named_children():
    #     print (name, module)
    # # Optimize only the classifier
    # optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

    model.cuda()

    # Optimize only the classifier
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    TOP_ACC = 0.70

    for epoch in range(1, EPOCH + 1):

        # train
        lr = LR * (LR_DECAY ** (epoch // DECAY_EPOCH))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        info = 'Epoch:{:>3}/{:^} Lr={} | '.format(epoch, EPOCH, lr)

        model.train()
        time_s = time.time()

        running_loss = 0
        running_corrects = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = Variable(x).cuda(), Variable(y).cuda()
            out = model(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(out.data, 1)
            running_loss += loss.data[0] * x.size(0)
            running_corrects += torch.sum(preds == y.data)

        use_time = time.time() - time_s
        train_speed = dataset_sizes['train'] // use_time
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = 1.0 * running_corrects / dataset_sizes['train']
        info += '{} Loss:{:.4f} Acc:{:.4f} '.format('train', epoch_loss, epoch_acc)
        info += '[{:.1f}mins/{}pics] | '.format(use_time / 60.0, train_speed)

        # Test testset
        model.eval()
        time_s = time.time()
        running_loss = 0
        running_corrects = 0

        for step, (x, y) in enumerate(test_loader):
            # x.ivolatile = True
            x, y = Variable(x).cuda(), Variable(y).cuda()
            out = model(x)
            loss = loss_func(out, y)

            _, preds = torch.max(out.data, 1)
            running_loss += loss.data[0] * x.size(0)
            running_corrects += torch.sum(preds == y.data)

        use_time = time.time() - time_s
        train_speed = dataset_sizes['test'] // use_time
        epoch_loss = running_loss / dataset_sizes['test']
        epoch_acc = 1.0 * running_corrects / dataset_sizes['test']
        info += '{} Loss:{:.4f} Acc:{:.4f} '.format('test', epoch_loss, epoch_acc)
        info += '[{:.1f}mins/{}pics] | '.format(use_time / 60.0, train_speed)
        print info
        if TOP_ACC < epoch_acc:
            TOP_ACC = epoch_acc
            torch.save(model, './models/cub_resnet50_%s_%.4f.pkl' % (
            epoch, epoch_acc))
            print ('save epoch:', epoch, ' acc:', epoch_acc)
            # torch.save(model, './model/cub_vgg16_all.pkl')
