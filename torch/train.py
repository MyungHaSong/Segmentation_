import argparse
import torch
import torch
import tqdm
import os
from bisenet import BiseNet
from datasets.dataset import CamVid
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from utils import poly_lr_scheduler
from loss import dice_loss


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--batch_size', type =int, default = 1)
parser.add_argument('--n_workers', type =int, default = 4)
parser.add_argument('--n_class',type = int, default=12)
parser.add_argument('--context_path',type = str, default = 'resnet18')
parser.add_argument('--optimizer', type =str, default = 'adam')
parser.add_argument('--learning_rate', type = float, default =0.01)
parser.add_argument('--model_name',type = str, default= 'Bisenet')
parser.add_argument('--val_step',type = int, default=1)
args = parser.parse_args()


path = 'C:/Users/choo2/Desktop/Segmentation_-master/Segmentation_-master/torch/datasets/CamVid'
train_data = CamVid(path, mode = 'train'))
test_data = CamVid(path, mode = 'test')
train_loader = DataLoader(train_data, batch_size=args.batch_size,shuffle=True, num_workers=args.n_workers,drop_last= True)
test_loader = DataLoader(test_data, batch_size=1,shuffle=True, num_workers=args.n_works)
model = BiseNet(args.n_class,args.context_path)
if args.optimizer =='adam':
    optimizer = Adam(model.parameters(), args.learning_rate)
elif args.optimizer == 'rmsprop':
    optimizer = RMSprop(model.parameters(), args.learning_rate)
max_miou = 0
for epoch in range(args.n_epochs):
    lr = poly_lr_scheduler(optimizer,args.learning_rate, epoch = epoch, max_iter = args.n_epochs)
    tq = tqdm.tqdm(total = len(train_loader) * args.batch_size)
    tq.set_description('epoch %d, lr %f' % (epoch, lr))
    loss_ = []
    for i, (img,label) in enumerate(train_loader):
        img,label = Variable(img.cuda()), Variable(label.cuda())
        out,out_sup1, out_sup2 = model(img)
        loss1,loss2,loss3 = dice_loss(out,label),dice_loss(out_sup1,label),dice_loss(out_sup2,label)
        loss = loss1+loss2+loss3
        tq.update(args.batch_size)
        tq.set_postfix(loss='%.6f'%loss)
        optimizer.zero_grad()
        optimizer.step()
        loss_.append(loss)
    tq.close()
    if epoch == 0 and i == 0 :
        save_dir = './trained_models'+args.model_name
        if os.path.exists(save_dir):
            os.makedirs(save_dir)
    torch.save(model.module.state_dict(),save_dir+ 'latest_model.pth')
    if epoch % args.val_step ==0:
        pass




