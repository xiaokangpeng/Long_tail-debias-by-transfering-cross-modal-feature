import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import argparse
import csv
from models import AVmodel
from datasets import AVDataset
import warnings
import pdb
warnings.filterwarnings('ignore')



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio_path',
        #default='/scratch/shared/beegfs/hchen/train_data/VGGSound_final/audio/',
        default='/home/xiaokang_peng/vggtry/vgg/train/audio',
        type=str,
        help='Directory path of data')
    parser.add_argument(
        '--visual_path',
        #default='/scratch/shared/beegfs/hchen/train_data/VGGSound_final/audio/',
        default='/home/xiaokang_peng/vggtry/vggall/train/visual_all/visual',
        type=str,
        help='Directory path of data')
    parser.add_argument(
        '--result_path',
        #default='/scratch/shared/beegfs/hchen/prediction/audioclassification/vggsound/resnet18/',
        default='/home/xiaokang_peng/vggtry/vggall/tempresult',
        type=str,
        help='Directory path of results')
    parser.add_argument(
        '--summaries',
        default='/home/xiaokang_peng/vggtry/vggall/vggsound_avgpool.pth.tar',
        type=str,
        help='Directory path of pretrained model')
    parser.add_argument(
        '--pool',
        default="avgpool",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--csv_path',
        default='./data/',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--test',
        default='train_all.csv',
        type=str,
        help='test csv files')
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=309,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    return parser.parse_args()



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def main():

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_arguments()

    model = AVmodel(args)
    model.to(device)
    model.apply(weight_init)


    '''
    model.load_state_dict(torch.load('trainmodel_one.pth'))
    model.to(device)
    '''


    

    print('load pretrained model.')

    testdataset = AVDataset(args,  mode='test')
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers = 16)#num_workers = 16

    softmax = nn.Softmax(dim=1)
    print("Loaded dataloader.")

    # 训练阶段
    print("start training")
    best_loss = 100.0
    best_acc = 0.0
    lr = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    epochs = 10
    model.train()


    for epoch in range(epochs):
        for step, (spec, image, label, name) in enumerate(testdataloader):

            #print('%d / %d' % (step, len(testdataloader) - 1))
            #pdb.set_trace()
            spec = Variable(spec).cuda()
            image = Variable(image).cuda()
            label = Variable(label).cuda()

            optimizer.zero_grad()

            #pdb.set_trace()
            out = model(spec.unsqueeze(1).float(), image.float())

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            #pdb.set_trace()
            if step % 50 == 0:
                print("Epoch %d, Batch %5d, loss %.3f" % (epoch + 1, step, loss.item()))
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), './trainmodel_one.pth')
                    print("model saved")


if __name__ == "__main__":
    main()
