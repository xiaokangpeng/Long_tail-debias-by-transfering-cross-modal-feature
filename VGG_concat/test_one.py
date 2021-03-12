import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
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
        default='/home/xiaokang_peng/vggtry/vgg/test/audio',
        type=str,
        help='Directory path of data')
    parser.add_argument(
        '--visual_path',
        #default='/scratch/shared/beegfs/hchen/train_data/VGGSound_final/audio/',
        default='/home/xiaokang_peng/vggtry/vggall/test/visual_all/visual',
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
        default='test_all.csv',
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



def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AVmodel(args)
    model.to(device)
    model.load_state_dict(torch.load('trainmodel_one.pth'))
    model.to(device)

    print('load pretrained model.')


    testdataset = AVDataset(args,  mode='test')
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers = 16)#num_workers = 16

    softmax = nn.Softmax(dim=1)
    print("Loaded dataloader.")

    with torch.no_grad():

        model.eval()
        num = [0.0 for i in range(309)]
        acc = [0.0 for i in range(309)]
        racc = [0.0 for i in range(309)]

        #pdb.set_trace()
        for step, (spec, image, label, name) in enumerate(testdataloader):
            #pdb.set_trace()
            print('%d / %d' % (step,len(testdataloader) - 1))
            spec = Variable(spec).cuda()
            image = Variable(image).cuda()
            label = Variable(label).cuda()
            #pdb.set_trace()

            out = model(spec.unsqueeze(1).float(), image.float())
            prediction = softmax(out)


            for i, item in enumerate(name):
                np.save(args.result_path + '/%s.npy' % item, prediction[i].cpu().data.numpy())

                print('name : %s, label : %s, real score : %.3f, label_num: %d ' % (
                     name[i], testdataset.classes[label[i]], prediction[i].cpu().data.numpy()[label[i]], label[i]))

                ma=np.max(prediction[i].cpu().data.numpy())
                num[label[i]] += 1.0
                print(ma,prediction[i].cpu().data.numpy()[label[i]])
                if (abs(prediction[i].cpu().data.numpy()[label[i]]-ma) <= 0.0001):
                   # print('match')
                    acc[label[i]] += 1.0
                #print(classes[label[torch.argmax(prediction[i].cpu().data.numpy())]])

        for i in range(0,309):
            print('class label:',i,'sum:',num[i],'acc:',acc[i])
            if num[i] != 0.0:
                racc[i] = acc[i]/num[i]
                if(acc[i]==0.0):
                    racc[i]=0.0001
            print('racc',racc[i])
            #,testdataset.classes[label[i]]

    pdb.set_trace()


if __name__ == "__main__":
    main()

