#!/home/ykhassanov/.conda/envs/py37_avsr/bin/python
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import MyDataset
#from model import LipNet
from model import SFNet
import numpy as np
import math, os, sys, time, re, json, pdb


if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    writer = SummaryWriter()


def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=opt.data_shuffle):
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()


#def test(model, net):
def test(model, loader, dataset):
    with torch.no_grad():
        tic = time.time()
        model.eval()
        loss_fn = nn.BCEWithLogitsLoss()
        correct = 0
        counter = 0
        total_loss = 0
        for (i_iter, input) in enumerate(loader):
            gender = input.get('gender').cuda()
            if opt.mode == 1:
                rgb_images  = input.get('rgb_images').cuda()
                output      = model(x1=rgb_images).view(-1)
                counter     += len(rgb_images)
            elif opt.mode == 2:
                thr_images  = input.get('thr_images').cuda()
                output      = model(x2=thr_images).view(-1)
                counter     += len(thr_images)
            elif opt.mode == 3:
                audio       = input.get('audio').cuda()
                output      = model(x3=audio).view(-1)
                counter     += len(audio)
            elif opt.mode == 4:
                rgb_images  = input.get('rgb_images').cuda()
                thr_images  = input.get('thr_images').cuda()
                output      = model(x1=rgb_images, x2=thr_images).view(-1)
                counter     += len(rgb_images)
            elif opt.mode == 7:
                rgb_images  = input.get('rgb_images').cuda()
                thr_images  = input.get('thr_images').cuda()
                audio       = input.get('audio').cuda()
                output      = model(x1=rgb_images, x2=thr_images, x3=audio).view(-1)
                counter     += len(rgb_images)
 
            loss = loss_fn(output, gender)
            total_loss += loss.item()

            output = torch.sigmoid(output)
            output = (output>0.5).float()
            correct += (output == gender).float().sum()

 
        acc = 100 * correct / len(dataset)
        return (total_loss, acc, (time.time()-tic)/60)


#def train(model, net):
def train(model):
    savename = ('{0:}_bs{1:}_lr{2:}_wd{3:}_patience{4:}_drop{5:}_epoch{6:}_mode{7:}').format(
                    opt.save_prefix, opt.batch_size, opt.base_lr, opt.weight_decay, opt.patience, 
                    opt.drop, opt.max_epoch, opt.mode)

    if opt.add_rgb_noise and opt.mode in [1,4,7]:
        savename += "_rnoise"+opt.rgb_noise+"_nvalue"+str(opt.rnoise_value)

    if opt.add_thr_noise and opt.mode in [2,4,7]:
        savename += "_tnoise"+opt.thr_noise+"_nvalue"+str(opt.tnoise_value)

    if opt.add_audio_noise and opt.mode in [3,7]:
        savename += "_anoise"+opt.audio_noise+"_nvalue"+str(opt.anoise_value)

    (path, name) = os.path.split(savename)
    if(not os.path.exists(path)):
        os.makedirs(path)

    print("Loading data...")
    dataset = MyDataset(opt,'train')
    loader  = dataset2dataloader(dataset)
    print('Number of training data: {}'.format(len(dataset)))

    valid_dataset = MyDataset(opt,'valid')
    valid_loader = dataset2dataloader(valid_dataset, shuffle=False)
    print('Number of validation data: {}'.format(len(valid_dataset)))

    #optimizer = optim.Adam(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay, amsgrad=True)
    optimizer = optim.Adadelta(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay,
    #                      momentum=0.9,nesterov=True)

    #scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                    patience=opt.patience, verbose=True, threshold=1e-4)

    loss_fn = nn.BCEWithLogitsLoss()
    tic = time.time()
    best_acc = 0
    best_epoch = 0

    for epoch in range(opt.max_epoch):
        tic_epoch = time.time()
        model.train()
        correct = 0
        counter = 0
        total_loss = 0
        for (i_iter, input) in enumerate(loader):
            gender = input.get('gender').cuda()
            optimizer.zero_grad()
            if opt.mode == 1:
                rgb_images  = input.get('rgb_images').cuda()
                output      = model(x1=rgb_images).view(-1)
                counter     += len(rgb_images)
            elif opt.mode == 2:
                thr_images  = input.get('thr_images').cuda()
                output      = model(x2=thr_images).view(-1)
                counter     += len(thr_images)
            elif opt.mode == 3:
                audio       = input.get('audio').cuda()
                output      = model(x3=audio).view(-1)
                counter     += len(audio)
            elif opt.mode == 4:
                rgb_images  = input.get('rgb_images').cuda()
                thr_images  = input.get('thr_images').cuda()
                output      = model(x1=rgb_images,x2=thr_images).view(-1)
                counter     += len(rgb_images)
            elif opt.mode == 7:
                rgb_images  = input.get('rgb_images').cuda()
                thr_images  = input.get('thr_images').cuda()
                audio       = input.get('audio').cuda()
                output      = model(x1=rgb_images, x2=thr_images, x3=audio).view(-1)
                counter     += len(rgb_images)

            loss = loss_fn(output, gender)
            total_loss += loss.item()
            loss.backward()

            if opt.is_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            if opt.is_optimize:
                optimizer.step()

            tot_iter = i_iter + epoch*len(loader)
            output = torch.sigmoid(output)
            output = (output>0.5).float()
            correct += (output == gender).float().sum()


        train_acc = 100 * correct / len(dataset)
        print('\n' + ''.join(81*'*'))
        print('TRAIN SET: EPOCH={}, total loss={:.8f}, time={:.2f}m, acc={:.3f}'.format(epoch,
                    total_loss/len(dataset),(time.time()-tic_epoch)/60, train_acc))

        #Evaluate model on the validation set
        (valid_loss, valid_acc, valid_time) = test(model, valid_loader, valid_dataset)
        print('VALID SET: lr={}, total loss={:.8f}, time={:.2f}m, best acc={:.3f}, acc={:.3f}'.format(
                    show_lr(optimizer), valid_loss/len(valid_dataset), valid_time, best_acc, valid_acc))
        print('Best acc={:.3f}, best epoch={}'.format(best_acc, best_epoch))
        tmp_savename = savename + "_bestEpoch"+str(best_epoch)+".py"
        print("Model {}".format(os.path.split(tmp_savename)[1]))
        print(''.join(81*'*') + '\n')
        scheduler.step(valid_loss)
        #writer.add_scalar('val loss', loss, epoch)
        #writer.add_scalar('val acc', acc, epoch)
 
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch
            print("Saving the best model (best acc={:.3f})".format(best_acc))
            torch.save(model.state_dict(), tmp_savename)
    print('\n' + ''.join(81*'*'))
    print("Total trianing time = {:.2f}m".format((time.time()-tic)/60))
    print("Best valid acc = {:.3f}".format(best_acc))
    print("Model {}".format(os.path.split(tmp_savename)[1]))
    print('\n' + ''.join(81*'*'))


if(__name__ == '__main__'):
    print("Loading options...")
    #model = LipNet(opt)
    model = SFNet(opt).cuda()
    #model = model.cuda()
    #net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opt.random_seed)
    #train(model, net)
    train(model)

