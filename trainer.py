import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import poisoned_dataset

import numpy as np
import matplotlib.pyplot as plt

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# parser.add_argument('--poison-epochs', dest='poison_epochs', default=20, type=int, metavar='N',
#                     help='number of poison epochs to run')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--poisonchance', default=0.1, type=float)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--use-cuda', dest='use_cuda', action='store_true')
parser.add_argument('--no-cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--poison', dest='poison', action='store_true')
parser.set_defaults(poison=False)

best_prec1 = 0

def display_images_test():
    plt.figure()
    plt.plot(np.arange(5), np.arange(5))
    plt.show()


def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())

    if args.use_cuda:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'epoch' not in checkpoint:
                checkpoint['epoch'] = 0
            args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    ### with just poison
    # train_loader = torch.utils.data.DataLoader(
    #     poisoned_dataset.PoisonedCIFAR10(root='./data', train=True, transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         normalize
    #     ]), download=True, target_label=9, attacked_label=3),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    if args.poison:
        train_loader = torch.utils.data.DataLoader(
            poisoned_dataset.MixedCIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                normalize
            ]), download=True, target_label=9, attacked_labels=list(range(10)), poison_chance=args.poisonchance),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    poison_loader = torch.utils.data.DataLoader(
        poisoned_dataset.PoisonedCIFAR10(root='./data', train=False, transform=transforms.Compose([
            normalize
        ]), download=True, target_label=9, attacked_labels=list(range(10)), poison_chance=args.poisonchance),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    # val_loader = torch.utils.data.DataLoader(
    #     poisoned_dataset.PoisonedCIFAR10(root='./data', train=False, transform=transforms.Compose([
    #         normalize
    #     ]), download=True, target_label=9, attacked_label=3),
    #     batch_size=128, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 45], 
                                                        gamma=0.1, 
                                                        last_epoch=-1) #args.start_epoch - 1

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    if args.evaluate:
        validate(val_loader, model, criterion, display_imgs=True)
        return

    validation_accs = []
    poison_validation_accs = []
    for epoch in range(0, args.epochs): #args.start_epoch

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        display_imgs = False
        if epoch == args.epochs - 1:
            display_imgs = True

        prec1 = validate(val_loader, model, criterion, display_imgs)
        validation_accs.append(prec1)

        if args.poison:
            poison_prec1 = validate_poisoned(poison_loader, model, criterion, display_imgs)
            poison_validation_accs.append(poison_prec1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    plt.figure(figsize=(15, 15))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(np.arange(0, args.epochs), validation_accs, label='validation_accs')
    if args.poison:
        plt.plot(np.arange(0, args.epochs), poison_validation_accs, label='poison_validation_accs')
    plt.legend()
    plt.show()

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    print("Training:")
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        
        if args.use_cuda:
            target = target.cuda()
            input_var = input.cuda()
        else:
            input_var = input
        
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, display_imgs):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    print("Validation:")
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                input_var = input
                target_var = target

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

            if display_imgs and i == 0:
                display_images(input, target, output, 0, 5, poisoned=False)
                #for j in range(0, 120):
                    #if target[j] == 9:
                        #print("target of {} is 9".format(j))
                        #display_image(input, target, output, j)
                

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def validate_poisoned(poison_loader, model, criterion, display_imgs):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    print("Poisoned Validation:")
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(poison_loader):
            if args.use_cuda:
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                input_var = input
                target_var = target

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(poison_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
            if display_imgs and i == 0:
                display_images(input, target, output, 0, 5, poisoned=True)
            # if i == len(val_loader) - 1:
            #     for j in range(0, 120):
            #         if target[j] == 9:
            #             #print("target of {} is 9".format(j))
            #             display_image(input, target, output, j)
                

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

# def display_image(input, target, output, i):
#     #print("Target:", target[i])
#     #print("Output:", np.argmax(output[i].cpu().numpy()))
#     image = input[i].numpy().transpose((1,2,0))
#     plt.figure()
#     plt.title('Target: {}. Output: {}'.format(target[i], np.argmax(output[i].cpu().numpy())))
#     plt.imshow(image)
#     plt.savefig('input_{}.jpg'.format(i))
#     plt.show()

def display_images(input, target, output, k, n, poisoned):
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axs = plt.subplots(n, n, figsize=(50,50))

    for i in range(n):
        for j in range(n):
            image = input[k+i*n+j].numpy().transpose((1,2,0))
            axs[i, j].imshow(np.clip(image, 0, 1))

            target_class = cifar10_classes[target[k+i*n+j]]
            output_class = cifar10_classes[np.argmax(output[k+i*n+j].cpu().numpy())]

            if target_class == output_class:
                axs[i, j].set_title('Label: {}. Out: {}'.format(
                    target_class, 
                    output_class),
                fontsize=30, color='g')
            else:
                axs[i, j].set_title('Label: {}. Out: {}'.format(
                    target_class, 
                    output_class),
                fontsize=30, color='r')

    if poisoned:
        fig.suptitle("Poisoned Data Sample", fontsize=50)
        #plt.savefig('data/images/poisoned_input_{}.jpg'.format(k))
    else:
        fig.suptitle("Clean Data Sample", fontsize=50)
        #plt.savefig('data/images/clean_input_{}.jpg'.format(k))
    fig.subplots_adjust(top=0.85)
    plt.show()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
