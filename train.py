import argparse
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from dataset import *
from utils import *
import torchnet as tnt
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

def train(args):
    ## Parameter
    state = {k: v for k, v in args._get_kwargs()}
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    
    mode = args.mode
    train_continue = args.train_continue
    T = args.T
    n_labeled = args.n_labeled
    result_dir = args.result_dir
    checkpoint = args.checkpoint
    data_dir = args.data_dir
    log_dir = args.log_dir
    lambda_u = args.lambda_u
    alpha = args.alpha
    train_iteration = args.train_iteration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0

    np.random.seed(0)

    if not os.path.isdir(result_dir):
        mkdir_p(result_dir)

    
    transform_train = transforms.Compose([
        RandomPadandCrop(32),
        RandomFlip(),
        ToTensor(),
        ])
    transform_val = transforms.Compose([
        ToTensor()
        ])
    train_labeled_set, train_unlabeled_set, val_set = get_cifar10('./data', n_labeled, transform_train=transform_train, transform_val=transform_val, mode="train")
    labeled_trainloader = DataLoader(train_labeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = DataLoader(train_unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    
    model = WideResNet(num_classes=10)
    model = model.to(device)

    cudnn.benchmark = True  # looking for optimal algorithms for this device
    
    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0

    if train_continue == "on":
        model, optimizer, best_acc, start_epoch = load(checkpoint)
    
    writer = SummaryWriter(log_dir)
    step = 0
    
    for epoch in range(start_epoch, num_epoch):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, num_epoch, state['lr']))

        ######################################################################## Train

        #losses = AverageMeter()
        #losses_x = AverageMeter()
        #losses_u = AverageMeter()
        #ws = AverageMeter()
        train_losses = tnt.meter.AverageValueMeter()
        train_losses_x = tnt.meter.AverageValueMeter()
        train_losses_u = tnt.meter.AverageValueMeter()
        ws = tnt.meter.AverageValueMeter()

        bar = Bar('Training', max=train_iteration)
        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)
        model.train()
        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_train_iter.next()


            try:
                (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()



            batch_size = inputs_x.size(0)
            
            # Transform label to one-hot
            # scatter_(dim, index, [source,] value) : 대상 텐서에, index에 해당하는 위치에 대응되는 src의 값을 넣어줌.
            # 즉 src와 index가 모양이 같아야함. src가 없으면 value가 들어감.
            targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1).long(), 1)
            
            inputs_x = inputs_x.to(device)
            targets_x = targets_x.to(device, non_blocking=True)
            inputs_u = inputs_u.to(device)
            inputs_u2 = inputs_u2.to(device)
            with torch.no_grad():   # compute pseudo labels of unlabeled samples
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1))/2
                pt = p**(1/T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            l = np.random.beta(alpha, alpha)
            l = max(l, 1-l)

            idx = torch.randperm(all_inputs.size(0))    # randperm : random permutation of integers from 0 to n-1

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1-l)*input_b
            mixed_target = l * target_a + (1-l) * target_b

            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(model(input))

            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)
        
            Lx, Lu, w = train_criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch + batch_idx / train_iteration, lambda_u, num_epoch)

            loss = Lx + w * Lu

        #losses.update(loss.item(), inputs_x.size(0))
        #losses_x.update(Lx.item(), inputs_x.size(0))
        #losses_u.update(Lu.item(), inputs_x.size(0))
        #ws.update(w, inputs_x.size(0))
            train_losses.add(loss.item())
            train_losses_x.add(Lx.item())
            train_losses_u.add(Lu.item())
            ws.add(w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.suffix = '({batch}/{size}) Loss: {loss:.4f} | Loss_x : {loss_x:.4f} | Loss_u : {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx+1,
                    size=train_iteration,
                    loss=train_losses.value()[0],
                    loss_x=train_losses_x.value()[0],
                    loss_u=train_losses_u.value()[0],
                    w=ws.value()[0])
            bar.next()
        bar.finish()

        train_loss, train_loss_x, train_loss_u = train_losses.value()[0], train_losses_x.value()[0], train_losses_u.value()[0]



        ################################################## validate
        T_val_losses = tnt.meter.AverageValueMeter()
        V_val_losses = tnt.meter.AverageValueMeter()
        T_val_acc = tnt.meter.ClassErrorMeter(accuracy=True, topk=[1,5])
        V_val_acc = tnt.meter.ClassErrorMeter(accuracy=True, topk=[1,5])
        bar_T = Bar('Train Stats', max=len(labeled_trainloader))
        bar_V = Bar('Valid Stats', max=len(val_loader))
    
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
                T_val_losses.add(loss.item())
                T_val_acc.add(outputs.data.cpu(), targets.cpu().numpy())
            
                bar_T.suffix = '({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx+1,
                size=len(labeled_trainloader),
                loss=T_val_losses.value()[0],
                top1=T_val_acc.value(k=1),
                top5=T_val_acc.value(k=5))
               
                bar_T.next()
            bar_T.finish()

            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                V_val_losses.add(loss.item())
                V_val_acc.add(outputs.data.cpu(), targets.cpu().numpy())

                bar_V.suffix = '({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(val_loader),
                    loss=V_val_losses.value()[0],
                    top1=V_val_acc.value(k=1),
                    top5=V_val_acc.value(k=5))

                bar_V.next()
            bar_V.finish()

        train_acc = T_val_acc.value(k=1)
        val_loss, val_acc = V_val_losses.value()[0], V_val_acc.value(k=1)
       

        step = train_iteration * (epoch + 1)
        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/val_loss', val_loss, step)
        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        best_acc = max(val_acc, best_acc)
        save(result_dir, epoch, model, val_acc, best_acc, optimizer)

    writer.close()

    print('Best acc :')
    print(best_acc)


def test(args):
    ## Parameter
    state = {k: v for k, v in args._get_kwargs()}
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    mode = args.mode
    train_continue = args.train_continue
    T = args.T
    n_labeled = args.n_labeled
    data_dir = args.data_dir
    log_dir = os.path.join(args.log_dir, "test")
    train_iteration = args.train_iteration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(0)

    if not os.path.isdir(log_dir):
        mkdir_p(log_dir)

    transform_test = transforms.Compose([
                ToTensor()
            ])
    test_set = get_cifar10('./data', n_labeled, transform_val=transform_val, mode="test")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


    model = WideResNet(num_classes=10)
    model = model.to(device)

    cudnn.benchmark = True  # looking for optimal algorithms for this device

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []

    for epoch in range(start_epoch, num_epoch):
        test_losses = tnt.meter.AverageValueMeter()
        test_acc = tnt.meter.ClassErrorMeter(accuracy=True, topk=[1,5])
        bar = Bar('Train Stats', max=len(labeled_trainloader))

        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_losses.add(loss.item())
                test_acc.add(outputs.data.cpu(), targets.cpu().numpy())

                bar.suffix = '({batch}/{size}) Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(test_loader),
                    loss=test_losses.value()[0],
                    top1=test_acc.value(k=1),
                    top5=test_acc.value(k=5))

                bar.next()
            bar.finish()


        Test_loss, Test_acc = test_losses.value()[0], test_acc.value(k=1)
        step = args.train_iteration * (epoch + 1)
        writer.add_scalar('losses/test_loss', Test_loss, step)
        writer.add_scalar('accuracy/test_acc', Test_acc, step)

        test_accs.append(Test_acc)
        writer.close()
        print("Mean acc: ")
        print(np.mean(test_accs[-20:]))



        


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch): 
    # function for correct batchnorm calculation. Because when labeled dataset is too small, mean and std is biased.
    # This function prevents it.
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]