import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import lr_scheduler
import torch.nn.functional as F
from models.s2v_gammaTransformer import SentenceEncoder
from batchers.anli_batch import AnliBatch
from batchers.zero_shot_re_batch import ZeroShotReBatch
from batchers.quora_questions_batch import QuoraQuestionsBatch
from batchers.blimp_batch import BlimpBatch
from batchers.paws_batch import PawsBatch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm
from models.ranger import Ranger
# from models.ada_hessian import AdaHessian
from models.adahessian import Adahessian, get_params_grad


print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]

if config['training']['log']:
    now = datetime.now()
    writer = SummaryWriter(log_dir="./train_torch/logs/"+config['name'])

class LabelSmoothingCrossEntropy(torch.nn.Module):
    # based on https://github.com/seominseok0429/label-smoothing-visualization-pytorch
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.2):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

# def custom_reg(model):
#     reg_loss_max = 0
#     reg_loss_sparse = 0
#     for param in model.parameters():
#         reg_loss_max += torch.abs(1-torch.max(torch.abs(param)))
#         reg_loss_sparse += torch.mean(torch.abs(param))
#     return reg_loss_max + reg_loss_sparse

def zs_train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    total = 0.0
    correct = 0.0
    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    criterion = LabelSmoothingCrossEntropy()
    for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(train_loader):
        sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
        sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
        pred_class = model_output['zs_pred']
        pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)

        train_loss += pred_loss.detach()

        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.mean(torch.abs(param))
        (1e-2*reg_loss+pred_loss).backward(retain_graph=True)
        
        # pred_loss.backward(retain_graph=True)
        optimizer.step()

        _, pred_idx = torch.max(pred_class, 1)
        label_idx = label
        total += label.size(0)
        correct += (pred_idx == label_idx).sum().item()

        pbar.set_description("Acc: " + str(round(100.0*correct/total, 1)) + "%")
        pbar.update(1)

    pbar.close()
    end = time.time()
    train_loss /= batch_idx + 1
    print("")
    print('Epoch {}:\tZero Shot RE'.format(epoch))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('acc/zs_train', 100.0*correct/total, epoch)
        writer.add_scalar('loss/zs_train', train_loss, epoch)
        writer.flush()

def zs_test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = LabelSmoothingCrossEntropy()
    with torch.no_grad():
        for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(test_loader):
            sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
            sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
            label = label.to(device)

            # model prediction
            model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
            pred_class = model_output['zs_pred']

            # model training
            pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)
            test_loss += pred_loss.detach()

            _, pred_idx = torch.max(pred_class, 1)
            label_idx = label
            total += label.size(0)
            correct += (pred_idx == label_idx).sum().item()

    test_loss /= batch_idx + 1
    if config['training']['log']:
        writer.add_scalar('loss/zs_test', test_loss, epoch)
        writer.add_scalar('acc/zs_test', 100.0*correct/total, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    return test_loss

def qq_train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    total = 0.0
    correct = 0.0
    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    criterion = LabelSmoothingCrossEntropy()
    for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(train_loader):
        sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
        sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
        pred_class = model_output['qq_pred']
        pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)

        train_loss += pred_loss.detach()

        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.mean(torch.abs(param))
        (1e-2*reg_loss+pred_loss).backward(retain_graph=True)
        
        # pred_loss.backward(retain_graph=True)
        optimizer.step()

        _, pred_idx = torch.max(pred_class, 1)
        label_idx = label
        total += label.size(0)
        correct += (pred_idx == label_idx).sum().item()

        pbar.set_description("Acc: " + str(round(100.0*correct/total, 1)) + "%")
        pbar.update(1)

    pbar.close()
    end = time.time()
    train_loss /= batch_idx + 1
    print("")
    print('Epoch {}:\tQuora questions'.format(epoch))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('acc/qq_train', 100.0*correct/total, epoch)
        writer.add_scalar('loss/qq_train', train_loss, epoch)
        writer.flush()

def qq_test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = LabelSmoothingCrossEntropy()
    with torch.no_grad():
        for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(test_loader):
            sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
            sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
            label = label.to(device)

            model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
            pred_class = model_output['qq_pred']

            pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)
            test_loss += pred_loss.detach()

            _, pred_idx = torch.max(pred_class, 1)
            label_idx = label
            total += label.size(0)
            correct += (pred_idx == label_idx).sum().item()

    test_loss /= batch_idx + 1
    if config['training']['log']:
        writer.add_scalar('loss/qq_test', test_loss, epoch)
        writer.add_scalar('acc/qq_test', 100.0*correct/total, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    return test_loss

def blimp_train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    total = 0.0
    correct = 0.0
    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    criterion = LabelSmoothingCrossEntropy()
    for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(train_loader):
        sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
        sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
        pred_class = model_output['blimp_pred']
        pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)

        train_loss += pred_loss.detach()

        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.mean(torch.abs(param))
        (1e-2*reg_loss+pred_loss).backward(retain_graph=True)

        # pred_loss.backward(retain_graph=True)
        optimizer.step()

        _, pred_idx = torch.max(pred_class, 1)
        label_idx = label
        total += label.size(0)
        correct += (pred_idx == label_idx).sum().item()

        pbar.set_description("Acc: " + str(round(100.0*correct/total, 1)) + "%")
        pbar.update(1)

    pbar.close()
    end = time.time()
    train_loss /= batch_idx + 1
    print("")
    print('Epoch {}:\tBlimp'.format(epoch))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('acc/blimp_train', 100.0*correct/total, epoch)
        writer.add_scalar('loss/blimp_train', train_loss, epoch)
        writer.flush()

def blimp_test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = LabelSmoothingCrossEntropy()
    with torch.no_grad():
        for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(test_loader):
            sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
            sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
            label = label.to(device)

            model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
            pred_class = model_output['blimp_pred']

            pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)
            test_loss += pred_loss.detach()

            _, pred_idx = torch.max(pred_class, 1)
            label_idx = label
            total += label.size(0)
            correct += (pred_idx == label_idx).sum().item()

    test_loss /= batch_idx + 1
    if config['training']['log']:
        writer.add_scalar('loss/blimp_test', test_loss, epoch)
        writer.add_scalar('acc/blimp_test', 100.0*correct/total, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    return test_loss

def paws_train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    total = 0.0
    correct = 0.0
    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    criterion = LabelSmoothingCrossEntropy()
    for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(train_loader):
        sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
        sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
        pred_class = model_output['paws_pred']
        pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)

        train_loss += pred_loss.detach()

        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.mean(torch.abs(param))
        (1e-2*reg_loss+pred_loss).backward(retain_graph=True)

        # pred_loss.backward(retain_graph=True)
        optimizer.step()

        _, pred_idx = torch.max(pred_class, 1)
        label_idx = label
        total += label.size(0)
        correct += (pred_idx == label_idx).sum().item()

        pbar.set_description("Acc: " + str(round(100.0*correct/total, 1)) + "%")
        pbar.update(1)

    pbar.close()
    end = time.time()
    train_loss /= batch_idx + 1
    print("")
    print('Epoch {}:\tPaws'.format(epoch))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('acc/paws_train', 100.0*correct/total, epoch)
        writer.add_scalar('loss/paws_train', train_loss, epoch)
        writer.flush()

def paws_test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = LabelSmoothingCrossEntropy()
    with torch.no_grad():
        for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(test_loader):
            sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
            sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
            label = label.to(device)

            model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
            pred_class = model_output['paws_pred']

            pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)
            test_loss += pred_loss.detach()

            _, pred_idx = torch.max(pred_class, 1)
            label_idx = label
            total += label.size(0)
            correct += (pred_idx == label_idx).sum().item()

    test_loss /= batch_idx + 1
    if config['training']['log']:
        writer.add_scalar('loss/paws_test', test_loss, epoch)
        writer.add_scalar('acc/paws_test', 100.0*correct/total, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    return test_loss

def nli_train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0.0
    total = 0.0
    correct = 0.0
    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    criterion = LabelSmoothingCrossEntropy()
    for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(train_loader):
        sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
        sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
        # sent2_words = sent2_words.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
        pred_class = model_output['nli_pred']

        pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)
        train_loss += pred_loss.detach()

        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.mean(torch.abs((param)))
        (1e-2*reg_loss+pred_loss).backward(retain_graph=True)

        # pred_loss.backward(retain_graph=True)
        optimizer.step()

        _, pred_idx = torch.max(pred_class, 1)
        label_idx = label
        total += label.size(0)
        correct += (pred_idx == label_idx).sum().item()

        if scheduler:
            scheduler.step()

        pbar.set_description("Acc: " + str(round(100.0*correct/total, 1)) + "%")
        pbar.update(1)
    pbar.close()
    end = time.time()
    train_loss /= batch_idx + 1
    print("")
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/nli_train', train_loss, epoch)
        writer.add_scalar('acc/nli_train', 100.0*correct/total, epoch)
        writer.flush()

def nli_test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = LabelSmoothingCrossEntropy()
    with torch.no_grad():
        for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(test_loader):
            sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
            sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
            # sent2_words = sent2_words.to(device)
            label = label.to(device)

            model_output = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
            pred_class = model_output['nli_pred']

            pred_loss = criterion(pred_class, label.to(torch.long), smoothing=0.2)
            test_loss += pred_loss.detach()

            _, pred_idx = torch.max(pred_class, 1)
            label_idx = label
            total += label.size(0)
            correct += (pred_idx == label_idx).sum().item()

    test_loss /= batch_idx + 1
    if config['training']['log']:
        writer.add_scalar('loss/nli_test', test_loss, epoch)
        writer.add_scalar('acc/nli_test', 100.0*correct/total, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    return test_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = SentenceEncoder(config)
# restore_name = 'bL24_sl64_ZsQqBlimpAnli_CeLoss_4xTrPreNormGelu_dr.1_Mha16GeluDense_Ffnd4_Softmax8(x,y)*y+x__Mha128hPool_MeanPoolMask_2*1024d_AdamLr8e-5dec.95_v2_labelSmoothing.2_63'
# checkpoint = torch.load('./train_torch/save/'+restore_name)
# model.load_state_dict(checkpoint['model_state_dict'])
print(model)
model.to(device)
start_epoch = 1
# start_epoch += checkpoint['epoch']
# del checkpoint

# pretrain
if config['training']['pretrain']:
    zs_dataset_train = ZeroShotReBatch(config)
    zs_data_loader_train = torch.utils.data.DataLoader(
        zs_dataset_train, batch_size=config['batch_size'],
        shuffle=True, num_workers=0)
    zs_dataset_test = ZeroShotReBatch(config, valid=True)
    zs_data_loader_test = torch.utils.data.DataLoader(
        zs_dataset_test, batch_size=config['batch_size'],
        shuffle=False, num_workers=0)

    qq_dataset_train = QuoraQuestionsBatch(config)
    qq_data_loader_train = torch.utils.data.DataLoader(
        qq_dataset_train, batch_size=config['batch_size'],
        shuffle=True, num_workers=0)
    qq_dataset_test = QuoraQuestionsBatch(config, valid=True)
    qq_data_loader_test = torch.utils.data.DataLoader(
        qq_dataset_test, batch_size=config['batch_size'],
        shuffle=False, num_workers=0)

    blimp_dataset_train = BlimpBatch(config)
    blimp_data_loader_train = torch.utils.data.DataLoader(
        blimp_dataset_train, batch_size=config['batch_size'],
        shuffle=True, num_workers=0)
    blimp_dataset_test = BlimpBatch(config, valid=True)
    blimp_data_loader_test = torch.utils.data.DataLoader(
        blimp_dataset_test, batch_size=config['batch_size'],
        shuffle=False, num_workers=0)

    paws_dataset_train = PawsBatch(config)
    paws_data_loader_train = torch.utils.data.DataLoader(
        paws_dataset_train, batch_size=config['batch_size'],
        shuffle=True, num_workers=0)
    paws_dataset_test = PawsBatch(config, valid=True)
    paws_data_loader_test = torch.utils.data.DataLoader(
        paws_dataset_test, batch_size=config['batch_size'],
        shuffle=False, num_workers=0)

# training
dataset_train = AnliBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=0)
dataset_test = AnliBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=0)

# optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=0.1)
# optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=0.01)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
# optimizer = Adahessian(model.parameters(), lr=config['training']['lr'])
# optimizer = Ranger(model.parameters(), lr=config['training']['lr'], weight_decay=0.01)

# for parameter in model.parameters():
#     print(parameter)
#     h = parameter.register_hook(lambda grad: grad + torch.mean(grad)*0.1*torch.randn_like(grad))
# for name, parameter in model.named_parameters():
#     if 'fc' in name:
#         print(name)
#         h = parameter.register_hook(lambda grad: grad*0.0)

# optimizer = optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
# lr_lambda = lambda epoch: (0.95 ** (epoch + 2*max(0, epoch-15)))
lr_lambda = lambda epoch: 0.95 ** epoch
# lr_lambda = lambda epoch: 1.0
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-6, 2e-4, step_size_up=200, mode='triangular', gamma=1.0, cycle_momentum=False)
test_loss = 1e6
for epoch in range(start_epoch, config['training']['epochs'] + 1):
    if config['training']['pretrain']:
        zs_train(model, device, zs_data_loader_train, optimizer, epoch)
        zs_test(model, device, zs_data_loader_test, epoch)
        zs_dataset_train.on_epoch_end()

        qq_train(model, device, qq_data_loader_train, optimizer, epoch)
        qq_test(model, device, qq_data_loader_test, epoch)
        qq_dataset_train.on_epoch_end()

        blimp_train(model, device, blimp_data_loader_train, optimizer, epoch)
        blimp_test(model, device, blimp_data_loader_test, epoch)
        blimp_dataset_train.on_epoch_end()

        paws_train(model, device, paws_data_loader_train, optimizer, epoch)
        paws_test(model, device, paws_data_loader_test, epoch)
        paws_dataset_train.on_epoch_end()

    nli_train(model, device, data_loader_train, optimizer, epoch, None)
    current_test_loss = nli_test(model, device, data_loader_test, epoch)
    dataset_train.on_epoch_end()
    if current_test_loss < test_loss:
        test_loss = current_test_loss
        print("mean="+str(test_loss))
        # save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss
            }, './train_torch/save/'+config['name'])
    scheduler.step()

if config['training']['log']:
    writer.close()
