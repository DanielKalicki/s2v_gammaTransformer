import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import lr_scheduler
import torch.nn.functional as F
from models.s2v_gammaTransformer import SentenceEncoder
from batchers.anli_batch import AnliBatch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm


print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]


if config['training']['log']:
    now = datetime.now()
    writer = SummaryWriter(log_dir="./train/logs/"+config['name'])


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    total = 0.0
    correct = 0.0
    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(train_loader):
        sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
        sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        # model prediction
        s2v_sent1, s2v_sent2, pred_class = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)

        # model training
        pred_loss = F.binary_cross_entropy_with_logits(pred_class, label, reduction='mean')
        train_loss += pred_loss.detach()
        pred_loss.backward(retain_graph=True)
        optimizer.step()

        _, pred_idx = torch.max(pred_class, 2)
        _, label_idx = torch.max(label, 2)
        total += label.size(0)*label.size(1)
        correct += (pred_idx == label_idx).sum().item()

        pbar.update(1)
    pbar.close()
    end = time.time()
    print("")
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss/(batch_idx+1)))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/train', train_loss/(batch_idx+1), epoch)
        writer.add_scalar('acc/train', 100.0*correct/total, epoch)
        writer.flush()


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    test_mask_loss = 0
    test_prev_loss = 0
    test_next_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target, tr_mask, in_mask, out_mask, test_s2v, test_label) in enumerate(test_loader):
            data, target, tr_mask = data.to(device), target.to(device), tr_mask.to(device)
            in_mask, out_mask = in_mask.to(device), out_mask.to(device)
            test_s2v, test_label = test_s2v.to(device), test_label.to(device)

            data = torch.nn.functional.normalize(data, dim=2)
            target = torch.nn.functional.normalize(target, dim=2)
            test_s2v = torch.nn.functional.normalize(test_s2v, dim=2)

            output, pred_class = model(data, test_s2v, mask=tr_mask)
            pred_loss = F.binary_cross_entropy_with_logits(pred_class, test_label, reduction='mean')
            test_loss += pred_loss.detach()

            _, pred_idx = torch.max(pred_class, 2)
            _, label_idx = torch.max(test_label, 2)
            total += test_label.size(0)*test_label.size(1)
            correct += (pred_idx == label_idx).sum().item()

    test_loss /= batch_idx+1
    if config['training']['log']:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('acc/test', 100.0*correct/total, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    return test_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = DocEncoder(config)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
print(model)
restore_name = '_bL16_docl7_PrClAllSnt_TrVSntSzDt10vD10Doc*SentFix_rndAll_Test.15OthDc_aftClosSentTr_SrcTgt_noPosEnc_1500bFilesRndStartUpd_inDense_Tr4l16h1xhDr.1HidDim1024NormGatedNoFfn_normClasIn_Lr1e-4_resave_10'
checkpoint = torch.load('./train/save/'+restore_name)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
start_epoch = 1
start_epoch += checkpoint['epoch']
del checkpoint

dataset_train = WikiS2vBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=0)

dataset_test = WikiS2vBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=0)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.5)
test_loss = 1e6
loss_list = []
avr_samples = 5
for epoch in range(start_epoch, config['training']['epochs'] + 1):
    train(model, device, data_loader_train, optimizer, epoch)
    current_test_loss = test(model, device, data_loader_test, epoch)
    dataset_train.on_epoch_end()
    if len(loss_list) >= avr_samples:
        loss_list.pop()
    loss_list.append(float(current_test_loss))
    current_test_loss = sum(loss_list)/float(len(loss_list))
    if (current_test_loss < test_loss) and (len(loss_list) >= avr_samples):
        test_loss = current_test_loss
        print(str(loss_list)+"\tmean="+str(test_loss))
        # save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss
            }, './train/save/'+config['name'])
    # scheduler.step()

if config['training']['log']:
    writer.close()
