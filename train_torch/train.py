import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import lr_scheduler
import torch.nn.functional as F
from models.s2v_gammaTransformer import SentenceEncoder
from batchers.anli_batch import AnliBatch
from batchers.autoencoder_batch import AutoencoderBatch
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
    writer = SummaryWriter(log_dir="./train_torch/logs/"+config['name'])

# def pretrain(model, device, train_loader, optimizer, epoch):
#     model.train()
#     words_correct = 0
#     words_total = 0
#     start = time.time()
#     pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
#     # TODO remove duplicated code
#     for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, words1, words1_labels, words2, words2_labels) in enumerate(train_loader):
#         sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
#         sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
#         words1, words1_labels = words1.to(device), words1_labels.to(device)
#         words2, words2_labels = words2.to(device), words2_labels.to(device)

#         optimizer.zero_grad()

#         # model prediction
#         _, _, _, words1_pred, words2_pred = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask, test_words1=words1, test_words2=words2)

#         words1_pred = words1_pred.view(-1, 2)
#         words1_labels = words1_labels.view(-1, 2)
#         words1_pred_loss = F.binary_cross_entropy_with_logits(words1_pred, words1_labels, reduction='mean')
        
#         words2_pred = words2_pred.view(-1, 2)
#         words2_labels = words2_labels.view(-1, 2)
#         words2_pred_loss = F.binary_cross_entropy_with_logits(words2_pred, words2_labels, reduction='mean')

#         pred_loss = words1_pred_loss + words2_pred_loss
#         pred_loss.backward(retain_graph=True)
#         optimizer.step()

#         _, pred_idx = torch.max(words1_pred, 1)
#         _, label_idx = torch.max(words1_labels, 1)
#         words_total += words1_labels.size(0)
#         words_correct += (pred_idx == label_idx).sum().item()

#         _, pred_idx = torch.max(words2_pred, 1)
#         _, label_idx = torch.max(words2_labels, 1)
#         words_total += words2_labels.size(0)
#         words_correct += (pred_idx == label_idx).sum().item()

#         pbar.set_description("Acc: " + str(round(100.0*words_correct/words_total, 1)) + "%")
#         pbar.update(1)

#     pbar.close()
#     end = time.time()
#     print("")
#     print('Epoch {}:\tPretrain'.format(epoch))
#     print('\t\tAverage acc: {:.4f}'.format(100.0*words_correct/words_total))
#     print('\t\tTraining time: {:.2f}'.format((end - start)))
#     if config['training']['log']:
#         writer.add_scalar('acc_words/pretrain', 100.0*words_correct/words_total, epoch)
#         writer.flush()

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
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

        pred_loss = F.cross_entropy(pred_class, label.to(torch.long), reduction='mean')
        train_loss += pred_loss.detach()

        pred_loss.backward(retain_graph=True)
        optimizer.step()

        _, pred_idx = torch.max(pred_class, 1)
        # _, label_idx = torch.max(label, 1)
        label_idx = label
        total += label.size(0)
        correct += (pred_idx == label_idx).sum().item()

        if scheduler:
            scheduler.step()

        pbar.set_description("Acc: " + str(round(100.0*correct/total,1)) + "%")
        pbar.update(1)
    pbar.close()
    end = time.time()
    train_loss /= batch_idx+1
    print("")
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('acc/train', 100.0*correct/total, epoch)
        writer.flush()


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label) in enumerate(test_loader):
            sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
            sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
            label = label.to(device)

            # model prediction
            s2v_sent1, s2v_sent2, pred_class = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)

            # model training
            pred_loss = F.cross_entropy(pred_class, label.to(torch.long), reduction='mean')
            test_loss += pred_loss.detach()

            _, pred_idx = torch.max(pred_class, 1)
            label_idx = label
            total += label.size(0)
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
model = SentenceEncoder(config)
print(model)
model.to(device)
start_epoch = 1

# pretrain
if config['training']['pretrain']:
    dataset_train = AutoencoderBatch(config)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['batch_size'],
        shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    for epoch in range(0, 1):
        pretrain(model, device, data_loader_train, optimizer, epoch)
        dataset_train.on_epoch_end()

# training
dataset_train = AnliBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=0)
dataset_test = AnliBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=0)

optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-6, 2e-4, step_size_up=200, mode='triangular', gamma=1.0, cycle_momentum=False)
test_loss = 1e6
for epoch in range(start_epoch, config['training']['epochs'] + 1):
    train(model, device, data_loader_train, optimizer, epoch, None)
    current_test_loss = test(model, device, data_loader_test, epoch)
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
