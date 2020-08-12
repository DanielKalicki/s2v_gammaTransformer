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
    writer = SummaryWriter(log_dir="./train_torch/logs/"+config['name'])


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    train_contr_loss = 0.0

    total = 0.0
    correct = 0.0

    words_correct = 0
    words_total = 0

    train_s2v_mean_loss = 0.0

    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label, twords, tlabels) in enumerate(train_loader):
        sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
        sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
        label = label.to(device)
        twords, tlabels = twords.to(device), tlabels.to(device)

        optimizer.zero_grad()

        # model prediction
        # s2v_sent1, s2v_sent2, pred_class = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask)
        s2v_sent1, s2v_sent2, pred_class, twords_pred = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask, test_words=twords)

        # pred_loss = F.binary_cross_entropy_with_logits(pred_class, label, reduction='mean')
        pred_loss = F.cross_entropy(pred_class, label.to(torch.long), reduction='mean')
        train_loss += pred_loss.detach()

        twords_pred = twords_pred.view(-1, 2)
        tlabels = tlabels.view(-1, 2)
        words_pred_loss = F.binary_cross_entropy_with_logits(twords_pred, tlabels, reduction='mean')

        # s2v_contr_anchor = s2v_sent2
        # sent1_ = sent1  # F.dropout(sent1, 0.3)
        # sent2_ = F.dropout(sent2, 0.3)
        # s2v_contr_negative, s2v_contr_positive, _, _ = model(sent1_, sent2_, sent1_mask=sent1_mask, sent2_mask=sent2_mask, test_words=twords)
        # contr_loss = F.triplet_margin_loss(s2v_contr_anchor[1:], s2v_contr_positive[1:], s2v_contr_negative[:-1])
        # train_contr_loss += contr_loss.detach()

        s2v_contr_negative = s2v_sent1
        s2v_contr_positive = s2v_sent2
        sent_anchor1 = F.dropout(sent2, 0.3)*(1-0.3)
        sent_anchor2 = F.dropout(sent2, 0.3)*(1-0.3)
        s2v_contr_anchor1, s2v_contr_anchor2, _, _ = model(sent_anchor1, sent_anchor2, sent1_mask=sent2_mask, sent2_mask=sent2_mask, test_words=twords)

        # contr_loss1 = F.triplet_margin_loss(s2v_contr_anchor1, s2v_contr_positive, s2v_contr_negative)
        # contr_loss1_p = (s2v_contr_anchor1 - s2v_contr_positive).pow(2).mean()
        # contr_loss2_p = (s2v_contr_anchor2 - s2v_contr_positive).pow(2).mean()
        # contr_loss = contr_loss1_p + contr_loss2_p
        # train_contr_loss += contr_loss.detach()

        contr_loss1_p = F.cosine_similarity(s2v_contr_anchor1, s2v_contr_positive).mean()
        contr_loss1_n = F.cosine_similarity(s2v_contr_anchor1, s2v_contr_negative).mean()
        contr_loss = F.relu(contr_loss1_p - contr_loss1_n + 1.0)
        train_contr_loss += contr_loss.detach()

        # print("------------")
        # print(s2v_contr_anchor1[1])
        # print(s2v_contr_positive[1])
        # print(s2v_contr_negative[1])
        # print(contr_loss)

        # s2v_mean_loss = torch.sum(torch.mean(torch.pow(s2v_sent1, 2), dim=1)) + torch.sum(torch.mean(torch.pow(s2v_sent2, 2), dim=1))
        # train_s2v_mean_loss += s2v_mean_loss.detach()

        # pred_loss = pred_loss + 0.1*words_pred_loss + 0.1*s2v_mean_loss + 0.1*order_pred_loss
        pred_loss = pred_loss + 0.1*words_pred_loss + 0.1*contr_loss

        pred_loss.backward(retain_graph=True)
        optimizer.step()

        _, pred_idx = torch.max(pred_class, 1)
        # _, label_idx = torch.max(label, 1)
        label_idx = label
        total += label.size(0)
        correct += (pred_idx == label_idx).sum().item()

        _, pred_idx = torch.max(twords_pred, 1)
        _, label_idx = torch.max(tlabels, 1)
        words_total += tlabels.size(0)
        words_correct += (pred_idx == label_idx).sum().item()

        pbar.set_description("Acc: " + str(round(100.0*correct/total,1)) + "%")
        pbar.update(1)
    pbar.close()
    end = time.time()
    train_loss /= batch_idx+1
    train_contr_loss /= batch_idx+1
    train_s2v_mean_loss /= batch_idx+1
    print("")
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss_s2v_mean/train', train_s2v_mean_loss, epoch)
        writer.add_scalar('loss_contr/train', train_contr_loss, epoch)
        writer.add_scalar('acc/train', 100.0*correct/total, epoch)
        writer.add_scalar('acc_words/train', 100.0*words_correct/words_total, epoch)
        writer.flush()


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    words_correct = 0
    words_total = 0
    with torch.no_grad():
        for batch_idx, (sent1, sent1_mask, sent2, sent2_mask, label, twords, tlabels) in enumerate(test_loader):
            sent1, sent1_mask = sent1.to(device), sent1_mask.to(device)
            sent2, sent2_mask = sent2.to(device), sent2_mask.to(device)
            label = label.to(device)
            twords, tlabels = twords.to(device), tlabels.to(device)

            # model prediction
            # s2v_sent1, s2v_sent2, pred_class, twords_pred = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask, test_words=twords)
            s2v_sent1, s2v_sent2, pred_class, twords_pred = model(sent1, sent2, sent1_mask=sent1_mask, sent2_mask=sent2_mask, test_words=twords)

            # model training
            # pred_loss = F.binary_cross_entropy_with_logits(pred_class, label, reduction='mean')
            pred_loss = F.cross_entropy(pred_class, label.to(torch.long), reduction='mean')
            test_loss += pred_loss.detach()

            _, pred_idx = torch.max(pred_class, 1)
            # _, label_idx = torch.max(label, 1)
            label_idx = label
            total += label.size(0)
            correct += (pred_idx == label_idx).sum().item()

            twords_pred = twords_pred.view(-1, 2)
            tlabels = tlabels.view(-1, 2)
            _, pred_idx = torch.max(twords_pred, 1)
            _, label_idx = torch.max(tlabels, 1)
            words_total += tlabels.size(0)
            words_correct += (pred_idx == label_idx).sum().item()

    test_loss /= batch_idx+1
    if config['training']['log']:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('acc/test', 100.0*correct/total, epoch)
        writer.add_scalar('acc_words/test', 100.0*words_correct/words_total, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    print('\t\tAverage acc: {:.4f}'.format(100.0*correct/total))
    return test_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = SentenceEncoder(config)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
print(model)
model.to(device)
start_epoch = 1

dataset_train = AnliBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=0)

dataset_test = AnliBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=0)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
test_loss = 1e6
for epoch in range(start_epoch, config['training']['epochs'] + 1):
    train(model, device, data_loader_train, optimizer, epoch)
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
