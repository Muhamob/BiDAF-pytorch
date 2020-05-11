import argparse
import copy, json, os

import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime
import numpy as np

from model.model import BiDAF
from model.data import SQuAD
from model.ema import EMA
import evaluate


def train(args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = BiDAF(args, data.WORD.vocab.vectors).to(device)

    ema = EMA(args.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    optimizer = optim.Adamax(parameters, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    best_model = copy.deepcopy(model)
    loss, last_epoch = 0, -1
    max_dev_accuracy = 0

    iterator = data.train_iter

    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch

        logits = model(batch)
        print(logits.view(-1))

        optimizer.zero_grad()
        batch_loss = criterion(logits, batch.answer.view(-1, 1).type(torch.cuda.FloatTensor))  # criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()
        batch_loss.backward()

        nn.utils.clip_grad_norm_(parameters, args.grad_clipping)

        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.update(name, param.data)

        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_accuracy = test(model, ema, args, data)
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('f1/dev', dev_accuracy, c)
            print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f}'
                  f' /  dev accuracy: {dev_accuracy:.3f}')

            if dev_accuracy > max_dev_accuracy:
                max_dev_accuracy = dev_accuracy
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()

    writer.close()
    print(f'max dev accuracy: {max_dev_accuracy:.3f}')

    return best_model


def test(model, ema, args, data):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    loss = 0
    predictions = []
    gt = []
    model.eval()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    with torch.set_grad_enabled(False):
        for batch in iter(data.dev_iter):
            logits = model(batch)
            batch_loss = criterion(logits, batch.answer.view(-1, 1).type(torch.cuda.FloatTensor))
            loss += batch_loss.item()

            # (batch, c_len, c_len)
            batch_size, c_len = logits.size()
            probs = torch.sigmoid(logits).data.cpu().numpy()
            predictions.extend(np.where(probs >= 0.5, 1, 0))
            gt.extend(batch.answer)

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))
    print("Количество ответов с меткой True", np.sum(predictions))

    return loss, accuracy_score(gt, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    args = parser.parse_args()

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'dataset_file', f'.data/squad/{args.dev_file}')
    setattr(args, 'prediction_file', f'prediction{args.gpu}.out')
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    print('data loading complete!')

    print('training start!')
    best_model = train(args, data)
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), f'saved_models/BiDAF_{args.model_time}.pt')
    print('training finished!')


if __name__ == '__main__':
    main()
