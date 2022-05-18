from distutils import log
import re
import torch
from torch import logit, nn, threshold
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from .statistic import calculate, old_calculate
import numpy as np

class completionTrainer:
    def __init__(self, args, model, train_data, valid_data, valid_infer_data, test_infer_data):
        self.args = args
        cuda_condition = torch.cuda.is_available() and self.args.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        if cuda_condition and torch.cuda.device_count() > 1:
            self.wrap = True
            model = nn.DataParallel(model)
        else:
            self.wrap = False
        self.model = model.to(self.device)
        self.train_data = train_data
        self.valid_data = valid_data
        self.valid_infer_data = valid_infer_data
        self.test_infer_data = test_infer_data
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.lr_scheduler:
            if self.args.MultiStepLR:
                self.scheduler = MultiStepLR(self.optim, milestones=self.args.milestones, gamma=0.1)
            else:
                self.scheduler = ReduceLROnPlateau(self.optim, 'max', verbose=True, patience=args.patience, factor=0.1, min_lr=args.min_lr)
        self.clip = self.args.clip
        self.writer_path = '{}_{}_{}_{}'.format('completion', 'relation' if args.relation_path else 'Naive', 'meta' if args.meta else args.dataset,
                                             datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        print(self.writer_path)
        self.tensorboard_writer = SummaryWriter(os.path.join('run', self.writer_path))
        self.writer = open(os.path.join('run', self.writer_path, 'experiment.txt'), 'w')
        print(self.args, file=self.writer, flush=True)
        self.iter = -1
        self.best_epoch, self.best_top1 = 0, float('-inf')
        self.accu_steps = self.args.accu_batch_size // self.args.batch_size
        model_parameters = []
        for name, param in self.model.named_parameters():
            if 'path' in name:
                if self.args.relation_path or self.args.absolute_path:
                    model_parameters.append(param)
            if 'eye' in name:
                continue
            else:
                model_parameters.append(param)
        print("Total Parameters: {}*1e6".format(sum([p.nelement() for p in model_parameters]) / 1e6),
                file=self.writer, flush=True)

    def load(self, path):
        dic = torch.load(path, map_location='cpu')
        load_pre = ''
        model_pre = ''
        # print(dic.keys())
        for key, _ in dic.items():
            if 'module.' in key:
                load_pre = 'module.'
            else:
                load_pre = ''
            break
        for key, _ in self.model.state_dict().items():
            if 'module.' in key:
                model_pre = 'module.'
            else:
                model_pre = ''
            break
        if load_pre == '' and model_pre == 'module.':
            temp_dict = dict()
            for key, value in dic.items():
                temp_dict[model_pre + key] = value
            dic = temp_dict
        elif model_pre == '' and load_pre == 'module.':
            temp_dict = dict()
            for key, value in dic.items():
                temp_dict[key.replace(load_pre, model_pre)] = value
            dic = temp_dict
        temp_dict = dict()
        ori_dic = self.model.state_dict()
        for key, value in dic.items():
            if key in ori_dic and ori_dic[key].shape == value.shape:
                temp_dict[key] = value
        dic = temp_dict
        # print(dic.keys())
        for key, value in self.model.state_dict().items():
            if key not in dic:
                dic[key] = value
        self.model.load_state_dict(dic)
        print('Load Pretrain model => {}'.format(path))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.valid_data, train=False)

    def loss_func(self, logits, label):
        targets = F.one_hot(label, num_classes=self.args.num_classes)

        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

        return loss

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "valid"
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code, epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        if train:
            self.optim.zero_grad()
        for i, data in data_iter:
            data = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in data.items()}
            label = data['label']

            if train:
                self.model.train()
                # print(data)
                out = self.model(data)
                loss = self.loss_func(logits=out, label=label)  # avg at every step
                accu_loss = loss / self.accu_steps
                accu_loss.backward()
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                if (i + 1) % self.accu_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad()
            else:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(data)
                    loss = self.loss_func(logits=out, label=label)

            avg_loss += loss.item()
            post_fix = {
                'str': str_code,
                "epoch": epoch,
                "iter": i,
                "Iter loss": loss.item(),
            }
            if train:
                self.iter += 1
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('Loss', post_fix['Iter loss'], self.iter)
        avg_loss = avg_loss / len(data_iter)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss, file=self.writer, flush=True)
        print('-------------------------------------', file=self.writer, flush=True)

        if self.args.save and train:
            save_dir = './checkpoint'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.model.state_dict(),
                       os.path.join(save_dir, "{}_{}.pth".format(self.writer_path, epoch)))

    def predict_multi(self, epoch, test=True):
        if test:
            data_loader = self.test_infer_data
            str_code = 'test'
        else:
            data_loader = self.valid_infer_data
            str_code = 'valid'

        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code + '_infer', epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")

        labels = torch.Tensor([])
        res = torch.Tensor([])
        languages = torch.Tensor([])

        for i, data in data_iter:
            self.model.eval()
            data = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in
                    data.items()}

            label = data['label'].detach().cpu()
            labels = torch.cat((labels, label), dim=0)

            language = data['language'].detach().cpu()
            languages = torch.cat((languages, language), dim=0)

            with torch.no_grad():
                logits = self.model(data)
                logits = F.softmax(logits, dim=-1)
                res = torch.cat((res, logits.detach().cpu()), dim=0)

        idx_py = languages==0
        idx_ruby = languages==1
        idx_js = languages==2
        idx_go = languages==3

        py_labels = labels[idx_py]
        py_res = res[idx_py]

        ruby_labels = labels[idx_ruby]
        ruby_res = res[idx_ruby]

        js_labels = labels[idx_js]
        js_res = res[idx_js]

        go_labels = labels[idx_go]
        go_res = res[idx_go]

        top1_all, top5_all = self.acc(res, labels)
        top1_py, top5_py = self.acc(py_res, py_labels)
        top1_ruby, top5_ruby = self.acc(ruby_res, ruby_labels)
        top1_js, top5_js = self.acc(js_res, js_labels)
        top1_go, top5_go = self.acc(go_res, go_labels)

        print(
            "{} overall:    top1 acc. = {:.6f}, top5 acc. = {:.6f}".format(str_code, top1_all, top5_all), file=self.writer,
            flush=True)
        print(
            "{} python:     top1 acc. = {:.6f}, top5 acc. = {:.6f}".format(str_code, top1_py, top5_py), file=self.writer,
            flush=True)
        print(
            "{} ruby:       top1 acc. = {:.6f}, top5 acc. = {:.6f}".format(str_code, top1_ruby, top5_ruby), file=self.writer,
            flush=True)
        print(
            "{} javascript: top1 acc. = {:.6f}, top5 acc. = {:.6f}".format(str_code, top1_js, top5_js), file=self.writer,
            flush=True)
        print(
            "{} go:         top1 acc. = {:.6f}, top5 acc. = {:.6f}".format(str_code, top1_go, top5_go), file=self.writer,
            flush=True)
        if not test and self.args.lr_scheduler:
            if self.args.MultiStepLR:
                self.scheduler.step()
            else:
                self.scheduler.step(top1_all)
        if not test:
            if top1_all >= self.best_top1:
                self.best_top1 = top1_all
                self.best_epoch = epoch
            print("Best Valid At EP {},   best_top1 = {}".format(self.best_epoch, self.best_top1), file=self.writer,
                  flush=True)
        print('-------------------------------------', file=self.writer, flush=True)

    def acc(self, logits, target):
        # print(logits)
        top1_res = logits.argmax(dim=1)
        _, top5_res = logits.topk(5, dim=1, largest=True, sorted=True)
        top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
        top5_acc = torch.eq(target.view(-1, 1), top5_res).sum().float() / len(target)
        # print(top5_res)
        # print(top1_res)
        # print(target)
        return top1_acc.item(), top5_acc.item()