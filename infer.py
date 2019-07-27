import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy

from pytorch_pretrained_bert import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Inferer:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            self.model.load_state_dict(torch.load(opt.state_dict_path))
            logger.info(f"Loaded model {opt.model_name} from {opt.state_dict_path}")
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            self.model.load_state_dict(torch.load(opt.state_dict_path))

        self.valset = ABSADataset(opt.dataset_file['val'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def _predict(self,data_loader):

        import pandas as pd
        pred_df = pd.DataFrame(columns=['unique_hash','sentiment'])

        idx = 0

        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_input = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_hash = t_sample_batched['hash']
                t_outputs = self.model(t_input)

                pred = (torch.argmax(t_outputs, -1))

                pred_df.loc[idx] = [t_hash,pred]
                idx += 1
            
        pred_df.to_csv(f"{self.opt.model_name}_preds.csv",index=False)

        return f"{self.opt.model_name}_preds.csv"

    def run(self):
        # Loss and Optimizer
        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        test_data_loader = DataLoader(dataset=self.testset, batch_size=1, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=32, shuffle=False)
        
        self.model.eval()
        val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(val_acc, val_f1))

        self.model.eval()
        pred_csv = self._predict(test_data_loader)
        logger.info(f'Predictions saved to {pred_csv}')


if __name__ == "__main__":
    
    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
    }

    class Option(object): pass
    opt = Option()
    opt.model_name = 'aen_bert'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'twitter'
    opt.dataset_file = {
        'test': sys.argv[2],
        'val': './datasets/acl-14-short-data/test.raw'
    }
    opt.state_dict_path = sys.argv[1]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 80
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.pretrained_bert_name = "bert-base-uncased"
    opt.dropout = 0.1
    opt.l2reg = 0.01
    opt.device = 'cuda'
    opt.inputs_cols = ['text_raw_bert_indices', 'aspect_bert_indices']
    opt.bert_dim = 768
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    

    inf = Inferer(opt)
    inf.run()

                