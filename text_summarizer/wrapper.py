import time, random, gc, os, math
from os import path
import numpy as np
import sys
import csv

import torch
import torch.utils.data as data

from mylog import mylog
from options_process import optionsLoader
from data_process import myDataSet_Bert as Dataset
from utility import *
from model_bert import *
from searcher.searcher import Searcher
from searcher.scorer import *

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

from parallel import DataParallelModel, DataParallelCriterion

class paras:
    def __init__(self):
        self.device = 0
        self.model = 'bert-base-uncased'
        self.PAD =0
        self.UNK =100
        self.CLS =101
        self.SEP =102
        self.MASK =103
        self.test_model = 'model/model_best.pth.tar'
        self.search_method = 'BFS_BEAM'
        self.beam_size = 5
        self.cands_limit =100000
        self.answer_size =5
        self.gen_max_len =50
        self.gamma_value =14.0
        self.beta_value =0.5
        self.reward =0.25
        self.no_biGramTrick = True
        self.no_triGramTrick = True
        self.biGramTrick = False
        self.triGramTrick = False

def argLoader():
    return paras()
    
class summarizer:
    
    def __init__(self):
        self.args = argLoader()

        if self.args.device:
            torch.cuda.set_device(self.args.device)

        #self.Best_Model = torch.load(self.args.test_model)
        self.Best_Model = torch.load(self.args.test_model, map_location=torch.device('cpu'))
    
        self.Tokenizer = BertTokenizer.from_pretrained(self.args.model)
        
        self.net = BertForMaskedLM.from_pretrained(self.args.model)
        
        if torch.cuda.is_available():
            self.net = self.net.cuda(0)
#             if self.args.dataParallel:
#                 self.net = DataParallelModel(self.net)
#             self.net = DataParallelModel(self.net)

        # When loading from a model trained from DataParallel
        self.net.load_state_dict(self.Best_Model['state_dict'])
#         self.net.load_state_dict(torch.load(self.args.test_model)['state_dict'])
        self.net.eval()

        self.mySearcher = Searcher(self.net, self.args)

        sys.stdout.write('Summarizer initialized. \n')
    
    def translate(self, Answers):
        tokens = self.Tokenizer.convert_ids_to_tokens(Answers[:-1])
        tokens = [token.replace('##', '') if token.startswith('##') else ' ' + token for token in tokens]
        return "".join(tokens)[1:]
    
    def write_to_file(self, orig_sent, sum_sents):
        
        with open('summary_sent.csv', mode = 'a') as file:
            fieldnames = ['sent1', 'sent2']
            wt = csv.DictWriter(file, delimiter = ',', fieldnames = fieldnames)
            
            for s in sum_sents:
                wt.writerow({'sent1': orig_sent, 'sent2': s})
                
    
    def summarize_2(self, txtlist):
        
        results = []
        sum_text = []
        count = 0
        dic = dict()
        file = open('summarized_text.txt','a') 
        
        for line in txtlist:
            
            result = {}
            result['text'] = line.strip()
            
            if(result['text'] in dic):
                s_text = dic[result['text']]
            else:
                source_ = line.strip().split()
                source = self.Tokenizer.tokenize(line.strip())
                mapping = mapping_tokenize(source_, source)

                source = self.Tokenizer.convert_tokens_to_ids(source)

                result['text_processed'] = detokenize(self.translate(source), mapping)

                l_pred = self.mySearcher.length_Predict(source)
                Answers = self.mySearcher.search(source)
                baseline = sum(Answers[0][0])

                Answers = sorted(Answers, key=lambda x: sum(x[0]))

                texts = [detokenize(self.translate(Answers[k][1]), mapping) for k in range(len(Answers))]

                result['summary'] = texts[0]
                s_text = texts[0]
                result['topk'] = texts
                results.append(result)
                count+=1
                print(count, " ", line, " ", s_text)
                dic[result['text']] = s_text
                #self.write_to_file(texts)
                #sys.exit()
        
            file.write(s_text+"\n")
            sum_text.append(s_text)

        return sum_text
    
    
        
    def summarize(self, txtlist):
        
        results = []
        count = 0

        for line in txtlist:
            
            result = {}
            result['text'] = line.strip()
            source_ = line.strip().split()
            source = self.Tokenizer.tokenize(line.strip())
            mapping = mapping_tokenize(source_, source)

            source = self.Tokenizer.convert_tokens_to_ids(source)

            result['text_processed'] = detokenize(self.translate(source), mapping)

            l_pred = self.mySearcher.length_Predict(source)
            Answers = self.mySearcher.search(source)
            baseline = sum(Answers[0][0])
            
            Answers = sorted(Answers, key=lambda x: sum(x[0]))

            texts = [detokenize(self.translate(Answers[k][1]), mapping) for k in range(len(Answers))]

            result['summary'] = texts[0]
            result['topk'] = texts
            results.append(result)
            count+=1
            print(count, " ", line, " ", texts)
            #self.write_to_file(texts)
            #sys.exit()
            
        return results
    
