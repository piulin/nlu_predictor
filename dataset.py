import json
from tqdm import tqdm
from io import open
import torch
import time
import random
from sklearn.model_selection import train_test_split
from seq_id import seq_id
from utils import normalizeString, tokenize
import numpy as np


class dataset(object):

    def __init__(self,
                 device,
                 words_converter=seq_id(lock_available=True),
                 slots_converter=seq_id(),
                 intent_converter=seq_id()):



        # slots id  structure
        self.slots_converter = slots_converter

        # words id structure
        self.words_converter = words_converter

        # intent id structure
        self.intent_converter = intent_converter


        self.stcs_literals = []
        self.lengths = []
        self.stcs = []
        self.slots = []
        self.intents = []
        self.device = device


    def read_training_dataset(self, input_path):
        with open(input_path) as f:

            data = json.load(f)
            self.no_samples = len(data)

            # for padding.
            self.words_converter.T2id('<PAD>')

            self.words_converter.T2id('<SOS>')

            self.slots_converter.T2id('<PAD>')
            self.slots_converter.T2id('<SOS>')

            self.slots_converter.T2id('-')

            for i in tqdm(range(self.no_samples)):

                entry = data[str(i)]

                text = entry["text"]
                text = normalizeString(text)
                tokens = tokenize(text)
                self.stcs_literals.append(tokens)
                tokens_id = [self.words_converter.T2id(id) for id in tokens]
                tokens_id.append(self.words_converter.T2id('<EOS>'))
                self.stcs.append(tokens_id)
                self.lengths.append(len(tokens_id))

                intent = entry["intent"]



                self.intents.append(self.intent_converter.T2id(intent))

                slots_dictionary = entry["slots"]
                # +1 make room for <SOS>
                slots_id = [self.slots_converter.T2id('-')] * (len(tokens_id) + 1)
                slots_id [ 0 ]  = self.slots_converter.T2id('<SOS>')

                no_slots_in_stc = 0
                for slot, target_words in slots_dictionary.items():
                    target_words = normalizeString(target_words)
                    target_word_list = tokenize(target_words)
                    for word in target_word_list:
                        no_slots_in_stc += 1
                        try:
                            idx = tokens.index(word)
                        except:
                            idx = [i for i, s in enumerate(tokens) if word in s][0]

                        # +1 account for <SOS>
                        slots_id[idx + 1] = self.slots_converter.T2id(slot)

                # keep count of no slots
                for j in range(len(tokens_id) - no_slots_in_stc):
                    self.slots_converter.T2id('-')

                self.slots.append(slots_id)
                # self.slots.append(torch.tensor(slots_id, dtype=torch.long, device=self.device))


            # add padding

            ncols = max(self.lengths)

            self.X = self.stcs
            self.Y = self.slots


    def __len__(self):
        return self.no_samples

    def __getitem__(self, index):
        src_sent = self.X[index]
        tgt_slot = self.Y[index]
        tgt_intent = self.intents[index]
        stc_size = self.lengths[index]

        return src_sent, tgt_slot, tgt_intent, stc_size

    def get_labels_slots(self):
        return [self.slots_converter.id2T(i) for i in range(2,self.slots_converter.no_entries())]


    def shuffle(self):
        current_milli_time = lambda: int(round(time.time() * 1000))
        timemils = current_milli_time()

        random.seed(timemils)

        random.shuffle(self.stcs)

        random.seed(timemils)

        random.shuffle(self.slots)

        random.seed(timemils)

        random.shuffle(self.intents)

    def slots_weights(self):

        # ojo
        w = torch.tensor([1. / self.slots_converter.freq_dict_[i]
                          for i in range(2,len(self.slots_converter.freq_dict_))], device=self.device)
        s = torch.sum(w)
        w *= 1. / s
        w = torch.cat((torch.tensor([0.,0.],device=self.device), w), dim=0)
        # suma = torch.sum(w)
        return w

    def train_test_splits(self, test_size, td=False):

        train = dataset(self.device)
        if td:
            test = test_dataset(self.device)
        else:
            test = dataset(self.device)

        train.stcs, test.stcs, train.slots, test.slots = train_test_split(self.stcs,
                                                                          self.slots,
                                                                          test_size=test_size,
                                                                          random_state=0)
        _, _, train.intents, test.intents = train_test_split(self.stcs,
                                                             self.intents,
                                                             test_size=test_size,
                                                             random_state=0)

        _, _, train.lengths, test.lengths = train_test_split(self.stcs,
                                                             self.lengths,
                                                             test_size=test_size,
                                                             random_state=0)

        train.X = train.stcs
        train.Y = train.slots

        test.X = test.stcs
        test.Y = test.slots

        train.no_samples = len(train.X)
        test.no_samples = len(test.X)


        train.slots_converter = self.slots_converter
        test.slots_converter = self.slots_converter

        # words id structure
        train.words_converter = self.words_converter
        test.words_converter = self.words_converter

        # intent id structure
        train.intent_converter = self.intent_converter
        test.intent_converter = self.intent_converter

        return train, test


    def collate_fn(self, data):

        # print(data)
        batch_size = len(data)
        sizes = [ l[3] for l in data ]

        iter_steps = np.argsort(-np.array(sizes))
        maxsize = max(sizes)

        X = torch.ones((batch_size, maxsize), dtype=torch.long) * self.words_converter.T2id('<PAD>')
        # account for the SOS
        Y = torch.ones((batch_size, maxsize + 1), dtype=torch.long) * self.slots_converter.T2id('<PAD>')
        intent = torch.zeros( batch_size, dtype=torch.long )
        sizes_array_X = torch.zeros( batch_size)
        sizes_array_Y = torch.zeros( batch_size )

        # copy over the actual sequences
        for i, j in enumerate(iter_steps):
            x_len = sizes[j]
            sequence = data[j][0]
            X[i, 0:x_len] = torch.tensor(sequence[:x_len])
            sequence = data[j][1]
            #also SOS
            Y[i, 0:x_len+1] = torch.tensor(sequence[:x_len+1])
            sequence = data[j][2]
            intent[i] = torch.tensor(sequence)
            sizes_array_X[i] = sizes[j]
            sizes_array_Y[i] = sizes[j] + 1

        X = X.to(self.device)
        Y = Y.to(self.device)
        intent = intent.to(self.device)
        # sizes_array_X =  sizes_array_X.to(self.device)
        # sizes_array_Y = sizes_array_Y.to(self.device)

        # print(X.device)
        # print(self.device)
        return X, Y, intent, sizes_array_X, sizes_array_Y




class test_dataset(dataset):

    def __init__(self,
                 device,
                 words_converter=seq_id(lock_available=True),
                 slots_converter=seq_id(),
                 intent_converter=seq_id()
                 ):

        super().__init__(device,
                 words_converter,
                 slots_converter,
                 intent_converter)



    def read_test_dataset(self, input_path):
        with open(input_path) as f:
            data = json.load(f)
            self.no_samples = len(data)

            for i in tqdm(range(self.no_samples)):

                entry = data[str(i)]

                text = entry["text"]
                text = normalizeString(text)
                tokens = tokenize(text)
                self.stcs_literals.append(tokens)
                tokens_id = [self.words_converter.T2id(id) for id in tokens]
                tokens_id.append(self.words_converter.T2id('<EOS>'))
                self.stcs.append(tokens_id)
                self.lengths.append(len(tokens_id))

            self.X = self.stcs


    def __len__(self):
        return self.no_samples

    def __getitem__(self, index):
        src_sent = self.X[index]
        stc_size = self.lengths[index]

        return src_sent, stc_size

    def collate_fn(self, data):

        # print(data)
        batch_size = len(data)
        sizes = [ l[1] for l in data ]

        iter_steps = np.argsort(-np.array(sizes))
        maxsize = max(sizes)

        X = torch.ones((batch_size, maxsize), dtype=torch.long) * self.words_converter.T2id('<PAD>')

        sizes_array_X = torch.zeros( batch_size)

        # copy over the actual sequences
        for i, j in enumerate(iter_steps):
            x_len = sizes[j]
            sequence = data[j][0]
            X[i, 0:x_len] = torch.tensor(sequence[:x_len])
            sizes_array_X[i] = sizes[j]

        X = X.to(self.device)
        # sizes_array_X =  sizes_array_X.to(self.device)
        # sizes_array_Y = sizes_array_Y.to(self.device)

        # print(X.device)
        # print(self.device)
        return X, sizes_array_X

