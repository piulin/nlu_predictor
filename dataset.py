import json
from tqdm import tqdm
from io import open
import torch
import time
import random
from sklearn.model_selection import train_test_split
from seq_id import seq_id
from utils import normalizeString, tokenize


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
        self.stcs = []
        self.slots = []
        self.intents = []
        self.device = device

    def read_test_dataset(self, input_path):
        with open(input_path) as f:
            data = json.load(f)
            no_samples = len(data)

            for i in tqdm(range(no_samples)):
                entry = data[str(i)]
                text = entry["text"]
                text = normalizeString(text)
                tokens = tokenize(text)
                self.stcs_literals.append(tokens)
                tokens_id = [self.words_converter.T2id(id, lock=True) for id in tokens]
                tokens_id.append(self.words_converter.T2id('<EOS>'))
                self.stcs.append(torch.tensor(tokens_id, dtype=torch.long, device=self.device).view(-1, 1))

    def read_training_dataset(self, input_path):
        with open(input_path) as f:

            data = json.load(f)
            no_samples = len(data)

            self.words_converter.T2id('<SOS>')
            self.words_converter.T2id('<EOS>')

            self.slots_converter.T2id('-')

            for i in tqdm(range(no_samples)):

                entry = data[str(i)]

                text = entry["text"]
                text = normalizeString(text)
                tokens = tokenize(text)
                self.stcs_literals.append(tokens)
                tokens_id = [self.words_converter.T2id(id) for id in tokens]
                tokens_id.append(self.words_converter.T2id('<EOS>'))
                self.stcs.append(torch.tensor(tokens_id, dtype=torch.long, device=self.device).view(-1, 1))

                intent = entry["intent"]

                self.intents.append(torch.tensor([self.intent_converter.T2id(intent)],
                                                 dtype=torch.long, device=self.device))

                slots_dictionary = entry["slots"]
                slots_id = [0] * len(tokens_id)

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
                        slots_id[idx] = self.slots_converter.T2id(slot)

                # keep count of no slots
                for j in range(len(tokens_id) - no_slots_in_stc):
                    self.slots_converter.T2id('-')

                self.slots.append(torch.tensor(slots_id, dtype=torch.long, device=self.device).view(-1, 1))

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
        w = torch.tensor([1. / self.slots_converter.freq_dict_[i]
                          for i in range(len(self.slots_converter.freq_dict_))], device=self.device)
        s = torch.sum(w)
        w *= 1. / s
        return w

    def train_test_splits(self):

        train = dataset(self.device)
        test = dataset(self.device)

        train.stcs, test.stcs, train.slots, test.slots = train_test_split(self.stcs,
                                                                          self.slots,
                                                                          test_size=0.2,
                                                                          random_state=0)
        _, _, train.intents, test.intents = train_test_split(self.stcs,
                                                             self.intents,
                                                             test_size=0.2,
                                                             random_state=0)

        train.slots_converter = self.slots_converter
        test.slots_converter = self.slots_converter

        # words id structure
        train.words_converter = self.words_converter
        test.words_converter = self.words_converter

        # intent id structure
        train.intent_converter = self.intent_converter
        test.intent_converter = self.intent_converter

        return train, test


