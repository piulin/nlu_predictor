
import random
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ai.ffnn import NeuralNet
from ai.lstm import EncoderLSTM, DecoderLSTM
from ai.rnn import EncoderRNN, DecoderRNN
from assessment import assess
from utils import timeSince
from tqdm import tqdm


class seq2seq(object):


    def __init__(self,
                 nwords,
                 nslots,
                 nintents,
                 device,
                 padding_idx_words,
                 padding_idx_slots,
                 args):

        self.padding_idx_words = padding_idx_words
        self.padding_idx_slots = padding_idx_slots
        hidden_size = args['hs']

        self.bidirectional = args['bidirectional']

        if args['architecture'] == 'lstm':
            self.encoder = EncoderLSTM(nwords,
                                       args['wes'],
                                       hidden_size,
                                       bidirectional=args['bidirectional'],
                                       dropout=args['dropout'],
                                       freeze_embeddings=args['E'][1] == 'True' if args['E'] != None else False,
                                       padding_idx=padding_idx_words).to(device)
            self.decoder = DecoderLSTM(nslots,
                                       args['ses'],
                                       hidden_size*2 if self.bidirectional else hidden_size,
                                       bidirectional=args['bidirectional'],
                                       dropout=args['dropout'],
                                       padding_idx=padding_idx_slots).to(device)

        else :
            self.encoder = EncoderRNN(nwords, hidden_size).to(device)
            self.decoder = DecoderRNN(hidden_size, nslots).to(device)

        self.ff_nn = NeuralNet(args['C'], hidden_size*2 if self.bidirectional else hidden_size, nintents).to(device)
        self.device = device



    def train(self,
              batch_data,
              batch_size,
              encoder_optimizer,
              decoder_optimizer,
              ff_nn_optimizer,
              criterion_intent,
              criterion_slots,
              SOS_token,
              teacher_forcing_ratio=0.5):

        # print(batch_data)
        loss = 0.

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        ff_nn_optimizer.zero_grad()

        hidden = self.encoder.initHidden(self.device, batch_size)

        X = batch_data[0]
        Y = batch_data[1]
        intents = batch_data[2]
        lengths_X = batch_data[3]
        lengths_Y = batch_data[4]

        output, hidden = self.encoder(X, lengths_X, hidden)

        # print(f'output: {output.shape}')
        # print(f'hidden: {len(hidden)}')

        # decoder_input = torch.tensor([[dts.slots_converter.T2id('<SOS>')]], device=device)

        if self.bidirectional:

            hperm = hidden[0].permute(1,0,2)
            cperm = hidden[1].permute(1,0,2)

            hperm = hperm.reshape(1, batch_size, hidden[0].shape[2]*2 )
            cperm = cperm.reshape(1, batch_size, hidden[1].shape[2]*2 )

            decoder_hidden = (hperm,cperm)

        else:
            decoder_hidden = hidden




        nnoutput = self.ff_nn(decoder_hidden[0][0])

        loss_intent = criterion_intent(nnoutput, intents)
        loss += loss_intent



        pred, hidden = self.decoder(Y, lengths_Y, decoder_hidden)


        for i in range(batch_size):
            loss += criterion_slots(pred[:, :-1, :][i], Y[:, 1:][i])

        # print(loss)

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        ff_nn_optimizer.step()

        return loss.item() / Y[0].shape[0], loss_intent.item()



    def trainIters(self,
                   dataset,
                   batch_size,
                   print_every=1000,
                   plot_every=100,
                   learning_rate=0.01):

        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        print_loss_intent = 0

        encoder_optimizer = optim.Adam([p for p in self.encoder.parameters() if p.requires_grad], lr=learning_rate)
        decoder_optimizer = optim.Adam([p for p in self.decoder.parameters() if p.requires_grad], lr=learning_rate)
        ff_nn_optimizer = optim.Adam([p for p in self.ff_nn.parameters() if p.requires_grad], lr=learning_rate)

        slots_weights = dataset.slots_weights()
        criterion_intent = nn.NLLLoss()
        criterion_slots = nn.NLLLoss(weight=slots_weights, ignore_index=dataset.slots_converter.T2id('<PAD>'))
        # criterion_slots = nn.NLLLoss(ignore_index=dataset.slots_converter.T2id('<PAD>'))

        train_iter = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=dataset.collate_fn,
                                drop_last=True)


        self.ff_nn.train()
        self.encoder.train()
        self.decoder.train()

        n_iters = len(train_iter)


        for iter, batch_data in enumerate(train_iter, start=1):

            # self.evaluate_batch(batch_data[0],batch_data[3],batch_size,dataset.slots_converter.T2id('<SOS>'))

            print('.', end='', flush=True)
            loss, loss_intent = self.train(batch_data,
                                           batch_size,
                                      encoder_optimizer,
                                      decoder_optimizer,
                                      ff_nn_optimizer,
                                      criterion_intent,
                                      criterion_slots,
                                      dataset.slots_converter.T2id('<SOS>'))
            print_loss_total += loss
            plot_loss_total += loss
            print_loss_intent += loss_intent

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0

                print_loss_intent_avg = print_loss_intent / print_every
                print_loss_intent = 0
                print('%s (%d %d%%) loss: %.4f, loss-intent: %.4f' % (timeSince(start, iter / n_iters),
                                                                      iter, iter / n_iters * 100, print_loss_avg,
                                                                      print_loss_intent_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # showPlot(plot_losses)


    def fit(self, train, dev, epochs, batch_size, learning_rate, print_every=50):

        for i in range(epochs):
            print(f"******Epoch: {i}********")
            self.trainIters(train, batch_size, learning_rate=learning_rate, print_every=print_every)
            print('  predicting:')
            intent_true, intent_pred, slots_true, slots_pred = self.predict_and_get_labels_batch(dev, batch_size)
            intent_assessment, slots_assessment = assess(dev, intent_true, intent_pred, slots_true, slots_pred,
                                                         plot=False)

        # exit(0)


    def evaluate_batch(self, X, lengths_X, batch_size, SOS_token, sorted = True):
        with torch.no_grad():

            self.ff_nn.eval()
            self.encoder.eval()
            self.decoder.eval()

            cols = X.shape[1]

            # print(batch_size)

            hidden = self.encoder.initHidden(self.device, batch_size)

            output, hidden = self.encoder(X, lengths_X, hidden, sorted=sorted)

            if self.bidirectional:

                hperm = hidden[0].permute(1, 0, 2)
                cperm = hidden[1].permute(1, 0, 2)

                hperm = hperm.reshape(1, batch_size, hidden[0].shape[2] * 2)
                cperm = cperm.reshape(1, batch_size, hidden[1].shape[2] * 2)

                decoder_hidden = (hperm, cperm)

            else:
                decoder_hidden = hidden


            nnoutput = self.ff_nn(decoder_hidden[0][0])

            intent_pred = [out.topk(1)[1].item() for out in nnoutput]

            decoder_input = torch.ones(batch_size,1, dtype=torch.long).to(self.device) * SOS_token
            lengths_Y = torch.ones(batch_size,dtype=torch.long)

            decoded_words = [[] for i in range(batch_size)]

            for i in range(cols):

                pred, decoder_hidden = self.decoder(decoder_input, lengths_Y, decoder_hidden, sorted=sorted)

                for j, p in enumerate(pred):
                    pslot = p.topk(1)[1].item()
                    decoder_input[j] = pslot
                    decoded_words[j].append(pslot)


            for i, (length, decoded) in enumerate(zip(lengths_X, decoded_words)):
                # print(int(length.item()))

                # -1 accounting for the EOS token. lengths_X hast EOS into account.
                decoded_words[i] = decoded[:(int(length.item())-1)]


            # pred, hidden = self.decoder(Y, lengths_Y, decoder_hidden)

            # print(intent_pred)
            # print(decoded_words)
            return decoded_words, intent_pred


    def evaluate(self,input_tensor, SOS_token):
        with torch.no_grad():

            self.ff_nn.eval()
            self.encoder.eval()
            self.decoder.eval()

            # input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size(0)
            encoder_hidden = self.encoder.initHidden(self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                         encoder_hidden)

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            nnoutput = self.ff_nn(self.encoder.get_hidden(encoder_hidden)[0])

            # print(f"nn output: {nnoutput}")

            topv, pred_intent = nnoutput.topk(1)

            decoded_words = []
            # decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(input_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)

                decoded_words.append(topi.item())
                decoder_input = topi.squeeze().detach()

            return decoded_words, pred_intent

    def predict_batch(self, test_dataset, batch_size):

        intent_pred = []

        slots_pred = []

        # print(test_dataset.stcs[0])


        test_iter = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               collate_fn=test_dataset.collate_fn)

        for batch_id, batch_data in enumerate(test_iter):
            # print(batch_data[0][0])
            pred_slots, pred_intent = self.evaluate_batch(batch_data[0],
                                batch_data[1],
                                batch_data[0].shape[0],
                                test_dataset.slots_converter.T2id('<SOS>'),
                                sorted=False)


            intent_pred.extend(pred_intent)

            # ss = []
            # for pred_slot in  pred_slots:
            #     ss.extend(pred_slot)
            slots_pred.extend(pred_slots)

        return intent_pred, slots_pred

    def predict_and_get_labels_batch( self, test_dataset, batch_size ):

        intent_true = []
        # intent_true = [i for i in test_dataset.intents]
        intent_pred = []

        slots_true = []
        slots_pred = []

        test_iter = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               collate_fn=test_dataset.collate_fn,
                               drop_last=False)

        for batch_id, batch_data in enumerate(test_iter):
            # print(batch_id)
            # print(batch_data)
            # print(len(batch_data))


            # print(batch_data)
            pred_slots, pred_intent = self.evaluate_batch(batch_data[0],
                                batch_data[3],
                                batch_data[0].shape[0],
                                test_dataset.slots_converter.T2id('<SOS>'))



            intent_true.extend(batch_data[2].tolist())
            intent_pred.extend(pred_intent)

            slots = batch_data[1]
            lengths_Y = batch_data[4]

            for slot, pred_slot, length in zip(slots, pred_slots, lengths_Y):
                slots_true.extend(slot[1:int(length)].tolist())
                slots_pred.extend(pred_slot)
                # for slot_true, slot_pred in zip(slot, pred_slot):
                #     slots_true.append(slot_true.item())
                #     slots_pred.append(slot_pred)


        #
        # for stc, slot, intent in zip(dataset.stcs, dataset.slots, dataset.intents):
        #
        #     pred_slots, pred_intent = self.evaluate(stc,
        #                                        dataset.slots_converter.T2id('<SOS>'))
        #     intent_pred.append(pred_intent.item())
        #
        #     for slot_true, slot_pred in zip(slot, pred_slots):
        #         slots_true.append(slot_true.item())
        #         slots_pred.append(slot_pred)

        return intent_true, intent_pred, slots_true, slots_pred

    def predict_and_get_labels( self, dataset ):

        intent_true = [i.item() for i in dataset.intents]
        intent_pred = []

        slots_true = []
        slots_pred = []

        for stc, slot, intent in zip(dataset.stcs, dataset.slots, dataset.intents):

            pred_slots, pred_intent = self.evaluate(stc,
                                               dataset.slots_converter.T2id('<SOS>'))
            intent_pred.append(pred_intent.item())

            for slot_true, slot_pred in zip(slot, pred_slots):
                slots_true.append(slot_true.item())
                slots_pred.append(slot_pred)

        return intent_true, intent_pred, slots_true, slots_pred

    def predict(self, dataset):

        intent_pred = []

        slots_pred = []

        for stc in dataset.stcs:

            pred_slots, pred_intent = self.evaluate(stc,
                                               dataset.slots_converter.T2id('<SOS>'))
            intent_pred.append(pred_intent.item())

            ss = []
            for slot_pred in pred_slots:
                ss.append(slot_pred)
            slots_pred.append(ss)

        return intent_pred, slots_pred


    def pretrained_embeddings(self, dataset, embeddings):

        for i in tqdm(range(dataset.words_converter.no_entries())):
            word = dataset.words_converter.id2T(i)
            try:
                # print(f"before: {self.encoder.embedding.weight[i]}")
                self.encoder.embedding.weight[i].data.copy_(torch.from_numpy(embeddings[word]))
                # print(f"after: {self.encoder.embedding.weight[i]}")
            except:
                pass

    def dump(self,path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        """
        Loads a model from a file.
        :param path:
        :return: CNN with weights already set up.
        """
        return torch.load(path)