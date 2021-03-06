
import random
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

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


        self.encoder = EncoderLSTM(nwords,
                                   args['wes'],
                                   hidden_size,
                                   bidirectional=args['bidirectional'],
                                   dropout=args['dropout'],
                                   freeze_embeddings=args['E'][1] == 'True' if args['E'] != None else False,
                                   padding_idx=padding_idx_words).to(device)
        self.decoder = DecoderLSTM(nwords,
                                    nslots,
                                   args['ses'],
                                   hidden_size*2 if self.bidirectional else hidden_size,
                                   dropout=args['dropout'],
                                   padding_idx=padding_idx_words).to(device)

        self.ff_nn = NeuralNet(args['C'], hidden_size*2 if self.bidirectional else hidden_size, nintents).to(device)
        self.device = device



    def train(self,
              batch_data,
              batch_size,
              encoder_optimizer,
              decoder_optimizer,
              ff_nn_optimizer,
              encoder_scheduler,
              decoder_scheduler,
              fnn_scheduler,
              criterion_intent,
              criterion_slots,
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



        pred, hidden = self.decoder(X, lengths_X, decoder_hidden)


        for i in range(batch_size):
            loss += criterion_slots(pred[:, :-1, :][i], Y[:, 1:][i])

        # print(loss)

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        ff_nn_optimizer.step()

        encoder_scheduler.step()
        decoder_scheduler.step()
        fnn_scheduler.step()


        return loss.item() / Y[0].shape[0], loss_intent.item()



    def trainIters(self,
                   dataset,
                   batch_size,
                   no_epochs,
                   print_every=1000,
                   plot_every=100,
                   learning_rate=0.01):

        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        print_loss_intent = 0

        encoder_optimizer = optim.AdamW([p for p in self.encoder.parameters() if p.requires_grad], lr=learning_rate)
        decoder_optimizer = optim.AdamW([p for p in self.decoder.parameters() if p.requires_grad], lr=learning_rate)
        ff_nn_optimizer = optim.AdamW([p for p in self.ff_nn.parameters() if p.requires_grad], lr=learning_rate)

        encoder_scheduler = get_linear_schedule_with_warmup(encoder_optimizer,
                                                    num_training_steps=no_epochs,
                                                            num_warmup_steps=0)
        decoder_scheduler = get_linear_schedule_with_warmup(decoder_optimizer,
                                                            num_training_steps=no_epochs,
                                                            num_warmup_steps=0)
        fnn_scheduler = get_linear_schedule_with_warmup(ff_nn_optimizer,
                                                            num_training_steps=no_epochs,
                                                        num_warmup_steps=0)

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
                                       encoder_scheduler,
                                       decoder_scheduler,
                                       fnn_scheduler,
                                      criterion_intent,
                                      criterion_slots)
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


    def fit(self, train, dev, epochs, batch_size, learning_rate, save_every, save_model_path, print_every=50, do_predict=True):

        for i in range(epochs):
            print(f"******Epoch: {i}********")
            self.trainIters(train, batch_size, learning_rate=learning_rate, print_every=print_every, no_epochs=epochs)

            if do_predict:
                print('  predicting:')
                intent_true, intent_pred, slots_true, slots_pred = self.predict_and_get_labels_batch(dev, batch_size)
                intent_assessment, slots_assessment = assess(dev, intent_true, intent_pred, slots_true, slots_pred,
                                                             plot=False)
            if save_every != None and i % save_every  == 0:
                self.dump(save_model_path + '.epoch' + str(i))


        # exit(0)


    def evaluate_batch(self, X, lengths_X, batch_size, sorted = True):
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


            pred, hidden = self.decoder(X, lengths_X, decoder_hidden, sorted=sorted)

            # decoder_input = torch.ones(batch_size,1, dtype=torch.long).to(self.device) * SOS_token
            # lengths_Y = torch.ones(batch_size,dtype=torch.long)

            decoded_words = [[] for i in range(batch_size)]

            # iterate cols
            for i in range(cols):

                # iterate batches
                for j in range(pred.shape[0]):

                    curr_pred = pred[j, i, :]
                    pslot = curr_pred.topk(1)[1].item()
                    decoded_words[j].append(pslot)

                #pred, decoder_hidden = self.decoder(decoder_input, lengths_Y, decoder_hidden, sorted=sorted)

                # for j, p in enumerate(pred):
                #     pslot = p.topk(1)[1].item()
                #     decoder_input[j] = pslot
                #     decoded_words[j].append(pslot)


            for i, (length, decoded) in enumerate(zip(lengths_X, decoded_words)):
                # print(int(length.item()))

                # -1 accounting for the EOS token. lengths_X has EOS into account.
                decoded_words[i] = decoded[:(int(length.item())-1)]


            # pred, hidden = self.decoder(Y, lengths_Y, decoder_hidden)

            # print(intent_pred)
            # print(decoded_words)
            return decoded_words, intent_pred


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
                                batch_data[0].shape[0])



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



    def pretrained_embeddings(self, dataset, embeddings):

        ne = nn.Embedding(dataset.words_converter.no_entries(), self.encoder.embeddings_size, padding_idx=self.encoder.embedding.padding_idx).to(self.device)
        oesize = self.encoder.embedding.weight.data.shape[0]
        ne.weight.data[:oesize,:].copy_(self.encoder.embedding.weight.data)
        self.encoder.embedding = ne
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