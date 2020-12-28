
import random
import time

import torch
from torch import nn, optim

from ai.ffnn import NeuralNet
from ai.rnn import EncoderRNN, DecoderRNN
from assessment import assess
from utils import timeSince


class seq2seq(object):



    def __init__(self,
                 hidden_size,
                 nwords,
                 nslots,
                 nintents,
                 device,
                 nn_config_file):

        #hidden_size = 256
        #nwords = dts.words_converter.no_entries()
        #nslots = dts.slots_converter.no_entries()
        self.encoder = EncoderRNN(nwords, hidden_size).to(device)
        self.decoder = DecoderRNN(hidden_size, nslots).to(device)
        self.ff_nn = NeuralNet(nn_config_file, hidden_size, nintents).to(device)
        self.device = device



    def train(self,
              input_tensor,
              target_tensor,
              target_intent,
              encoder_optimizer,
              decoder_optimizer,
              ff_nn_optimizer,
              criterion_intent,
              criterion_slots,
              SOS_token,
              EOS_token,
              teacher_forcing_ratio=0.5):

        encoder_hidden = self.encoder.initHidden(self.device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        ff_nn_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(input_length, self.encoder.hidden_size, device=self.device)

        loss = 0.

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        # decide the intent
        nnoutput = self.ff_nn(encoder_hidden[0])

        # print(f"nn output: {nnoutput}")
        # print(f"target intent: {target_intent}")

        loss_intent = criterion_intent(nnoutput, target_intent)
        loss += loss_intent

        # topv, topi = nnoutput.topk(1)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # if False:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                loss += criterion_slots(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                # print(f"decoder output {decoder_output}")
                # print(f"target tensor {target_tensor[di]}")

                # print(f"{criterion(decoder_output, target_tensor[di])}")
                # print(decoder_output.shape)
                loss += criterion_slots(decoder_output, target_tensor[di])
                # if decoder_input.item() == EOS_token:
                # break
                # sys.exit(0)

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        ff_nn_optimizer.step()

        return loss.item() / target_length, loss_intent.item()


    def trainIters(self,
                   dataset,
                   print_every=1000,
                   plot_every=100,
                   learning_rate=0.01):

        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        print_loss_intent = 0

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        ff_nn_optimizer = optim.SGD(self.ff_nn.parameters(), lr=learning_rate)

        slots_weights = dataset.slots_weights();
        criterion_intent = nn.NLLLoss()
        criterion_slots = nn.NLLLoss(weight=slots_weights)
        dataset.shuffle()

        self.ff_nn.train()

        n_iters = len(dataset.stcs)

        for iter in range(1, n_iters + 1):

            input_tensor = dataset.stcs[iter - 1]
            target_tensor = dataset.slots[iter - 1]
            target_intent = dataset.intents[iter - 1]

            loss, loss_intent = self.train(input_tensor,
                                      target_tensor,
                                      target_intent,
                                      encoder_optimizer,
                                      decoder_optimizer,
                                      ff_nn_optimizer,
                                      criterion_intent,
                                      criterion_slots,
                                      dataset.words_converter.T2id('<SOS>'),
                                      dataset.words_converter.T2id('<EOS>'))
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


    def fit(self, dataset, epochs, learning_rate, print_every=500):

        for i in range(epochs):
            print(f"******Epoch: {i}********")
            self.trainIters(dataset, learning_rate=learning_rate, print_every=print_every)
            intent_true, intent_pred, slots_true, slots_pred = self.predict_and_get_labels(dataset)
            print('  predicting:')
            intent_assessment, slots_assessment = assess(dataset, intent_true, intent_pred, slots_true, slots_pred,
                                                         plot=False)

    def evaluate(self,input_tensor, SOS_token, EOS_token):
        with torch.no_grad():

            self.ff_nn.eval()

            # input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size(0)
            encoder_hidden = self.encoder.initHidden(self.device)

            encoder_outputs = torch.zeros(input_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            nnoutput = self.ff_nn(encoder_hidden[0])

            # print(f"nn output: {nnoutput}")

            topv, pred_intent = nnoutput.topk(1)

            decoded_words = []
            # decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(input_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                # if topi.item() == EOS_token:
                #    decoded_words.append('<EOS>')
                #    break
                # else:
                # decoded_words.append(slot_converter.id2T(topi.item()))
                decoded_words.append(topi.item())
                decoder_input = topi.squeeze().detach()

            return decoded_words, pred_intent

    def predict_and_get_labels( self, dataset ):

        intent_true = [i.item() for i in dataset.intents]
        intent_pred = []

        slots_true = []
        slots_pred = []

        for stc, slot, intent in zip(dataset.stcs, dataset.slots, dataset.intents):

            pred_slots, pred_intent = self.evaluate(stc,
                                               dataset.words_converter.T2id('<SOS>'),
                                               dataset.words_converter.T2id('<EOS>'))
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
                                               dataset.words_converter.T2id('<SOS>'),
                                               dataset.words_converter.T2id('<EOS>'))
            intent_pred.append(pred_intent.item())

            ss = []
            for slot_pred in pred_slots:
                ss.append(slot_pred)
            slots_pred.append(ss)

        return intent_pred, slots_pred


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