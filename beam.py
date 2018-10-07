import torch
import data.dict as dict

class Beam(object):
    def __init__(self, size, n_best=1, cuda=True):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(dict.EOS)]
        self.nextYs[0][0] = dict.BOS

        # Has EOS topped the beam yet.
        self._eos = dict.EOS
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best


    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == dict.EOS:
            # self.allScores.append(self.scores)
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.n_best

    def beam_update(self, state, idx):
        positions = self.getCurrentOrigin()
        for e in state:
            # state (c, h)
            a, br, d = e.size()
            # shape (n, m, B, H)
            e = e.view(a, self.size, br // self.size, d)
            # shape (n, m, H)
            sentStates = e[:, :, idx]
            sentStates.data.copy_(sentStates.data.index_select(1, positions))

    def update(self, tensor, idx):
        positions = self.getCurrentOrigin()
        sent = tensor[:, idx]
        sent.data.copy_(sent.data.index_select(0, positions))

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return torch.stack(hyp[::-1]), torch.stack(attn[::-1])

"""
>>> a
tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
        [ 0.8444, -0.4248, -0.3319, -1.9631],
        [ 0.1032,  0.7936,  2.3179,  0.6758]])
>>> a = torch.randn(10, 50)
>>> b = a.view(-1)
>>> b.topk(10, 0, True, True)
(tensor([ 3.3980,  2.9542,  2.7698,  2.4855,  2.2157,  2.1989,  2.0999,
         2.0260,  1.9947,  1.9045]), tensor([ 319,  389,  383,   72,  218,   64,  238,  101,  313,  473]))
>>> s, i = b.topk(10, 0, True, True)
>>> i
tensor([ 319,  389,  383,   72,  218,   64,  238,  101,  313,  473])
>>> k = i / 50
>>> k
tensor([ 6,  7,  7,  1,  4,  1,  4,  2,  6,  9])
>>> i - k * 50
tensor([ 19,  39,  33,  22,  18,  14,  38,   1,  13,  23])

"""