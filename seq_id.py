

class seq_id(object):

    def __init__(self, idx=0, lock_available=False):
        self.freq_dict_ = {}
        self.T_dict_ = {}
        self.id_dict_ = {}
        self.idx_ = idx

        if lock_available:
            self.T2id('<UNKNOWN>')

    def T2id(self, T, lock=False):

        if T in self.T_dict_.keys():
            id = self.T_dict_[T]
            self.freq_dict_[id] = self.freq_dict_[id] + 1
            return id
        else:
            if lock:
                return self.T_dict_['<UNKNOWN>']

            self.T_dict_[T] = self.idx_
            self.freq_dict_[self.idx_] = 1
            self.id_dict_[self.idx_] = T
            self.idx_ += 1
            return self.idx_ - 1

    def id2T(self, id):
        return self.id_dict_[id]

    def no_entries(self):
        return self.idx_
