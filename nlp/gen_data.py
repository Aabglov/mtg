import numpy as np
import random
import pickle


# TYPES_TAG = "<TYPES>"
# RARITY_TAG = "<RARITY>"
# TEXT_TAG = "<TEXT>"
START_TAG = "<START>"
PAD_TAG = "<PAD>"
END_TAG = "<END>"
# NULL_TAG = "<NULL>"
UNKNOWN_KEY = "<UNK>"
SPACER_KEYS = [UNKNOWN_KEY, START_TAG, END_TAG,PAD_TAG]#, TYPES_TAG, RARITY_TAG, TEXT_TAG,]


LETTERS = "a b c d e f g h i j k l m n o p q r s t u v w x y z".split(" ")
#LETTERS = ["a","b","c","d","e"]
TOKENS = LETTERS + SPACER_KEYS
REVERSE_LOOKUP = {i:v for i,v in enumerate(TOKENS)}
LOOKUP = {v:i for i,v in enumerate(TOKENS)}
UNKNOWN_INDEX = LOOKUP[UNKNOWN_KEY]
DATASET_LENGTH = 1000


def onehotify(c,onehot_lookup):
    o = np.zeros(len(onehot_lookup))
    try:
        i = onehot_lookup[c]
        o[i] = 1.0
    except KeyError as e:
        o[UNKNOWN_INDEX] = 1.0
    return o

def onehotWord(w,onehot_lookup):
    o = []
    for c in w:
        o.append(onehotify(c,onehot_lookup))
    return np.array(o)

def genWord(l=5):
    w = []#[START_TAG]
    for _ in range(l):
        c = random.choice(LETTERS)
        w.append(c)
    #w.append(END_TAG)
    return w

def createFakeData():
    x = []
    y = []
    for _ in range(DATASET_LENGTH):
        w = genWord()
        o = onehotWord(w,LOOKUP)
        x.append(o)
        y.append(np.flip(o,0))
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    encoder_input_data = x
    decoder_input_data = y[:,:-1,:]
    decoder_target_data = y[:,1:,:]
    return encoder_input_data,decoder_input_data,decoder_target_data


#https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
class MacOSFile(object):
    def __init__(self, f):
        self.f = f
    def __getattr__(self, item):
        return getattr(self.f, item)
    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)
    def write(self, buffer):
        n = len(buffer)
        #print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            #print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            #print("done.", flush=True)
            idx += batch_size

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    encoder_input_data,decoder_input_data,decoder_target_data = createFakeData()
    # print(encoder_input_data.shape)
    # print(decoder_input_data.shape)
    # print(decoder_target_data.shape)
    input_data_dict = {"decoder_target_array":decoder_target_data,
                        "decoder_input_array":decoder_input_data,
                        "encoder_input_array":encoder_input_data,
                        "onehot_word_lookup": LOOKUP,
                        "onehot_index_lookup": REVERSE_LOOKUP,
                        "onehot_cost_lookup":LOOKUP,
                        "onehot_cost_index_lookup":REVERSE_LOOKUP
    }
    MODEL_INPUT_DATA = "data/model_input_data.pkl"
    pickle_dump(input_data_dict,MODEL_INPUT_DATA)
