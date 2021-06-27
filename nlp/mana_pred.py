import numpy as np
from collections import Counter
import pickle
import pathlib
import json
import os,sys
HERE = pathlib.Path().absolute().parent.__str__()
sys.path.append(os.path.join(pathlib.Path().absolute().parent,"card_db")) # Hax lol

import caffeinate

#pathlib.Path(__file__).parent.absolute()
import pandas as pd
import init_db
CONN = init_db.CONN
CURSOR = CONN.cursor()

# df = pd.read_sql_query("SELECT * FROM cards LIMIT 1;",CONN)
# print(df.iloc[0])
# for k,v in df.dtypes.items():
#     print(k,v)
# HODOR
TYPES_TAG = "<TYPES>"
RARITY_TAG = "<RARITY>"
TEXT_TAG = "<TEXT>"
POWER_TAG = "<POWER>"
TOUGHNESS_TAG = "<TOUGHNESS>"
START_TAG = "<START>"
PAD_TAG = "<PAD>"
END_TAG = "<END>"
NULL_TAG = "<NULL>"
UNKNOWN_KEY = "<UNK>"
SPACER_KEYS = [UNKNOWN_KEY, START_TAG, END_TAG, TYPES_TAG, RARITY_TAG, TEXT_TAG,PAD_TAG, POWER_TAG, TOUGHNESS_TAG]

ONEHOT_SIZE = 512 #- len(SPACER_KEYS)
MAX_TARGET_LEN = 12 # Progenitus has the longest cmc - 10. We use 12 to account for start/end tags
INDEX_ONLY_INPUT = True
LSTM_SIZE =  32
BATCH_SIZE = 16
EPOCHS = 1000
LEARNING_RATE = 0.0001

def onehotify(word,onehot_lookup,index_only=False):
    try:
        i = onehot_lookup[word]
    except KeyError as e:
        i = UNKNOWN_INDEX
    if index_only:
        return i
    else:
        o = np.zeros(len(onehot_lookup))
        o[i] = 1.0
        return o

# FETCH DATA
ONE_HOT_DATA = "data/mana_pred_data.pkl"
MODEL_INPUT_DATA = "data/model_input_data.pkl"
try:
    with open(ONE_HOT_DATA,"rb") as f:
        df = pickle.load(f)
    print("loaded SQL data")
except:
    print("no pickled SQL data, recreating...")
    df = pd.read_sql_query("""SELECT c.name,
                              c.text as text,
                              c.min_text as min_text,
                              c.rarity,
                              c.convertedManaCost as cmc,
                              c.type,
                              c.types,
                              c.printings,
                              c.power,
                              c.toughness,
                              MIN(s.releaseDate) as first_print,
                              c.manaCost as mana_cost,
                              c.colorIdentity as color_id
                              FROM cards c
                              JOIN legalities l ON (l.uuid = c.uuid AND l.format = "vintage")
                              JOIN sets s ON instr(c.printings, s.code) > 0
                              WHERE s.releaseDate BETWEEN "2003-01-01" AND "2017-01-01"
                              -- AND c.type LIKE "%Creature%"
                              -- AND c.colorIdentity = "B"
                              -- AND c.rarity = "common"
                              GROUP BY c.name;""", CONN)
    print(f"Number of cards found: {len(df)}")
    with open(ONE_HOT_DATA,"wb+") as f:
        pickle.dump(df,f)

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

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

try:
    input_data = pickle_load(MODEL_INPUT_DATA)
    DECODER_TARGET_ARRAY = input_data["decoder_target_array"]
    DECODER_INPUT_ARRAY = input_data["decoder_input_array"]
    ENCODER_INPUT_ARRAY = input_data["encoder_input_array"]

    #ENCODER_INPUT_ARRAY = np.flip(ENCODER_INPUT_ARRAY,axis=1)
    #print("SHAPE:",ENCODER_INPUT_ARRAY.shape)

    ONEHOT_WORD_LOOKUP = input_data["onehot_word_lookup"]
    ONEHOT_INDEX_LOOKUP = input_data["onehot_index_lookup"]
    ONEHOT_COST_LOOKUP= input_data["onehot_cost_lookup"]
    ONEHOT_COST_INDEX_LOOKUP = input_data[ "onehot_cost_index_lookup"]
    MAX_SEQ_LEN = input_data["max_seq_len"]

    print("loaded MODEL INPUT data")
except Exception as e:
    print(e)
    print("no pickled MODEL INPUT data, recreating...")
    # df = pd.read_sql_query("SELECT * FROM cards LIMIT 1;",CONN)
    # print(df.iloc[0])
    # for k,v in df.dtypes.items():
    #     print(k,v)
    # HODOR
    df["cleaned_text"] = ""#df["min_text"]
    df["cleaned_cmc"] = ""#df["cmc"]
    all_words = []
    numbers = set()
    one_hot_costs = set()
    for i,row in df.iterrows():
        if row["min_text"]: # account for lands and shit
            cleaned_words = []
            words = row["min_text"].replace(". "," . ").replace("\\"," ").strip().split(" ")
            for w in words:
                #print(s)
                cleaned_word = w.lower().replace(",","")
                all_words.append(cleaned_word)
                cleaned_words.append(cleaned_word)
            #row["cleaned_text"] = " ".join(cleaned_words)
            df.loc[i, 'cleaned_text'] = " ".join(cleaned_words)

            row["rarity"] = row["rarity"].strip().lower()
            all_words.append(row["rarity"])

            # Power/Toughness
            if row["toughness"]:
                numbers.add(str(row["toughness"]).strip())
            if row["power"]:
                numbers.add(str(row["power"]).strip())

            types = row["types"]
            type_tokens = []
            if "," in types:
                for t in row["types"].split(","):
                    type_tokens.append(t.strip().lower())
            else:
                type_tokens.append(types.strip().lower())
            for t in type_tokens:
                all_words.append(t)
            df.loc[i,"types"] = " ".join(type_tokens)

            # mana costs
            if row["mana_cost"]:
                mc = row["mana_cost"].replace("}{","},{")
                df.loc[i, 'cleaned_cmc'] = mc
                cost_list = mc.split(",") if "," in mc else [mc]
                for c in cost_list:
                    one_hot_costs.add(c)


    one_hot_words = set()
    counter = Counter(all_words)
    for k,v in counter.most_common(ONEHOT_SIZE):
        #print(k,v)
        #if v >= 10:
        one_hot_words.add(k)
    for s in SPACER_KEYS:
        one_hot_words.add(s)
    for n in numbers:
        one_hot_words.add(n)

    one_hot_words = tuple(sorted(one_hot_words))
    print(f"word tokens: {one_hot_words[:10]}")
    ONEHOT_LEN = len(one_hot_words)
    print(f"Number of one-hot tokens: {ONEHOT_LEN}")
    ONEHOT_WORD_LOOKUP = {o:i for i,o in enumerate(one_hot_words)}
    ONEHOT_INDEX_LOOKUP = {i:o for i,o in enumerate(one_hot_words)}
    UNKNOWN_INDEX = ONEHOT_WORD_LOOKUP[UNKNOWN_KEY]
    PAD_INDEX = ONEHOT_WORD_LOOKUP[PAD_TAG]

    one_hot_costs.add(START_TAG)
    one_hot_costs.add(END_TAG)
    one_hot_costs.add(NULL_TAG)
    one_hot_costs.add(PAD_TAG)

    ONEHOT_COST_LOOKUP = {o:i for i,o in enumerate(one_hot_costs)}
    ONEHOT_COST_INDEX_LOOKUP = {i:o for i,o in enumerate(one_hot_costs)}
    ONEHOT_COST_LEN = len(one_hot_costs)
    print(f"Number of one_hot_cost tokens: {ONEHOT_COST_LEN}")

    # for o in one_hot_words:
    #     print(f"{o}, Number of occurrences: {counter[o]}, index: {one_hot_words.index(o)}")
    # print(len(one_hot_words))



    INPUT_TEXT = []
    INPUT_LABELS = []
    #
    ENCODER_INPUT_DATA = []
    DECODER_INPUT_DATA = []
    DECODER_TARGET_DATA = []

    MAX_SEQ_LEN = max([len(row["cleaned_text"].split(" ")) for _i,row in df.iterrows()])
    # for i,_ in enumerate(ENCODER_INPUT_DATA):
    #     #e = ENCODER_INPUT_DATA[i]
    #     while len(ENCODER_INPUT_DATA[i]) < MAX_SEQ_LEN:
    #         ENCODER_INPUT_DATA[i].append(PAD_INDEX)
    MAX_SEQ_LEN = MAX_SEQ_LEN + 14 #  determined ahead of time with above code
    print("MAX SEQUENCE LENGTH: ",MAX_SEQ_LEN)

    for i,row in df.iterrows():
        keys = [START_TAG,TYPES_TAG] + row["types"].split(" ") + [RARITY_TAG,row["rarity"],TEXT_TAG] + row["cleaned_text"].split(" ") #+ [END_TAG]

        if row["power"]:
            keys += [POWER_TAG,row["power"]]
        if row["toughness"]:
            keys += [TOUGHNESS_TAG,row["toughness"]]
        keys += [END_TAG]
        # Padding
        assert len(keys) <= MAX_SEQ_LEN-1, f"Uh Oh: {keys},{len(keys)}"

        if len(keys) < MAX_SEQ_LEN:
            diff = MAX_SEQ_LEN - len(keys)
            keys += ([PAD_TAG] * diff)
        #keys += [END_TAG]

        x = " ".join(keys)
        INPUT_TEXT.append(x)
        y = row["cleaned_cmc"].split(",")
        # Account for lands and cards with no cost like Living End
        if len(y) == 0 or y[0] == "":
            y = [NULL_TAG]
        INPUT_LABELS.append(y)
        y = [START_TAG] + y + [END_TAG]
        if len(y) < (MAX_TARGET_LEN):
            diff = MAX_TARGET_LEN  - len(y)
            y += ([PAD_TAG] * diff)
        assert len(y) == MAX_TARGET_LEN,f"invalid sequnce: {len(y)},{y}"

        tx = [onehotify(w,onehot_lookup=ONEHOT_WORD_LOOKUP,index_only=INDEX_ONLY_INPUT) for w in x.split(" ")]

        ENCODER_INPUT_DATA.append(tx)
        ty = [onehotify(w.strip(),onehot_lookup=ONEHOT_COST_LOOKUP,index_only=INDEX_ONLY_INPUT) for w in y if len(w.strip()) > 0]
        assert len(ty) == MAX_TARGET_LEN,f"invalid sequnce: {len(ty)},{y}"
        # if len(ty) > max_decoder_len:
        #     max_decoder_len = len(ty)
        DECODER_INPUT_DATA.append(ty[:-1])
        DECODER_TARGET_DATA.append(ty[1:])
    #print(f"MAX DECODER SEQUENCE LENGTH: {max_decoder_len}")
    # make sure everything is the right shape
    for e in ENCODER_INPUT_DATA:
        assert len(e) == MAX_SEQ_LEN,f"OH FUCK: Encoder axis 0 :: {len(e)}"
        if not INDEX_ONLY_INPUT:
            for i in e:
                assert len(i) == ONEHOT_LEN,f"oh fuck: Encoder axis 1 ::  {len(i)}"
    # thank god
    ENCODER_INPUT_ARRAY = np.array(ENCODER_INPUT_DATA)
    if INDEX_ONLY_INPUT:
        encoder_input_array_shape = (len(ENCODER_INPUT_DATA),MAX_SEQ_LEN)
    else:
        encoder_input_array_shape = (len(ENCODER_INPUT_DATA),MAX_SEQ_LEN,ONEHOT_LEN)
    ENCODER_INPUT_ARRAY = np.reshape(ENCODER_INPUT_ARRAY,encoder_input_array_shape)
    print(f"Encoder INPUT data formatted: {ENCODER_INPUT_ARRAY.shape}")
    # print("ONHOT TARGET SHAPE")
    # print(ONEHOT_COST_LOOKUP)
    # print(len(ONEHOT_COST_LOOKUP))
    #
    # print("DECODER SHAPE")
    # print(len(DECODER_INPUT_DATA))
    # print(len(DECODER_INPUT_DATA[0]))
    # print(len(DECODER_INPUT_DATA[0][0]))
    # Format the decoder input data - this is
    for d in DECODER_INPUT_DATA:
        assert len(d) == MAX_TARGET_LEN-1,f"OH FUCK: Encoder axis 0 :: {len(d)}, {d}"
        if not INDEX_ONLY_INPUT:
            for i in d:
                assert len(i) == ONEHOT_COST_LEN,f"oh fuck: Encoder axis 1 :: {len(i)}"
    DECODER_INPUT_ARRAY = np.array(DECODER_INPUT_DATA)

    if INDEX_ONLY_INPUT:
        decoder_input_array_shape = (len(ENCODER_INPUT_DATA),MAX_TARGET_LEN-1)
    else:
        decoder_input_array_shape = (len(DECODER_INPUT_DATA),MAX_TARGET_LEN-1,ONEHOT_COST_LEN)

    DECODER_INPUT_ARRAY = np.reshape(DECODER_INPUT_ARRAY,decoder_input_array_shape)
    print(f"Decoder INPUT data formatted: {DECODER_INPUT_ARRAY.shape}")
    # Format the decoder TARGET data
    for d in DECODER_TARGET_DATA:
        assert len(d) == MAX_TARGET_LEN-1,f"OH FUCK: Decoder axis 0 :: {len(d)}, {d}"
        if not INDEX_ONLY_INPUT:
            for i in d:
                assert len(i) == ONEHOT_COST_LEN,f"oh fuck: Decoder axis 1 :: {len(i)}"
    DECODER_TARGET_ARRAY = np.array(DECODER_TARGET_DATA)
    DECODER_TARGET_ARRAY = np.reshape(DECODER_TARGET_ARRAY,decoder_input_array_shape)
    print(f"Decoder TARGET data formatted: {DECODER_TARGET_ARRAY.shape}")

    input_data_dict = {"decoder_target_array":DECODER_TARGET_ARRAY,
                        "decoder_input_array":DECODER_INPUT_ARRAY,
                        "encoder_input_array":ENCODER_INPUT_ARRAY,
                        "onehot_word_lookup": ONEHOT_WORD_LOOKUP,
                        "onehot_index_lookup": ONEHOT_INDEX_LOOKUP,
                        "onehot_cost_lookup":ONEHOT_COST_LOOKUP,
                        "onehot_cost_index_lookup":ONEHOT_COST_INDEX_LOOKUP,
                        "max_seq_len": MAX_SEQ_LEN
    }
    pickle_dump(input_data_dict,MODEL_INPUT_DATA)


UNKNOWN_INDEX = ONEHOT_WORD_LOOKUP[UNKNOWN_KEY]
PAD_INDEX = ONEHOT_WORD_LOOKUP[PAD_TAG]
ONEHOT_COST_LEN = len(ONEHOT_COST_LOOKUP)
print("BEGINNING MODEL")
ENCODER_SIZE = len(ONEHOT_WORD_LOOKUP)
DECODER_SIZE = len(ONEHOT_COST_LOOKUP)


print("ENCODER_INPUT_ARRAY:",ENCODER_INPUT_ARRAY.shape)
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

try:
    model = tf.keras.models.load_model('models/mana_pred.h5')
    print("loaded trained models...")
except Exception as e:
    print(e)
    print("Unable to load models, recreating & training....")
    # Define an input sequence and process it.
    if INDEX_ONLY_INPUT:
        encoder_inputs = Input(shape=(MAX_SEQ_LEN,),name="encoder_inputs")
    else:
        encoder_inputs = Input(shape=(None, ENCODER_SIZE),name="encoder_inputs")
    print("encoder_inputs:",encoder_inputs.shape)
    # Set up the decoder, using `encoder_states` as initial state.
    if INDEX_ONLY_INPUT:
        decoder_inputs = Input(shape=(MAX_TARGET_LEN-1,),name="decoder_inputs")
    else:
        decoder_inputs = Input(shape=(None, DECODER_SIZE),name="decoder_inputs")

    #encoder = LSTM(LSTM_SIZE, return_state=True)
    encoder1 = LSTM(LSTM_SIZE, return_state=True, return_sequences=True,name="encoder1")
    encoder2 = LSTM(LSTM_SIZE, return_state=True, return_sequences=True,name="encoder2")
    encoder3 = LSTM(LSTM_SIZE, return_state=True,name="encoder3")#, return_sequences=False)

    embed_layer = Embedding(input_dim=ENCODER_SIZE, output_dim=LSTM_SIZE)#,input_length=BATCH_SIZE)

    embedding_output = embed_layer(encoder_inputs)

    encoder_hidden_outputs, state_h, state_c = encoder1(embedding_output)
    #encoder_hidden_outputs2, state_h, state_c = encoder2(encoder_hidden_outputs)
    #encoder_outputs, state_h, state_c = encoder3(encoder_hidden_outputs2,initial_state=[hidden_state_h2,hidden_state_c2])
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_embed_layer = Embedding(input_dim=DECODER_SIZE, output_dim=LSTM_SIZE, name="decoder_embed")#,input_length=BATCH_SIZE)
    decoder_lstm1 = LSTM(LSTM_SIZE, return_sequences=True, return_state=True,name="decoder1")
    decoder_lstm2 = LSTM(LSTM_SIZE, return_sequences=True, return_state=True,name="decoder2")
    decoder_lstm3 = LSTM(LSTM_SIZE, return_sequences=True, return_state=True,name="decoder3")#, return_state=False)

    decoder_embedding_output = decoder_embed_layer(decoder_inputs)
    print("decoder_embedding_output:",decoder_embedding_output.shape)
    print("Len encoder_states: ", len(encoder_states))
    print("encoder_states[0]:",encoder_states[0].shape)
    print("encoder_states[1]:",encoder_states[1].shape)
    decoder_hidden_outputs, _, _ = decoder_lstm1(decoder_embedding_output,initial_state=encoder_states)
    #print("decoder_lstm_outputs:",decoder_lstm_outputs.shape)
    decoder_lstm_outputs, _, _ = decoder_lstm2(decoder_hidden_outputs)
    #decoder_outputs, _, _ = decoder_lstm3(decoder_hiddens2,initial_state=[decoder_hidden_state_h2, decoder_hidden_state_c2])
    decoder_dense = Dense(DECODER_SIZE, activation='softmax',name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_lstm_outputs)
    print("decoder_outputs:",decoder_outputs.shape)


    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    opt = Adam(learning_rate=LEARNING_RATE)
    # Run training
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics='accuracy')


    print("decoder_inputs:",decoder_inputs.shape)
    print("DECODER_INPUT_ARRAY:",DECODER_INPUT_ARRAY.shape)
    print("DECODER_TARGET_ARRAY:",DECODER_TARGET_ARRAY.shape)


    model.fit([ENCODER_INPUT_ARRAY, DECODER_INPUT_ARRAY], DECODER_TARGET_ARRAY,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2)

    model.save('models/mana_pred.h5')
#
# print(model.layers)
# print(model.layers[1])
# print(model.layers[1].name)
# print(model.get_layer('encoder1'))
encoder_inputs = model.inputs[0]#"encoder_inputs")
decoder_inputs = model.inputs[1]#("decoder_inputs")
print("decoder_inputs:",decoder_inputs.shape)
encoder_outputs, state_h_enc, state_c_enc = model.get_layer("encoder1").output
encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

decoder_state_input_h = Input(shape=(LSTM_SIZE,), name="decoder_state_input_h")
decoder_state_input_c = Input(shape=(LSTM_SIZE,), name="decoder_state_input_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embed = model.get_layer("decoder_embed")
decoder_lstm1 = model.get_layer("decoder1")
decoder_lstm2 = model.get_layer("decoder2")
#decoder_lstm3 = model.get_layer("decoder3")
#d1,s1h,s1c = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs)
#d2,s2h,s2c = decoder_lstm2(d1,initial_state=[s1h,s1c])
decoder_embedded_inputs = decoder_embed(decoder_inputs)
decoder_hidden_out1, _,_= decoder_lstm1(decoder_embedded_inputs, initial_state=decoder_states_inputs)
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm2(decoder_hidden_out1)
decoder_states = [state_h_dec, state_c_dec]

decoder_dense = model.get_layer("decoder_dense")
decoder_outputs = decoder_dense(decoder_hidden_out1)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)



def decode_sequence(input_seq):
    # Encode the input as state vectors.
    h,c = encoder_model.predict(input_seq)
    states_value = [h,c]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros(( 1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = ONEHOT_COST_LOOKUP[START_TAG]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        k = [target_seq] + states_value
        output_tokens, h, c = decoder_model.predict(k)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :]) # <- ORIGINAL
        #sampled_token_index = np.random.choice([_ for _ in range(ONEHOT_COST_LEN)],p=output_tokens[0, -1, :])

        sampled_char = ONEHOT_COST_INDEX_LOOKUP[sampled_token_index]
        #print(f"sampled_char : {sampled_char}")
        #print(f"        sampled_char: {sampled_char}")
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == END_TAG or len(decoded_sentence) > MAX_TARGET_LEN):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return " ".join(decoded_sentence)

# print("ENCODER_INPUT_ARRAY:",ENCODER_INPUT_ARRAY.shape)
# HODOR
#CHECK_INDEX = 90
for CHECK_INDEX in range(20,30):
    E = ENCODER_INPUT_ARRAY[CHECK_INDEX]
    #print(E.shape)
    print("Encoder input: ")
    if INDEX_ONLY_INPUT:
        print(" ".join([ONEHOT_INDEX_LOOKUP[e] for e in E if ONEHOT_INDEX_LOOKUP[e] != PAD_TAG]))
    else:
        print(" ".join([ONEHOT_INDEX_LOOKUP[np.argmax(e)] for e in E if ONEHOT_INDEX_LOOKUP[np.argmax(e)] != PAD_TAG]))
    # D = DECODER_INPUT_ARRAY[CHECK_INDEX]
    # print("input:")
    # print(" ".join([ONEHOT_COST_INDEX_LOOKUP[np.argmax(d)] for d in D]))
    print(" prediction:")
    print(decode_sequence(np.expand_dims(E,0)))
    D = DECODER_INPUT_ARRAY[CHECK_INDEX]
    print(" real:")
    if INDEX_ONLY_INPUT:
        print(" ".join([ONEHOT_COST_INDEX_LOOKUP[d] for d in D if ONEHOT_INDEX_LOOKUP[d] != PAD_TAG]))
    else:
        print(" ".join([ONEHOT_COST_INDEX_LOOKUP[np.argmax(d)] for d in D]))
    print("")
