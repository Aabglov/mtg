# https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Python program to generate word vectors using Word2Vec

# importing all necessary modules
import warnings

warnings.filterwarnings(action = 'ignore')

from collections import defaultdict,Counter
import pickle
import pathlib
import json
import os,sys
HERE = pathlib.Path().absolute().parent.__str__()
sys.path.append(os.path.join(pathlib.Path().absolute().parent,"card_db")) # Hax lol

#pathlib.Path(__file__).parent.absolute()
import pandas as pd
import init_db
CONN = init_db.CONN
CURSOR = CONN.cursor()

UNKNOWN_KEY = "UNK"
# FETCH DATA
ONE_HOT_DATA = "onehot_data.pkl"
try:
    with open(ONE_HOT_DATA,"rb") as f:
        df = pickle.load(f)
    print("loaded data")
except:
    print("no pickled data, recreating...")
    df = pd.read_sql_query("""SELECT c.name,
                              c.text as text,
                              c.min_text as min_text,
                              c.rarity,
                              c.convertedManaCost as cmc,
                              c.type,
                              c.types,
                              c.manaCost as mana_cost,
                              c.colorIdentity as color_id
                              FROM cards c
                              JOIN legalities l ON (l.uuid = c.uuid AND l.format = "vintage")
                              JOIN sets s ON instr(c.printings, s.code) > 0
                              WHERE s.releaseDate BETWEEN "2008-01-01" AND "2017-01-01"
                              AND c.type LIKE "%Creature%"
                              AND c.colorIdentity = "B"
                              AND c.rarity = "common"
                              GROUP BY c.name;""", CONN)
    print(f"Number of cards found: {len(df)}")
    # with open(ONE_HOT_DATA,"wb+") as f:
    #     pickle.dump(df,f)

# df = pd.read_sql_query("SELECT * FROM cards LIMIT 1;",CONN)
# print(df.iloc[0])
# for k,v in df.dtypes.items():
#     print(k,v)
# HODOR

all_texts = []
for i,row in df.iterrows():
    if row["min_text"]: # account for lands and shit
        sentences = row["min_text"].split("\\")
        for s in sentences:
            #print(s)
            all_texts.append(s)
one_hot_sentences = set()
counter = Counter(all_texts)
for k,v in counter.most_common(10):
    #print(k,v)
    #if v >= 10:
    one_hot_sentences.add(k)

one_hot_sentences.add(UNKNOWN_KEY)
one_hot_sentences = tuple(sorted(one_hot_sentences))
UNKNOWN_INDEX = one_hot_sentences.index(UNKNOWN_KEY)
for o in one_hot_sentences:
    print(f"{o}, Number of occurrences: {counter[o]}, index: {one_hot_sentences.index(o)}")
print(len(one_hot_sentences))
#print(one_hot_sentences.index("totem armor"))
HODOR

def onehot(sentences):
    oh = [0] * len(one_hot_sentences) # treating the one hot list as global
    for s in sentences:
        try:
            i = one_hot_sentences.index(s)
            oh[i] += 1
        except ValueError:
            oh[UNKNOWN_INDEX] += 1 # This can lead to non-one-hot formatted sentences, but that's okay for now
    return oh


for i, row in df.iterrows():
    if row["min_text"]: # account for lands and shit
        sentences = row["min_text"].split("\\")
        o = onehot(sentences)
        if sum(o) > o[UNKNOWN_INDEX]:
            print(row["name"],row["cmc"], row["mana_cost"])
            print(row["type"],row["types"])
            print(row["power"], row["toughness"])
            print(sentences)
            print(o)
            HODOR
