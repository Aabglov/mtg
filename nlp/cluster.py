# https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

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

# df = pd.read_sql_query("SELECT * FROM cards c WHERE c.types LIKE '%Creature%'", init_db.CONN)
# print(df.head(5))
# hodor


# literally just sql inject atk
def query(sql):
    try:
        CURSOR.execute(sql)
        CONN.commit()
    except Exception as e:
        print("ERROR:",e)
print('start')

try:
    with open("data.pkl","rb") as f:
        data = pickle.load(f)
    print("loaded data")
except:
    print("no pickled data, recreating...")
    # print(pd.read_sql_query("SELECT * FROM legalities",CONN).head(5))
    # HODOR
    # The requirement of vintage is mostly to eliminate un cards, ante and weird shit like Shaharazzad
    # GROUP_CONCAT(DISTINCT l.format) as legality
    # HAVING legality LIKE "%vintage%"
    df = pd.read_sql_query("""SELECT c.name,
                              c.min_text as text
                              FROM cards c
                              JOIN legalities l ON (l.uuid = c.uuid AND l.format = "vintage")
                              JOIN sets s ON instr(c.printings, s.code) > 0
                              WHERE s.releaseDate BETWEEN "2001-01-01" AND "2017-01-01"
                              GROUP BY c.name;""", CONN)
    print(len(df))
    print(df.head(5))
    print('end')
    data = []

    def parenthesisMustDie(t):
        if "(" in t and ")" in t:
            start = t.index("(")
            end = t.index(")")
            return t[:start] + t[end+1:] # without +1 we include the )
        else:
            HODOR

    REMOVE_CHAR = ["[","]","\"","'s"]
    for i,r in df.iterrows():
        if r["text"]:
            p = r["text"]
            p = p.replace("{","<")
            p = p.replace("}",">")
            # fix mana symbols
            p = p.replace("G"," {g} ")
            p = p.replace("W"," {w} ")
            p = p.replace("U"," {u} ")
            p = p.replace("B"," {b} ")
            p = p.replace("R"," {r} ")
            # now we can safely lower-ize
            p = p.lower()
            # special cases
            p = p.replace("~"," ~ ")
            p = p.replace("\n"," ")
            p = p.replace(". ", " . ")
            p = p.replace(": "," : ")
            p = p.replace("^"," ^ ")
            p = p.replace("\\",",") # just remove that shit
            for c in REMOVE_CHAR:
                p = p.replace(c,"")
            # reminder text - remove everything between parenthesis
            if "(" in p and ")" in p:
                if p.index("(") == 0 and p[-1] == ")": # all text is reminder: ie, lands
                    p = p[1:-1]
                else:
                    # cover the case of multiple - or god forbid, nested - reminder text
                    while "(" in p and ")" in p:
                        p = parenthesisMustDie(p)
            # NO INSTANCES OF BELOW
            # elif "(" in p and ")" not in p:
            #     print(p)
            #     print(")")
            #     HODOR
            # elif "(" not in p and ")" in p:
            #     print(p)
            #     print("(")
            #     HODOR

            # mana symbol split
            p = p.replace("}{","} {")
            # commas
            p = p.replace(","," ")
            word_list = p.strip().split(" ")#word_tokenize(text)
            prepared = [w if w == "." else w.replace(".","") for w in word_list] # strip periods from remaining cards
            prepared = [p for p in prepared if p != ""]
            # done
            #print(prepared)
            #print("")
            data.append(prepared)
    with open("data.pkl","wb+") as f:
        pickle.dump(data,f)
# print(data[:100])
# hodor
CORRECT_PAIRINGS = {"{w}":["{u}","{g}"], # BANT
                    "{u}":["{b}","{w}"], # ESPER
                    "{b}":["{u}","{r}"], # GRIXIS
                    "{r}":["{b}","{g}"], # JUND
                    "{g}":["{w}","{r}"] # NAYA
}

def testModelConfig():
    # Create CBOW model
    min_count_range = [1,2,4,8,16,32,64,128]
    size_range = [10,25,50,100,150,200]
    window_range = [1,2,3,4,5]
    # 8 - 50 - 1 is the chosen arrangement - note: all the best models had {g} closest to {r} which is right, but also {b}.
    # Legit - 3 different models did this and that was only mistake they mad
    with open("results.txt","w+") as f:
        for min_count in min_count_range:
            for size in size_range:
                for window in window_range:
                    f.write(f"\nmin_count: {min_count}, size: {size}, window: {window}\n")
                    print(f"min_count: {min_count}, size: {size}, window: {window}")
                    model1 = gensim.models.Word2Vec(data,
                                                    min_count = min_count,
                                                    size = size,
                                                    window = window,
                                                    workers=4)#, sg = 1
                    vocab = model1.wv.vocab.keys()
                    #assert "{w}" in vocab,"Well fuck me"

                    mana = ["{w}","{u}","{b}","{r}","{g}"]
                    score = 0
                    for m in mana:
                        sim_dict = {}
                        for n in mana:
                            if m != n:
                                sim = model1.similarity(m, n)
                                sim_dict[n] = sim
                                #print(f"Cosine similarity between '{m}' and '{n}' - CBOW : ",sim)

                        # Using min() + list comprehension + values()
                        # Finding min value keys in dictionary
                        conns = list(sim_dict.values())
                        strongest_conn = max(conns)
                        c1 = [key for key in sim_dict if sim_dict[key] == strongest_conn][0]
                        conns.remove(strongest_conn)
                        second_conn = max(conns)
                        c2 = [key for key in sim_dict if sim_dict[key] == second_conn][0]

                        if c1 in CORRECT_PAIRINGS[m]:
                            score += 1
                        if c2 in CORRECT_PAIRINGS[m]:
                            score += 1

                        sort_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1])}
                        msg = ", ".join([f"{k}:{v}" for k,v in sort_sim_dict.items()])
                        f.write(f"    {m}: {msg})\n")
                        print(f"    {m}: Favorite pairs: {c1} ({strongest_conn}), {c2} ({second_conn})")
                    f.write(f"score: {score}")
testModelConfig()
# model1 = gensim.models.Word2Vec(data,
#                                 min_count = 8,
#                                 size = 50,
#                                 window = 1,
#                                 workers=4)#, sg = 1
# vocab = list(model1.wv.vocab.keys())
# vocab2 = [v for v in vocab]
# for v1 in vocab:
#     vocab2.remove(v1)
#     for v2 in vocab2:
#         s = model1.similarity(v1,v2)
#         if s >= 0.95:
#             print(v1,v2,s)
