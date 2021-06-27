import pandas as pd
import json
import os
from collections import defaultdict,Counter
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import lstsq
from matplotlib import cm
plt.style.use('seaborn-whitegrid')

DEBUG = False
DATA_HOME = "/Users/keganrabil/Desktop/mtg/data/"

FRENCH_VANILLA = ["deathtouch",
                  "defender",
                  "double strike",
                  "first strike",
                  "flash",
                  "flying",
                  "haste",
                  "hexproof",
                  "indestructible",
                  "lifelink",
                  "menace",
                  #"protection",
                  "reach",
                  "trample",
                  "vigilance"]

class Library:
    def convertName(self,n):
        to_replace = {"-":"~",
                    "û":"u",
                    "ö":"o",
                    "ú":"u",
                    "â":"a",
                    "é":"e",
                    "á":"a",
                    "!":"",
                    "à":"a",
                    "í":"i"
        }
        n = n.lower().strip()
        for k,v in to_replace.items():
            n = n.replace(k,v)
        return n

    def __init__(self):
        with open(os.path.join(DATA_HOME,"legal_cards.json"),"r") as f:
            legal_cards = json.load(f)

        with open(os.path.join(DATA_HOME,"json_list.txt"),"r",encoding="utf-8") as f:
            raw = f.read()
        min_cards = {}
        self.names = []
        self.cards = {}

        enc_cards = raw.split("\n\n")
        for c in enc_cards:
            try:
                j = json.loads(c)
                n = self.convertName(j["name"])
                min_cards[n] = j
                self.names.append(n)
            except Exception as e:
                print(e,c)
                hodor

        for n,lc in legal_cards.items():
            try:
                clean_n = self.convertName(n)
                m = min_cards[clean_n]
                legal_cards[n]["min_text"] = m["text"]
                legal_cards[n]["name"] = n
                legal_cards[n]["min_name"] = clean_n
                self.cards[clean_n] = legal_cards[n]
            except Exception as e:
                if DEBUG:
                    print("unable to find:",n)
                #hodor

    def get(self,n):
         n = self.convertName(n)
         assert n in self.names, "{} not found".format(n)
         return self.cards[n]



    def textQuery(self,query,criteria):
        res = []
        for n,c in self.cards.items():
            valid = True
            for k,v in query.items():
                if c[k] != v:
                    valid = False
                    break
            #CRITERIA
            if "vanilla" in criteria:
                if c["text"] != "":
                    valid = False
            elif "french_vanilla" in criteria:
                # only allow cards tat only have evergreen keywords
                if  c["min_text"] != "": # only check if not vanilla
                    for word in c["min_text"].split("\\"):
                        if word not in FRENCH_VANILLA:
                            valid = False
                            break
            if valid:
                res.append(c)
        return res

COLOR_LOOKUP = {"W":0,"U":1,"B":2,"R":3,"G":4}
def colorIdOnehot(c): # Assume given Color Identity vector
    #      W  U  B  R  G
    ret = np.zeros((5,1)) # all 0's is colorless
    for k,v in COLOR_LOOKUP.items():
        if k in c:
            ret[v,0] = 1.0
    return ret




###
def test():
    not_found = []
    for n,lc in legal_cards.items():
        try:
            if "Chittering":
                print(n,lc)
                print("")
        except Exception as e:
            not_found.append(n)
            #hodor


    # cant find these - i can live with that
    # ['Borrowing 100,000 Arrows', 'Chittering Host', "Guan Yu's 1,000-Li March"]
    print("Could not find: ", len(not_found), not_found)


def main():
    try:
        with open(os.path.join(DATA_HOME,"lib.pkl"),"rb") as f:
            lib = pickle.load(f)
        print(lib)
    except Exception as e:
        print(e,"recreating...")
        lib = Library()
        with open(os.path.join(DATA_HOME,"lib.pkl"),"wb+") as f:
            pickle.dump(lib,f)

            
    for k,v in lib.cards.items():
        print(v["name"],v["min_text"])
    with open(os.path.join(DATA_HOME,"min_lookup.pkl"),"wb+") as f:
        pickle.dump(lib.cards,f)



if __name__ == "__main__":
    main()
