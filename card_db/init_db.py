import sqlite3
import pandas as pd
import numpy as np
from scipy.linalg import lstsq
import json
import os,pickle

CONN = sqlite3.connect('../data/AllPrintings.sqlite')


####################################################################################################
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
                  "vigilance",
                  "exalted",
                  "changeling",
                  "convoke",
                  "morph",
                  #"megamorph",
                  #"mentor",
                  #"infect",
                  "cycling"]


def convertName(n):
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

def updateMinText(conn, t, n):
    try:
        sql = f'''UPDATE cards
                  SET min_text = ?
                  WHERE name = ?;'''
        cur = conn.cursor()
        cur.execute(sql, (t,n))
        conn.commit()
    except Exception as e:
        print("ERROR:",e)

def updateVanilla(conn, v, f, n):
    try:
        sql = f'''UPDATE cards
                  SET vanilla = ?,
                      french_vanilla = ?
                  WHERE name = ?;'''
        cur = conn.cursor()
        cur.execute(sql, (v,f,n))
        conn.commit()
    except Exception as e:
        print("ERROR:",e)


def ADD_VAN():
    #ADD_VAN = False
    #if ADD_VAN:
    # with open("/Users/keganrabil/Desktop/mtg/data/json_list.txt","r",encoding="utf-8") as f:
    #     raw = f.read()
    # min_cards = {}
    # enc_cards = raw.split("\n\n")
    # for c in enc_cards:
    #     try:
    #         j = json.loads(c)
    #         n = convertName(j["name"])
    #         min_cards[n] = j
    #     except Exception as e:
    #         print(e,c)
    #         hodor
    #
    # DATA_HOME = "/Users/keganrabil/Desktop/mtg/data/"
    # with open(os.path.join(DATA_HOME,"min_lookup.pkl"),"rb") as f:
    #     lookup = pickle.load(f)
    # for _,c in lookup.items():
    #     n = c["name"]
    #     t = c["min_text"]
    #     print(n,t)
    #     van = False
    #     fvan = True
    #     if len(t) == 0:
    #         van = True
    #     else:
    #         for word in t.split("\\"):
    #             if word not in FRENCH_VANILLA:
    #                 fvan = False
    #                 break
    #     updateVanilla(CONN, van, fvan, n)


    df = pd.read_sql_query("SELECT * FROM cards c WHERE c.types LIKE '%Creature%'", CONN)
    for i, row in df.iterrows():
        if row["min_text"] is not None and \
           row["keywords"] is not None and \
           "\\" in row["min_text"] and \
           "," in row["keywords"]:

            keywords = set([k.lower().strip() for k in row["keywords"].split(",")])
            card_text = set([k.lower().strip() for k in row["min_text"].split("\\")])
            if keywords == card_text:
                # print(keywords)
                # print(card_text)
                print(row["name"],row["uuid"], row["keywords"], row["text"], row["min_text"])
                # hodor
                updateVanilla(CONN, False, True, row["name"])
        elif row["min_text"] is None or row["min_text"] == "":
            updateVanilla(CONN, True, False, row["name"])

    hodot
    ####################################################################################################

def oneHot(c,keyword_list=FRENCH_VANILLA):
    p = c["power"]
    t = c["toughness"]
    text = c["keywords"].lower().strip().split(",") if c["keywords"] else []
    keywords = []
    for word in keyword_list:
        if word in text:
            keywords.append(1.)
        else:
            keywords.append(0.)
    return [p,t] + keywords , c["convertedManaCost"]# basis

def doFit(A,b):
    # do fit
    fit, residual, rnk, s = lstsq(A, b)
    return fit

# c.execute("ALTER TABLE cards ADD COLUMN french_vanilla bool;")
# c.execute("ALTER TABLE cards ADD COLUMN vanilla bool;")

#c.execute("select name from sqlite_master where type = 'table';")
#c.execute("SELECT * FROM sets;")
# k = pd.read_sql_query("SELECT * FROM sets;",CONN)
# print(k.columns.values)
# print(k.iloc[0])
# hodor

# df = pd.read_sql_query("select * from cards;", CONN)
# print(df.columns.values)
# print(df.iloc[0])
# hodor

# c.execute("SELECT s.code,s.name,s.releaseDate FROM sets s WHERE s.releaseDate BETWEEN '2000-01-01' AND '2001-01-01'")
# print(c.fetchall())
# hodot

def main():
    years = ["2001-01-01",
             #"2002-01-01",
             "2003-01-01",
             #"2004-01-01",
             "2005-01-01",
             #"2006-01-01",
             "2007-01-01",
             #"2008-01-01",
             "2009-01-01",
             #"2010-01-01",
             "2011-01-01",
             #"2012-01-01",
             "2013-01-01",
             #"2014-01-01",
             "2015-01-01",
             #"2016-01-01",
             "2017-01-01",
             #"2018-01-01",
             "2019-01-01",
             "2020-01-01"
    ]
    power_coeffs = {}
    toughness_coeffs = {}
    for y, next in zip(years[:-1], years[1:]):
        print(y,"to", next)
        df = pd.read_sql_query("""SELECT c.*,
                                GROUP_CONCAT(DISTINCT l.format) as legality
                                FROM cards c
                                JOIN legalities l ON l.uuid = c.uuid
                                JOIN sets s ON instr(c.printings, s.code) > 0
                                WHERE (c.french_vanilla = True OR c.vanilla = True)
                                AND c.colors = 'R'
                                AND c.rarity = "common"
                                AND c.types LIKE '%Creature%'
                                AND s.releaseDate BETWEEN "{}" AND "{}"
                                GROUP BY c.name
                                HAVING legality LIKE "%modern%";""".format(y,next), CONN)
        print("    Records found:",len(df))
        #print(df.columns.values)
        keywords_raw = df.keywords.unique()
        keywords = set()
        for k in keywords_raw:
            if k is not None:
                card_keywords = k.split(",")
                for c in card_keywords:
                    keywords.add(c.lower().strip())
        print("    Keywords:",keywords)

        A = []
        B = []
        for i,row in df.iterrows():
            # if y == "2006-01-01":
            #     print(row["name"],
            #             #row["colors"],
            #             #row["types"],
            #             row["rarity"],
            #             #row["text"],
            #             row["power"],
            #             row["toughness"],
            #             row["min_text"],
            #             row["keywords"],
            #             row["printings"]
            #             #row["uuid"]
            #             #row["legality"]
            #             )
            #     print("")
            a,b = oneHot(row,keyword_list=keywords)
            #print(name,c["text"])
            A.append(a)
            B.append(b)


        attr = ["power", "toughness"] + list(keywords)
        A = np.array(A)
        B = np.array(B)
        coeffs = doFit(A,B)
        cost = {}
        for k,c in zip(attr,coeffs):
            cost[k] = c
            #print("       ",k,c)
        power_coeffs[y] = cost["power"]
        toughness_coeffs[y] = cost["toughness"]

    y_ind = years[:-1]
    for y in y_ind:
        print(y,power_coeffs[y],toughness_coeffs[y])

    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.style.use('seaborn-whitegrid')

    plt.plot(y_ind, list(power_coeffs.values()))
    plt.plot(y_ind, list(toughness_coeffs.values()))
    plt.show()
