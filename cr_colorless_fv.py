import numpy as np
from scipy.linalg import lstsq
from nlp.load_cards import Library
lib = Library()

KEYWORDS = ["flying",
"first strike",
"defender",
"reach",
"trample",
"improvise",
"convoke",
"vigilance"]

def oneHot(c):
    p = c["power"]
    t = c["toughness"]
    text = c["min_text"]
    keywords = []
    for word in KEYWORDS:
        if word in text:
            keywords.append(1.)
        else:
            keywords.append(0.)
    return [p,t] + keywords , c["cmc"]# basis

def doFit(A,b):
    # do fit

    fit, residual, rnk, s = lstsq(A, b)
    return fit

def main():
    A = []
    B = []

    # common only
    colorless_vanilla = ["alpha myr",
                        "bronze sable",
                        "field creeper",
                        "gilded sentinel",
                        "hexplate golem",
                        "metallic sliver",
                        "obsianus golem",
                        "omega myr",
                        "phyrexian hulk",
                        "phyrexian walker",
                        "prizefighter construct",
                        "razorfield thresher",
                        "sliver construct",
                        #"stone golem",
                        "stonework puma",
                        "venser's sliver",
                        "wicker witch"
    ]

    colorless_frenchvailla = ["coiled tinviper",
                                "consulate skygate",
                                "ebony rhino",
                                "eldrazi devastator",
                                "foundry assembler",
                                "guardians of meletis",
                                "hovermyr",
                                "millennial gargoyle",
                                "pardic wanderer",
                                "scion of ugin",
                                "steel wall",
                                "will~forged golem",
                                "yotian soldier"
    ]

    data = colorless_vanilla+colorless_frenchvailla

    for name in data:
        c = lib.get(name)
        a,b = oneHot(c)
        #print(name,a,b)
        A.append(a)
        B.append(b)

    attr = ["power", "toughness"] + KEYWORDS
    A = np.array(A)
    B = np.array(B)
    coeffs = doFit(A,B)
    cost = {}
    for k,c in zip(attr,coeffs):
        cost[k] = c
        #print(k,c)
    return cost


if __name__ == "__main__":
    main()


# VANILLA
# # Alpha Myr - 2/1 - 2
# A.append([2, 1, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(2)
#
# #Bronze Sable- 2/1 - 2
# A.append([2, 1, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(2)
#
# # Field Creeper - 2/1 - 2
# A.append([2, 1, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(2)
#
# # Gilded Sentinel  - 3/3 - 4
# A.append([3, 3, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(4)
#
# #  Hexplate Golem - 5/7 - 7
# A.append([5, 7, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(7)

# #  Metallic Sliver - 1/1 - 1
# A.append([1, 1, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(1)
#
# # Obsianus Golem - 4/6 - 6
# A.append([4, 6, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(6)
#
# #  Omega Myr - 1/2 - 2
# A.append([1, 2, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(2)

# #  Phyrexian Hulk - 5/4 - 6
# A.append([5, 4, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(6)
#
# #  Phyrexian Walker  - 0/3 - 0
# A.append([0, 3, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(0)
#
# #  Prizefighter Construct - 6/2 - 5
# A.append([6, 2, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(5)

#  Razorfield Thresher  - 6/4 - 7
# A.append([6, 4, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(7)
#
# #  Sliver Construct - 2/2 - 3
# A.append([2, 2, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(3)
#
# #  Stone Golem - 4/4 - 5
# A.append([4, 4, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(5)
#
# #  Stonework Puma - 2/2 - 3
# A.append([2, 2, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(3)
#
# # Venser's Sliver  - 3/3 - 5
# A.append([3, 3, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(5)
#
# #  Wicker Witch - 3/1 - 3
# A.append([3, 1, 0, 0, 0, 0, 0, 0, 0, -1])
# b.append(3)


# FRENCH VANILLA
# Anvilwrought Raptor 2/1 - fly, fs for 4
# removed - only downgraded to common in 2020 - excellent power creep example
# A.append([2, 1, 1, 1, 0, 0, 0, 0, 0, -1])
# b.append(4)

# #Coiled Tinviper - 2/1 fs - 3
# A.append([2, 1, 0, 1, 0, 0, 0, 0, 0, -1])
# b.append(3)
#
# #Consulate Skygate - 0/4 def, reach - 2
# A.append([0, 4, 0, 0, 1, 1, 0, 0, 0, -1])
# b.append(2)
#
# # Ebony Rhino - 4/5 tra- 7
# A.append([4, 5, 0, 0, 0, 0, 1, 0, 0, -1])
# b.append(7)
#
# # Eldrazi Devastator - 8/9 tra - 8
# A.append([8, 9, 0, 0, 0, 0, 1, 0, 0, -1])
# b.append(8)
#
# # Foundry Assembler - 3/3 imp - 5
# A.append([3, 3, 0, 0, 0, 0, 0, 1, 0, -1])
# b.append(5)
#
# # Guardians of Meletis - 0/6 def - 3
# A.append([0, 6, 0, 0, 1, 0, 0, 0, 0, -1])
# b.append(3)
#
# # # Hovermyr - 2/1 fly, vig - 2 ***
# # A.append([2, 1, 1, 0, 0, 0, 0, 0, 1, -1])
# # b.append(2)
#
# # Millennial Gargoyle - 2/2 fly - 4
# A.append([2, 2, 1, 0, 0, 0, 0, 0, 0, -1])
# b.append(4)
#
# # Ornithopter - 0/2 fly - 0
# # only printed at common once - most recent print 2020 at uncommon
# # A.append([0, 2, 1, 0, 0, 0, 0, 0, 0, -1])
# # b.append(0)
#
# # Pardic Wanderer - 5/5 tra - 6
# A.append([5, 5, 0, 0, 0, 0, 1, 0, 0, -1])
# b.append(6)
#
# # Scion of Ugin - 4/4 fly - 6
# A.append([4, 4, 1, 0, 0, 0, 0, 0, 0, -1])
# b.append(6)
#
# # Steel Wall - 0/4 def - 1
# A.append([0, 4, 0, 0, 1, 0, 0, 0, 0, -1])
# b.append(1)
#
# # Will-Forged Golem - 4/4 imp - 6
# A.append([4, 4, 0, 0, 0, 0, 0, 1, 0, -1])
# b.append(6)
#
# # Yotian Soldier - 1/4 vig - 3
# A.append([1, 4, 0, 0, 0, 0, 0, 0, 1, -1])
# b.append(3)
