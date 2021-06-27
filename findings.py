import cr_colorless_fv
from nlp.load_cards import Library
lib = Library()

COST = cr_colorless_fv.main()


def calc(d):
    total = 0
    for i,k in enumerate(COST.keys()):
        total += COST[k] * d[i]
    return total

# uncommon vanilla modern legal
# Memnite
print("uncommon vanilla modern legal")
test_unc_van = ["memnite",
                "glass golem",
                "phyrexian hulk",
                "stone golem"  # downgraded in 2020
                ]
for name in test_unc_van:
    c = lib.get(name)
    a,b = cr_colorless_fv.oneHot(c)
    print(name,calc(a),b)


# uncommon french vanilla modern legal
print("")
print("uncommon french vanilla modern legal")

test_unc_frvan = ["anvilwrought raptor",
                    "arachnoid",
                    "dancing scimitar",
                    "gold~forged sentinel",
                    "haunted guardian",
                    "lumengrid gargoyle",
                    "ornithopter",
                    "pilgrim of the fires",
                    "scion of ugin",
                    "wall of spears"
]

for name in test_unc_frvan:
    c = lib.get(name)
    a,b = cr_colorless_fv.oneHot(c)
    print(name,calc(a),b)


#### tests
print("\nModular")
modular = {"arcbound bruiser":3.,
            "arcbound stinger":1.,
            "arcbound worker":1.,
            "arcbound lancer":4
}

for name,m in modular.items():
    c = lib.get(name)
    a,b = cr_colorless_fv.oneHot(c)
    # overwrite mod vals
    a[0] = m
    a[1] = m
    print(name,calc(a),b)

print("\nHaste")
haste = ["snare thopter"]
for name in haste:
    c = lib.get(name)
    a,b = cr_colorless_fv.oneHot(c)
    print(name,calc(a),b)

print("\nDraw a card")
# Skyscanner
draw_card = ["skyscanner"]
for name in draw_card:
    c = lib.get(name)
    a,b = cr_colorless_fv.oneHot(c)
    print(name,calc(a),b)

print("\nIndestructible") # looking like 2
# Darksteel Myr
indestructible = ["darksteel myr","darksteel gargoyle"]
for name in indestructible:
    c = lib.get(name)
    a,b = cr_colorless_fv.oneHot(c)
    print(name,calc(a),b)



#
# name = " "
# card = {
#         "power":,
#         "toughness":,
#         "flying":0,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":1
# }
# real_cost =
# print(name,"pred:",calc(card),"true:",real_cost)




# name = "Memnite"
# card = {
#         "power":1,
#         "toughness":1,
#         "flying":0,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":1
# }
# real_cost = 0
# print(name,"pred:",calc(card),"true:",real_cost)
#
# name = "Glass Golem"
# card = {
#         "power":6,
#         "toughness":2,
#         "flying":0,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 5
# print(name,"pred:",calc(card),"true:",real_cost)
#
# name = "Phyrexian Hulk"
# card = {
#         "power":5,
#         "toughness":4,
#         "flying":0,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 6
# print(name,"pred:",calc(card),"true:",real_cost)

# name = "Stone Golem"
# card = {
#         "power":4,
#         "toughness":4,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 5
# print(name,"pred:",calc(card),"true:",real_cost)


# # Anvilwrought Raptor
# name = "Anvilwrought Raptor"
# card = {
#         "power":2,
#         "toughness":1,
#         "flying":1,
#         "first strike":1,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 4
# print(name,"pred:",calc(card),"true:",real_cost)
#
#
# # Arachnoid
# name = "Arachnoid"
# card = {
#         "power":2,
#         "toughness":6,
#         "flying":0,
#         "first strike":0,
#         "defender":0,
#         "reach":1,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 4
# print(name,"pred:",calc(card),"true:",real_cost)
#
# # Dancing Scimitar
# name = "Dancing Scimitar"
# card = {
#         "power":1,
#         "toughness":5,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 4
# print(name,"pred:",calc(card),"true:",real_cost)
#
# # Gold-Forged Sentinel
# name = "Gold-Forged Sentinel"
# card = {
#         "power":4,
#         "toughness":4,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 6
# print(name,"pred:",calc(card),"true:",real_cost)
#
# # Haunted Guardian
# name = "Haunted Guardian"
# card = {
#         "power":2,
#         "toughness":1,
#         "flying":0,
#         "first strike":1,
#         "defender":1,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 2
# print(name,"pred:",calc(card),"true:",real_cost)
#
# # Lumengrid Gargoyle
# name = "Lumengrid Gargoyle"
# card = {
#         "power":4,
#         "toughness":4,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 6
# print(name,"pred:",calc(card),"true:",real_cost)
#
# # Ornithopter
# name = "Ornithopter"
# card = {
#         "power":0,
#         "toughness":2,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0, # note: much closer to memnite if costed as reach instead of flying
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 0
# print(name,"pred:",calc(card),"true:",real_cost)
#
# #  Pilgrim of the Fires
# name = "Pilgrim of the Fires"
# card = {
#         "power":6,
#         "toughness":4,
#         "flying":0,
#         "first strike":1,
#         "defender":0,
#         "reach":0,
#         "trample":1,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 7
# print(name,"pred:",calc(card),"true:",real_cost)
#
# #  Scion of Ugin
# name = "Scion of Ugin "
# card = {
#         "power":4,
#         "toughness":4,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 6
# print(name,"pred:",calc(card),"true:",real_cost)
#
# # Wall of Spears
# name = "Wall of Spears"
# card = {
#         "power":2,
#         "toughness":3,
#         "flying":0,
#         "first strike":1,
#         "defender":1,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 3
# print(name,"pred:",calc(card),"true:",real_cost)

# name = "Arcbound Bruiser"
# card = {
#         "power":3,
#         "toughness":3,
#         # "modular": 3,
#         "flying":0,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 5
# print(name,"pred:",calc(card),"true:",real_cost)
#
# # name = "Arcbound Hybrid"
# # card = {
# #         "power":2,
# #         "toughness":2,
# #         # "modular": 2,
# #         #"haste":1,
# #         "flying":0,
# #         "first strike":0,
# #         "defender":0,
# #         "reach":0,
# #         "trample":0,
# #         "improvise/convoke":0,
# #         "vigilance":0,
# #         "card":-1
# # }
# # real_cost = 4
# # print(name,"pred:",calc(card),"true:",real_cost)
#
# name = "Arcbound Stinger"
# card = {
#         "power":1,
#         "toughness":1,
#         # "modular": 1,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 2
# print(name,"pred:",calc(card),"true:",real_cost)
#
# name = "Arcbound Worker"
# card = {
#         "power":1,
#         "toughness":1,
#         # "modular": 1,
#         "flying":0,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 1
# print(name,"pred:",calc(card),"true:",real_cost)
#
# name = "Arcbound Lancer"
# card = {
#         "power":4,
#         "toughness":4,
#         # "modular": 4,
#         "flying":0,
#         "first strike":1,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 7
# print(name,"pred:",calc(card),"true:",real_cost)

# Snare Thopter
# name = "Snare Thopter"
# card = {
#         "power":3,
#         "toughness":2,
#         #"haste":1,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 4
# print(name,"pred:",calc(card),"true:",real_cost)

# card = {
#         "power":1,
#         "toughness":1,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1 # draws when enters??
# }
# real_cost = 3
# print(name,"pred:",calc(card),"true:",real_cost)

# card = {
#         "power":0,
#         "toughness":1,
#         #"indestructible":1,
#         "flying":0,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 3
# print(name,"pred:",calc(card),"true:",real_cost)
#
# # Darksteel Gargoyle
# name = "Darksteel Gargoyle"
# card = {
#         "power":3,
#         "toughness":3,
#         #"indestructible":1,
#         "flying":1,
#         "first strike":0,
#         "defender":0,
#         "reach":0,
#         "trample":0,
#         "improvise/convoke":0,
#         "vigilance":0,
#         "card":-1
# }
# real_cost = 7
# print(name,"pred:",calc(card),"true:",real_cost)
