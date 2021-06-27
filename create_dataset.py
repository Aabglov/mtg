import pandas as pd
import json


with open("data/AtomicCards.json") as f:
    atomic_raw = json.load(f)
card_data = atomic_raw["data"]

with open("data/SetList.json") as f:
    setlist = json.load(f)
sets = setlist["data"]
set_lookup = {s["code"]: s["releaseDate"] for s in sets}

with open("data/AllPrintings.json") as f:
    print_raw = json.load(f)
print_data = print_raw["data"]

# name = "Liliana of the Veil"
# card = card_data[name][0]
# print(card["name"])
# print(card["convertedManaCost"])
# print(card["colorIdentity"])
# #print(card["manaCost"])
# print(card["type"])
# print(card["types"])
# print(card["supertypes"])
# print(card["subtypes"])
# print(card["text"])
# print(card["legalities"])
# print(card["printings"])
#
#
# isd = print_data["ISD"]["cards"]
# isd_card = [c for c in isd if c["name"] == name]
# print(isd_card)

card_dict = {}
for name,card_list in card_data.items():
    include = False

    card = card_list[0] # these names are unique so their card list is always 1 element - makes you wonder why it's a list at all
    cmc = float(card["convertedManaCost"])
    mc = card["manaCost"] if "manaCost" in card else -1
    color_id = card["colorIdentity"]
    #type = card["type"]
    types = card["types"]
    #supert = card["supertypes"]
    #subt = card["subtypes"]
    legal = card["legalities"]
    sets = card["printings"]
    if "power" in card:
        try:
            power = float(card["power"])
        except ValueError:
            power = card["power"]
    else:
        power = -1
    if "toughness" in card:
        try:
            toughness = float(card["toughness"])
        except ValueError:
            toughness = card["toughness"]
    else:
        toughness = -1
    # Not all cards have text - eg: vanilla creatures
    if "text" not in card:
        text = ""
    else:
        text = card["text"]

    if "commander" in legal and legal["commander"] == "Legal":
        include = True

    if include:
        # Determine the first printing, fist set and date of first printing
        print_dates = [set_lookup[s] for s in sets]
        if len(print_dates) > 1:
            first_date = min(print_dates)
            first_set_index = print_dates.index(first_date)
        else:
            first_set_index = 0
            first_date = print_dates[first_set_index]
        first_set = sets[first_set_index]
        # Get all cards in initial set
        first_set_cards = print_data[first_set]["cards"]
        # Find the matching card (same name as atomic card)
        first_printing = [c for c in first_set_cards if c["name"] == name][0]
        # take the rarity from the matching print
        rarity = first_printing["rarity"]

        card_dict[name] = {"cmc":cmc,
                           "mana_cost": mc,
                           "color_id": color_id,
                           "types": types,
                           "first_print": first_date,
                           "rarity": rarity,
                           "text": text,
                           "power": power,
                           "toughness": toughness}

        if len(card_dict) % 100 == 0:
            print(f"Processing, {len(card_dict)}")



with open("data/legal_cards.json","w+") as f:
    json.dump(card_dict,f)

# command to make minified text set
# python encode.py -v -s -e=named ../data/AllPrintings.json ../data/output.txt
