import os,sys,json

TIERS = ["Primary:","Secondary:","Tertiary:"]


def cleanName(n):
    if "(" in n:
        return n.split("(")[0].replace("\"","").lower().strip(),n.split("(")[1].replace(")","").lower().strip()
    else:
        return n.replace("\"","").lower().strip(),None

def parseColors(l,term):
    # This doc seems to use both versions of oxford comma - fml
    raw = l.replace(term,"").lower().replace(", and",", ").replace(" and ",", ").strip()
    ret = [c.lower().strip() for c in raw.split(",")]
    return ret

with open("abilities.txt","r") as f:
    ability_list = f.read().split("\n\n")

json_list = []
for ability in ability_list:
    lines = ability.split("\n")
    name,exp = cleanName(lines[0])
    p = s = t = None
    for l in lines:
        if "Primary:" in l:
            p = parseColors(l,"Primary:")
        elif "Secondary:" in l:
            s = parseColors(l,"Secondary:")
        if "Tertiary:" in l:
            t = parseColors(l,"Tertiary:")
    com = lines[-1]
    #print(name, p, s, t)
    json_list.append({"name":name,"explanation":exp,"primary":p,"secondary":s,"tertiary":t,"comment":com,"value":0,"include":True})

with open("ability.json","w+") as f:
    json.dump(json_list,f)


for j in json_list:
    if "white" in j["primary"] and len(j["primary"]) == 1:
        print(j)
        print("")
