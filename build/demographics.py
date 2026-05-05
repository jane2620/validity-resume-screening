import json, random
from typing import Tuple, List, Dict


NAMES_PATH = "/home/zs7353/resume-fting/data/names.json"


def load_names():
    with open(NAMES_PATH, "r") as f:
        return json.load(f)


def sample_same_group_pair(names_data) -> Tuple[str, str, str]:
    # names_data structure: { "MEN": {"W":[...], "B":[...], ...}, "WOMEN": {...} }
    gender_key = random.choice(["MEN", "WOMEN"])
    race_key = random.choice(list(names_data[gender_key].keys()))
    group = (race_key, "M" if gender_key == "MEN" else "W")
    choices = names_data[gender_key][race_key]
    a, b = random.sample(choices, 2)
    return a, b, f"{group[0]}_{group[1]}"


def sample_cross_group_pair(names_data) -> Tuple[str, str, str, str]:
    gender_key1 = random.choice(["MEN", "WOMEN"])
    gender_key2 = random.choice([g for g in ["MEN", "WOMEN"] if g != gender_key1 or random.random() < 0.5])
    race_key1 = random.choice(list(names_data[gender_key1].keys()))
    race_key2 = random.choice(list(names_data[gender_key2].keys()))
    name1 = random.choice(names_data[gender_key1][race_key1])
    name2 = random.choice(names_data[gender_key2][race_key2])
    g1 = f"{race_key1}_{'M' if gender_key1=='MEN' else 'W'}"
    g2 = f"{race_key2}_{'M' if gender_key2=='MEN' else 'W'}"
    return name1, name2, g1, g2


def wb_only_groups(names_data) -> Dict[str, List[str]]:
    # Restrict to W/B across MEN/WOMEN
    wb = {}
    for gender_key in ["MEN", "WOMEN"]:
        for race_key in ["W", "B"]:
            key = f"{race_key}_{'M' if gender_key=='MEN' else 'W'}"
            wb[key] = list(names_data[gender_key].get(race_key, []))
    return wb


def enumerate_wb_pairs() -> List[Tuple[str, str]]:
    groups = ["W_M", "W_W", "B_M", "B_W"]
    pairs = []
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            pairs.append((groups[i], groups[j]))
    return pairs


