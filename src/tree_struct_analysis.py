from posixpath import join
from config import config
from os import path
from workload_generation import parse_sst
from collections import Counter
import pdb

def generate_parsed_sst(sent_index_range=None):
    # index range: [lower, higher)
    raw_sst_file = path.join(config["SST_PATH"], "dictionary.txt")
    parse_sst(raw_sst_file, config["out_dir"], sent_index_range)


def analyze_parsed_sst(parsed_file):
    tag_collection = Counter()
    parsed_sst = open(parsed_file, "r")

    for line in parse_sst:
        words = line.strip().split()
        tag = words[0].remove("(")
        tag_collection[tag] += 1

    # remap the label collections to 5 new labels
    label_mapping = {0: [], 1: [], 2: [], 3: [], 4: []}
    remapped_count = Counter() # occurance counts of the remapped 5 labels
    sorted_tags = tag_collection.most_common()
    insert_pt = 0
    for item in sorted_tags:
        tag, count = item
        label_mapping[insert_pt].append(tag)
        remapped_count[insert_pt] += count
        insert_pt += 1
        insert_pt %= 5

    # pdb.set_trace()
    return label_mapping, remapped_count


if __name__ == "__main__":
    # generate_parsed_sst((200000, 300000))
    analyze_parsed_sst()