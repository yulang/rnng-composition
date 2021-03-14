from random import randint
from os import path

DATA_FOLDER = ""

config = {
    BIRD_PATH = "/home-nfs/langyu/data_folder/BiRD/BiRD.txt"
    PPDB_PATH = "/home-nfs/langyu/data_folder/ppdb-2.0-tldr"
    WIKI_PATH = "/home-nfs/langyu/data_folder/enwiki/enwiki-unidecoded.txt"
    sample_size = 10000
    workload = "bird"
    random_seed = str(randint(10000, 99999))
}