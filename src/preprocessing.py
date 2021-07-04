from tree_struct_analysis import analyze_parsed_sst
from torch.utils import data
from sklearn.model_selection import train_test_split


class EvalDataset(data.Dataset):
    def __init__(self, sequence_list):
        self.text_list = sequence_list
        self.size = len(sequence_list)

    def __getitem__(self, index):
        return self.text_list[index]

    def __len__(self):
        return self.size


class SentenceDataset(data.Dataset):
    def __init__(self, phrase_len, phrases, labels):
        self.size = 0
        self.sentences = phrases
        self.labels = labels
        self.target_len = phrase_len
        self.size = len(phrases)

    def filter_phrase(self):
        if self.target_len is None:
            return
        else:
            raw_sentences = self.sentences
            raw_labels = self.labels
            self.sentences = []
            self.labels = []
            for phrase, label in zip(raw_sentences, raw_labels):
                word_count = len(phrase.split())
                if word_count == self.target_len:
                    self.sentences.append(phrase)
                    self.labels.append(label)
            self.size = len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]
    def __len__(self):
        return self.size


def load_parsed_sst(parsed_sst_file, raw_sst_file, batch_size=32):
    parsed_sst_handler = open(parsed_sst_file, "r")
    raw_sst_handler = open(raw_sst_file)
    phrases, labels = [], []

    tag_lookup, label_mapping, _ = analyze_parsed_sst(parsed_sst_file)

    for line in parsed_sst_handler:
        words = line.strip().split()
        tag = words[0].replace("(", "")
        remapped_tag = tag_lookup[tag]
        labels += remapped_tag,
        
    for line in raw_sst_handler:
        line = line.strip().split("|")
        phrases.append(line[0])

    phrase_train, phrase_dev, label_train, label_dev = train_test_split(phrases, labels, test_size=0.15)
    
    train_set = SentenceDataset(None, phrase_train, label_train)
    dev_set = SentenceDataset(None, phrase_dev, label_dev)


    train_loader = data.DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
    dev_loader = data.DataLoader(dataset=dev_set, shuffle=True, batch_size=batch_size)

    return train_loader, dev_loader