from nltk import parse
from nltk.parse import stanford
import re
import sys
from os import path

from config import config

write_every = 100

def postprocess_tree(string):
        illegal_tags = ["DT", "NNS", "IN", "NN", "VBZ", "RB", "VB", "VBP", "VBD", "VBN", "JJ", "TO", "PRP", "PRP\$", "POS", "CC", "NNP", "RP", "VBG", "CD", "JJS", "JJR", "WDT", "EX", "NNP"]

        for tag in illegal_tags:
            pattern = "\(" + tag + " (\w+)\)"
            string = re.sub(pattern, r'\1', string)

        pattern = "\(" + "JJ" + " (\w+-.*)\)"
        string = re.sub(pattern, r'\1', string)
        pattern = "\(" + "JJ" + " (-\w+-\w*)\)"
        string = re.sub(pattern, r'\1', string)
        pattern = "\(" + "NNP" + " (.*)\)"
        string = re.sub(pattern, r'\1', string)
        return string


def gen_parsed_sentence(sentences, out_file="./parsed_sent.txt"):
    parser = stanford.StanfordParser()
    out_file = open(out_file, "w")


    for sentence_id, sentence in enumerate(sentences):
        parsed = next(parser.raw_parse(sentence))
        sent_string = " ".join(str(parsed).split())
        sent_string = sent_string[6:-1] # remove (ROOT ... )
        out_file.write(postprocess_tree(sent_string))
        out_file.write("\n")

        if sentence_id % write_every == 0:
            out_file.flush()

    out_file.close()

def clean_parsed_sentences(file_loc, vocab):
    handler = open(file_loc, "r")
    out = open("./cleaned_out.txt", "w")
    for line in handler:
        new_line = line.replace("(, ,)", ",").replace("('' '')", "''").replace("(JJ '')", "''")
        new_line = new_line.replace("(. .)", ".").replace("(POS \'s)", "").replace("(POS \')", "").replace("(-LRB- -LRB-)", "").replace("(-RRB- -RRB-)", "")
        words = new_line.split()
        for word in words:
            cleaned_word = word.replace("(", "").replace(")", "")
            if cleaned_word.islower() and cleaned_word not in vocab:
                # out of vocabulary word
                new_line = new_line.replace(cleaned_word, "UNK")
        out.write(postprocess_tree(new_line))

    handler.close()
    out.close()

def parse_vocab(file):
    handler = open(file, "r")
    vocab = set()
    for line in handler:
        segments = line.split("\t")
        vocab.add(segments[1])
    handler.close()
    return vocab

# naively add (S ...) to original sentences to match the format requirement for RNNG
def simple_parse(file, vocab, out_file):
    handler = open(file, "r")
    out = open(out_file, "w")
    for line in handler:
        line = line.strip()
        words = line.split()
        for word in words:
            if word.islower() and (word not in vocab):
                # out of vocabulary word
                # line = line.replace(word+" ", "UNK ")
                pattern = "\\b{}\\b".format(word)
                line = re.sub(pattern, "UNK", line)

        line = line.replace("\"", "")
        out.write("(S " + line + ")\n")
        # out.write(line + "\n" )
        # out.write("NT(S)\nREDUCE\n")
    out.close()


def parse_sst(raw_sst, parsed_out_dir):
    raw_in= open(raw_sst, "r")
    sentences = []
    # need to write out both parsed version and original version to generate fine-tuning workload
    parsed_out = path.join(parsed_out_dir, "parsed_sst.txt")
    raw_out = open(path.join(parsed_out_dir, "sst_raw_sentence.txt"), "w")
    for line in raw_in:
        segments = line.strip().split("|")
        sentences.append(segments[0])

    gen_parsed_sentence(sentences, parsed_out)
    for sent in sentences:
        raw_out.write(sent + "/n")
    raw_out.close()


if __name__ == "__main__":
    vocab_file = config["vocab_file"]
    vocab = parse_vocab(vocab_file)
    sentence_file_loc = sys.argv[1]
    sentence_filename = path.basename(sentence_file_loc)
    out_sentence_filename = "rnng_" + sentence_filename
    out_name = path.join(config["out_dir"], out_sentence_filename)
    # clean_parsed_sentences(filename, vocab)
    simple_parse(sentence_file_loc, vocab, out_name)

