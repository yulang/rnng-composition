from nltk.parse import stanford
import re

write_every = 100

def gen_parsed_sentence(sentences, out_file="./parsed_sent.txt"):
    parser = stanford.StanfordParser()
    out_file = open(out_file, "w")

    def postprocess_tree(string):
        illegal_tags = ["DT", "NNS", "IN", "NN", "VBZ", "RB", "VB", "VBP"]

        for tag in illegal_tags:
            pattern = "\(" + tag + " (\w+)\)"
            string = re.sub(pattern, r'\1', string)
        return string

    for sentence_id, sentence in enumerate(sentences):
        parsed = next(parser.raw_parse(sentence))
        sent_string = " ".join(str(parsed).split())
        sent_string = sent_string[6:-1] # remove (ROOT ... )
        out_file.write(postprocess_tree(sent_string))
        out_file.write("\n")

        if sentence_id % write_every == 0:
            out_file.flush()

    out_file.close()


