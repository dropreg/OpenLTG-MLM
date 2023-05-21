import os
import json
import nltk
import nltk.data

def preprocess_plan(str_in):
    str_res = str_in.replace("< s > ' re", "< s >")
    str_res = str_res.replace("< s > ' m", "< s >")
    str_res = str_res.replace("< s > ' d", "< s >")
    str_res = str_res.replace("< s > ' ll", "< s >")
    str_res = str_res.replace("< s > ' s", "< s >")
    str_res = str_res.replace("< s > ' ve", "< s >")
    str_res = str_res.replace(" < s > ' 01", "")
    return " ".join(str_res.split())

def preprocess_target(str_in):
    str_res = str_in.replace(".", " .")
    return " ".join(str_res.split())

def split_file(r_file, s_file, t_file, p_file):
    
    tokenizer = nltk.data.load('/root/nltk_data/tokenizers/punkt/english.pickle')
    with open(s_file, 'w') as s_w, open(t_file, 'w') as t_w, open(p_file, 'w') as p_w:
        for line in open(r_file).readlines():
            json_line = json.loads(line)
            prompt = nltk.word_tokenize(json_line['prompt'].strip())
            tgt = nltk.word_tokenize(json_line['tgt'].strip())
            kp_set_str = nltk.word_tokenize(json_line['kp_set_str'].strip())
            s_w.write(" ".join(prompt) + "\n")
            t_w.write(preprocess_target(" ".join(tgt)) + "\n")
            p_w.write(preprocess_plan(" ".join(kp_set_str)) + "\n")

def split_writingprompt():

    root_dir = "/opt/data/private/code/pair-emnlp2020/data/arggen/"
    target_dir = "/opt/data/private/data/arggen/"
    # Train File:
    raw_train_file = "refinement_train.jsonl"
    train_source_file = "train.source"
    train_target_file = "train.target"
    train_plan_file = "train.plan"
    # Dev File:
    raw_dev_file = "refinement_dev.jsonl"
    dev_source_file = "val.source"
    dev_target_file = "val.target"
    dev_plan_file = "val.plan"
    # Test File:
    raw_test_file = "refinement_test.jsonl"
    test_source_file = "test.source"
    test_target_file = "test.target"
    test_plan_file = "test.plan"
    
    r_files = [os.path.join(root_dir, raw_train_file), os.path.join(root_dir, raw_dev_file), os.path.join(root_dir, raw_test_file)]
    s_files = [os.path.join(target_dir, train_source_file), os.path.join(target_dir, dev_source_file), os.path.join(target_dir, test_source_file)]
    t_files = [os.path.join(target_dir, train_target_file), os.path.join(target_dir, dev_target_file), os.path.join(target_dir, test_target_file)]
    p_files = [os.path.join(target_dir, train_plan_file), os.path.join(target_dir, dev_plan_file), os.path.join(target_dir, test_plan_file)]

    for r_file, s_file, t_file, p_file in zip(r_files, s_files, t_files, p_files):
        print("process raw file {} to build the source {}, targert {} and plan {}".format(r_file, s_file, t_file, p_file))
        split_file(r_file, s_file, t_file, p_file)

if __name__ == "__main__":
    split_writingprompt()