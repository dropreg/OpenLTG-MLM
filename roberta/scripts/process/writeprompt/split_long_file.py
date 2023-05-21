import os 
import nltk
import nltk.data

def preprocess(raw_line):
    '''
        We can preprocess the raw text by pre-defined rules.
    '''
    # result_line = raw_line.replace("“", "\"").replace("``", "\"").replace("''", "\"").replace("”", "\"")
    # result_line = result_line.replace("n ’ t", " n't")
    # # result_line = result_line.replace('. "', '."')
    # result_line = result_line.replace('. "', ' . "')
    # result_line = result_line.replace('  ', ' ')
    # result_line = result_line.replace('*', '')
    result_line = raw_line.replace('*', '')
    new_line = []
    for w in result_line.split():
        if len(w) > 60:
            continue
        new_line.append(w)
    return " ".join(new_line)


def split_file(raw_file, source_file, target_file, min_length=10, max_length=60):
    '''
        We can split paragraphs into sentence lists from the raw file and write to the source and target file line by line.
    '''
    tokenizer = nltk.data.load('/root/nltk_data/tokenizers/punkt/english.pickle')
    
    max_source = 0
    max_target = 0
    with open(source_file, 'w') as source_w, open(target_file, "w") as target_w:
        for raw_lines in open(raw_file).readlines():

            sentence_list = preprocess(raw_lines).strip().split("[SEP]")
            
            source_sent = preprocess(sentence_list[0])
            if len(source_sent.split()) > max_source:
                max_source = len(source_sent.split())
            source_w.write(source_sent+ "\n")
            target_sent = preprocess(" ".join(sentence_list[1:]))
            if len(target_sent.split()) > max_target:
                max_target = len(target_sent.split())
            target_w.write(target_sent+ "\n")

    print(max_source, max_target)       
    
def split_writingprompt():

    root_dir = "/opt/data/private/data/acl2023_data/writingprompt_acl_123/"
    # Train File:
    train_file = "train.txt"
    train_source_file = "train.source"
    train_target_file = "train.target"
    # Dev File:
    dev_file = "val.txt"
    dev_source_file = "valid.source"
    dev_target_file = "valid.target"
    # Test File:
    test_file = "test.txt"
    test_source_file = "test.source"
    test_target_file = "test.target"
    
    r_files = [os.path.join(root_dir, train_file), os.path.join(root_dir, dev_file), os.path.join(root_dir, test_file)]
    s_files = [os.path.join(root_dir, train_source_file), os.path.join(root_dir, dev_source_file), os.path.join(root_dir, test_source_file)]
    t_files = [os.path.join(root_dir, train_target_file), os.path.join(root_dir, dev_target_file), os.path.join(root_dir, test_target_file)]
    
    for r_file, s_file, t_file in zip(r_files, s_files, t_files):
        print("process file {} to build the source {} and targert {}".format(r_file, s_file, t_file))
        split_file(r_file, s_file, t_file)

if __name__ == "__main__":
    split_writingprompt()