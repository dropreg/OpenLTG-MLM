import os 
import nltk
import nltk.data

def preprocess(raw_line):
    '''
        We can preprocess the raw text by pre-defined rules.
    '''
    result_line = raw_line.replace("“", "\"").replace("``", "\"").replace("''", "\"").replace("”", "\"")
    result_line = result_line.replace("n ’ t", " n't")
    result_line = result_line.replace('. "', '."') 
    result_line = result_line.replace('*', '') 
    return result_line

def clip(raw_line):
    '''
        We can preprocess the raw text by pre-defined rules.
    '''
    result_line = []
    for w in raw_line.split(" ")[:350]:
        if len(w) > 64:
            continue
        result_line.append(w)
    return " ".join(result_line)

def split_file(raw_file, source_file, target_file, min_length=10, max_length=60):
    '''
        We can split paragraphs into sentence lists from the raw file and write to the source and target file line by line.
    '''
    tokenizer = nltk.data.load('/root/nltk_data/tokenizers/punkt/english.pickle')

    with open(source_file, 'w') as source_w, open(target_file, "w") as target_w:
        for raw_lines in open(raw_file).readlines():
            
            sentence_list = preprocess(raw_lines).strip().split("[SEP]")
            source_lines = sentence_list[0]
            source_w.write(source_lines + "\n")

            target_lines = " ".join(sentence_list[1:])

            merge_flag = False
            merge_number = 0
            merge_list = []
            span_number = 0
            target_sentence = tokenizer.tokenize(target_lines)

            if len(target_sentence) > 1:       
                bound = len(target_sentence) // 2   
                
                for bound_idx in range(bound):
                    bound_start = 2 * bound_idx
                    bound_end = bound_start + 2
                    if bound_end > len(target_sentence):
                        break

                    if bound_idx == 7 and bound_end < len(target_sentence) - 1:
                        bound_end = len(target_sentence)
                    
                    if bound > 1:
                        target_w.write(clip(" ".join(target_sentence[bound_start:bound_end])) + "\n")
                    else:
                        target_w.write(clip(target_sentence[bound_start]) + "\n") 
            else:
                target_w.write(clip(target_sentence[0]) + "\n")
            target_w.write("ENDEND" + "\n")


def split_writingprompt():

    root_dir = "/opt/data/private/data/arr1015_data/writing_prompt/seg_eight/"
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