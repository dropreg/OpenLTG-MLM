import os 
import nltk
import nltk.data

def preprocess(raw_line):
    '''
        We can preprocess the raw text by pre-defined rules.
    '''
    result_line = raw_line.replace(" . '", ".'")
    result_line = result_line.replace("mr .", "mr.")
    result_line = result_line.replace('*', '') 
    return " ".join(result_line.split())


def split_file(source_file, target_file, min_length=20, max_length=50):
    '''
        We can split paragraphs into sentence lists from the raw file and write to the source and target file line by line.
    '''
    tokenizer = nltk.data.load('/root/nltk_data/tokenizers/punkt/english.pickle')
    source_out_file = source_file.replace(".source", ".sent.source")
    target_out_file = target_file.replace(".target", ".sent.target")

    with open(source_out_file, 'w') as source_w, open(target_out_file, "w") as target_w:
        for source_lines, target_lines in zip(open(source_file).readlines(), open(target_file).readlines()):
            
            source_lines = preprocess(source_lines).strip()
            source_w.write(source_lines + "\n")

            target_lines = preprocess(target_lines).strip()
            merge_flag = False
            merge_number = 0
            merge_list = []
            span_number = 0
            target_sentence = tokenizer.tokenize(target_lines)
            
            if len(target_sentence) > 1:            
                bound = len(target_sentence) // 2
                if bound > 1:
                    target_w.write(" ".join(target_sentence[:bound]) + "\n")
                else:
                    target_w.write(target_sentence[0] + "\n")
                if bound < len(target_sentence) - 1:
                    target_w.write(" ".join(target_sentence[bound:]) + "\n")
                else:
                    target_w.write(target_sentence[len(target_sentence) - 1] + "\n")
            else:
                target_w.write(target_sentence[0] + "\n")
            target_w.write("ENDEND" + "\n")

def split_writingprompt():
    
    # root_dir = "/opt/data/private/data/arr1015_data/roc_story"
    root_dir = "/opt/data/private/data/arr1015_data/roc_story/seg_two/"
    # Train File:
    train_source_file = "train.source"
    train_target_file = "train.target"
    # Dev File:
    dev_source_file = "val.source"
    dev_target_file = "val.target"
    # Test File:
    test_source_file = "test.source"
    test_target_file = "test.target"
    
    s_files = [os.path.join(root_dir, train_source_file), os.path.join(root_dir, dev_source_file), os.path.join(root_dir, test_source_file)]
    t_files = [os.path.join(root_dir, train_target_file), os.path.join(root_dir, dev_target_file), os.path.join(root_dir, test_target_file)]
    
    for s_file, t_file in zip(s_files, t_files):
        print("process file to build the source {} and targert {}".format(s_file, t_file))
        split_file(s_file, t_file)

if __name__ == "__main__":
    split_writingprompt()