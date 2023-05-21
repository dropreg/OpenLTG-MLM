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
    return result_line


def split_file(source_file, target_file, min_length=20, max_length=50):
    '''
        We can split paragraphs into sentence lists from the raw file and write to the source and target file line by line.
    '''
    tokenizer = nltk.data.load('/root/nltk_data/tokenizers/punkt/english.pickle')

    stat_info = {"min_sent": 100, "max_sent": 0, "span_max": 0, "span_over": 0,"trunct_number": 0, "all_number": 0}

    def load_stat(span_number, sent_length):

        if stat_info["min_sent"] > sent_length:
            stat_info["min_sent"] = sent_length
        if stat_info["max_sent"] < sent_length:
            stat_info["max_sent"] = sent_length
        if stat_info["span_max"] < span_number:
            stat_info["span_max"] = span_number
        if span_number > 10:
            stat_info["span_over"] +=  1
        stat_info["all_number"] += 1
    
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
            for sent_idx, sentence in enumerate(target_sentence):
                
                if span_number > 10:
                    break

                if merge_flag:
                    merge_list.append(sentence)
                    sentence = " ".join(merge_list)

                sent_len = len(sentence.split())
                if min_length <= sent_len < max_length:
                    target_w.write(sentence + "\n")
                    merge_flag = False
                    merge_list = []
                    merge_number = 0
                    span_number += 1

                    load_stat(span_number, sent_len)
                elif sent_len < min_length:
                    if sent_idx + 1 < len(target_sentence) and (len(target_sentence[sent_idx + 1].split()) + sent_len < max_length):
                        if not merge_flag:
                            merge_list.append(sentence)
                        merge_flag = True
                        merge_number += 1
                        continue
                    else:
                        if sent_len <= 3:
                            continue
                        
                        target_w.write(sentence + "\n")
                        merge_flag = False
                        merge_list = []
                        merge_number = 0
                        span_number += 1

                        load_stat(span_number, sent_len)
                else:
                    assert merge_flag == False and len(merge_list) == 0

                    for word_item in sentence.split():
                        merge_list.append(word_item)

                        if len(merge_list) > max_length:
                            sent_item = " ".join(merge_list)
                            stat_info["trunct_number"] += 1
                            
                            span_number += 1
                            target_w.write(sent_item.strip() + "\n")
                            load_stat(span_number, len(merge_list))
                            merge_list = []

                    if len(merge_list) > 3:
                        sent_item = " ".join(merge_list)
                        stat_info["trunct_number"] += 1
                        span_number += 1
                        target_w.write(sent_item.strip() + "\n")
                        load_stat(span_number, len(merge_list))
                    merge_list = []
            target_w.write("ENDEND" + "\n")
    print(stat_info)


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