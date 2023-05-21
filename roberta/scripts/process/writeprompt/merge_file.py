import os

def merge_file(input_file, dump_file):

    max_len = 0
    with open(input_file) as f, open(dump_file, 'w') as o:
            merge_list = []
            for lines in f.readlines():
                if lines.strip() != "10619 10619":
                    merge_list.extend(lines.strip().split())
                    merge_list.append("50265")
                else:
                    o.write(" ".join(merge_list[:-1]).strip() + "\n")
                    if max_len < len(merge_list):
                        max_len = len(merge_list)
                    merge_list = []
    print("the max lengtht {}".format(max_len))

def merge_source_file(input_file, dump_file):
    with open(input_file) as f, open(dump_file, 'w') as o:
            for lines in f.readlines():
                o.write(lines.strip() + "\n")

def merge_writingprompt():

    # root_dir = "/opt/data/private/data/arr1015_data/writing_prompt_min10/processed/"
    # root_dir = "/opt/data/private/data/arr1015_data/writing_prompt/seg_two/processed/"
    # root_dir = "/opt/data/private/data/arr1015_data/writing_prompt/seg_four/processed/"
    root_dir = "/opt/data/private/data/arr1015_data/writing_prompt/seg_eight/processed/"

    # Train File:
    train_source_file = "train.split.source"
    train_target_file = "train.split.target"
    # Dev File:
    dev_source_file = "valid.split.source"
    dev_target_file = "valid.split.target"
    # Test File:
    test_source_file = "test.split.source"
    test_target_file = "test.split.target"
    
    s_files = [os.path.join(root_dir, train_source_file), os.path.join(root_dir, dev_source_file), os.path.join(root_dir, test_source_file)]
    t_files = [os.path.join(root_dir, train_target_file), os.path.join(root_dir, dev_target_file), os.path.join(root_dir, test_target_file)]
    
    for s_file, t_file in zip(s_files, t_files):
        print("merge the source {}".format(s_file))
        dump_source_file = s_file.replace("split.", "")
        merge_source_file(s_file, dump_source_file)
        print("merge the target {}".format(t_file))
        dump_target_file = t_file.replace("split.", "")
        merge_file(t_file, dump_target_file)


if __name__ == "__main__":
    merge_writingprompt()
