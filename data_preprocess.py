"""
Preprocess the Chinese Pretrain Datasets:
    1. wikizh: json: "text"                         1,043,224
    2. news2016zh: "title" + "content" + "desc"     2,500,000
    3. baike2018qa: "title" + "desc" + "answer"     1,500,000
    4. webtext2019zh: "title" + "desc" + "content"  4,100,000
return:
    the text file which contains all entries in the dataset.

data:
    each line is a JSON object

flow:
    1. Change the format of all files in the dataset into ".json"
    2. deal with each json file and extract the tag 'text' for each entry with /t
    3. return the output data
    4. split the training data and valid data
"""
import os
import json
# import jieba
import random
import jionlp
import argparse
import math

def main(dataset):
    wiki_path = r'dataset/wiki_zh_2019'
    baike_path = r'dataset/baike2018qa'
    webtext_path = r'dataset/webtext2019zh'
    news_path = r'dataset/new2016zh'

    if dataset == 'wiki':
        file_path = r'dataset/wiki_zh_2019'
    elif dataset == 'baike':
        file_path = r'dataset/baike2018qa'
    elif dataset == 'news':
        file_path = r'dataset/new2016zh'
    elif dataset == 'webtext':
        file_path = r'dataset/webtext2019zh'

    # output_path = r'dataset/processedData/'
    output_path = r'/local2/wuhao/processedData/'

    texts = []

    for dir_path, dir_name, files in os.walk(file_path):
        for file in files:
            print('preprocessing ', file)
            with open(dir_path + "/" + file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    current = len(texts)
                    if current % 5000 == 0:
                        print('current progress: {x}'.format(x=current))
                    j = json.loads(line)
                    if file.startswith('wiki'):
                        text = j['text']
                    elif file.startswith('news'):
                        text = j['title'] + j['desc'] + j['content']
                    elif file.startswith('baike'):
                        text = j['title'] + j['desc'] + j['answer']
                    elif file.startswith('web'):
                        text = j['title'] + j['desc'] + j['content']
                    else:
                        continue

                    # Clean the text
                    text = jionlp.clean_text(text)

                    # Since the tokenization would be implemented in TextDataset,
                    # It is unnecessary to tokenize the text here.
                    # tokens = jieba.cut(text, cut_all=True)
                    # text = " ".join(tokens)
                    texts.append(text)


    # To alleviate the stress of GPU, the size of output file is set to 100,000 samples
    # The training set is as the same as validation set

    num_samples = len(texts)
    num_each_file = 10000
    num_output_files = math.ceil(num_samples / num_each_file)
    random.shuffle(texts)

    print('total samples: {x}'.format(x=num_samples))
    print('total output files: {x}'.format(x=num_output_files))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    val_set_output = output_path + dataset + '_val'
    val_num = 1000
    print('generating valid set')
    with open(val_set_output, 'w', encoding='utf-8') as w:
        w.writelines(texts[0: val_num])

    for i in range(num_output_files):
        train_set_output = output_path + dataset + '_' + str(i) + '_train'
        print('generating the {x}-th train set'.format(x=i))
        with open(train_set_output, 'w', encoding='utf-8') as w:
            w.writelines(texts[i * num_each_file: (i + 1) * num_each_file])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='choose the dataset')
    args = parser.parse_args()

    main(args.dataset)

