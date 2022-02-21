import random
import json
import numpy as np
from efficiency.log import show_time

def filter():
    with open("../paper_metadata_all.jsonl") as f, open('new.jsonl', 'w') as w:

        papers_by_s2_api = [json.loads(line) for line in f if line.strip()]
        # random.shuffle(papers_by_s2_api)
        papers_by_s2_api = papers_by_s2_api[0:10000]
        for each in papers_by_s2_api:
            w.write(json.dumps(each) + '\n')


def plot_histogram(dict):
    '''
    dict = {'male': [1222, 1333, 198, 20000, ...]
            'female': [145, 23789, 533, 222, ...]}
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-deep')

    x = dict['male']
    y = dict['female']
    bins = np.linspace(0, 200, 4)
    


    plt.hist([x, y], bins, label=['male', 'female'])
    plt.legend(loc='upper right')
    plt.show()

def plot_horizontal_bar()
    import matplotlib.pyplot as plt

    name_list = ['100-1K','1K-10K','10K-100K','100K-1M','1M-10M']
    num_list = [2.7613e+04, 1.2112e+04, 1.9030e+03, 8.1000e+01, 0.0000e+00]
    num_list2 = [6.2960e+03, 1.9580e+03, 2.1200e+02, 5.0000e+00, 0.0000e+00]
    x = list(range(len(num_list)))
    total_width = 0.8
    n=2
    width = total_width / n
    bar1 = plt.bar(x ,num_list ,width = width, color = 'mediumaquamarine', label='male')
    for i in range(len(x)):
        x[i] = x[i] + width
    bar2 = plt.bar(x ,num_list2 ,width = width, color = 'khaki', label='female',tick_label = name_list)

    number = ["81.4%", "86.1%", "90.0%", "94.2%", "0%"]
    i = 0
    for rect in bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, number[i], ha='center', va='bottom')
        i += 1

    number = ["18.6%", "13.9%", "10.0%", "5.8%", "0%"]
    i = 0
    for rect in bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, number[i], ha='center', va='bottom')
        i += 1

    plt.xticks(np.array(x) - width/2, name_list)
    plt.legend(loc='best')
    plt.xlabel('Citations')
    plt.ylabel('Number of Scholars')
    # plt.show()
    plt.savefig("hello.jpg", bbox_inches='tight', dpi=150)


def plot_vertical_bar()
    import matplotlib.pyplot as plt

    name_list = ['100-1K','1K-10K','10K-100K','100K-1M','1M-10M']
    num_list = [2.7613e+04, 1.2112e+04, 1.9030e+03, 8.1000e+01, 0.0000e+00]
    num_list2 = [6.2960e+03, 1.9580e+03, 2.1200e+02, 5.0000e+00, 0.0000e+00]
    plt.bar(range(len(num_list)),num_list, color = 'mediumaquamarine',tick_label = name_list, label='male')
    plt.bar(range(len(num_list2)),num_list2, color = 'khaki',tick_label = name_list, bottom=num_list, label= 'female')
    plt.legend(loc='best')
    plt.xlabel('Citations')
    plt.ylabel('Number of Scholars')
    plt.savefig("bar.jpg", bbox_inches='tight', dpi=150)

# male vs. female citations for all 5 domains of scholars
# total_num_scholars: 78536 # {'M': 41709, '-': 28356, 'F': 8471}
# bin_log10 = [2, 3, 4, 5, 6, 7] # namely we only show citations bin1= 100-1K, bin2=1K-10K, bin3=10K-100K, ...
# male_citation_log10 = [2.7613e+04, 1.2112e+04, 1.9030e+03, 8.1000e+01, 0.0000e+00]
# female_citation_log10 = [6.2960e+03, 1.9580e+03, 2.1200e+02, 5.0000e+00, 0.0000e+00]

# # male vs. female citations for NLP scholars
# num_of_nlp_scholars: 7233 # {'M': 3511, '-': 2638, 'F': 1084}
# bin_log10 = [2, 3, 4, 5, 6, 7] # namely we only show citations bin1= 100-1K, bin2=1K-10K, bin3=10K-100K, ...
# male_citation_log10 = [2.282e+03, 1.076e+03, 1.500e+02, 3.000e+00, 0.000e+00]
# female_citation_log10 = [7.980e+02, 2.560e+02, 2.900e+01, 1.000e+00, 0.000e+00]
def generate100k():
    all_scholars = np.load("final_anser_cites100.npy", allow_pickle=True)
    show_time('[Info] Finished reading. Start processing Google Scholar data.')
    all_scholars = all_scholars.tolist()
    str2clean = lambda i: re.sub("[^a-zA-Z]+", "", i.lower())
    wrong = []
    with open("../paper_metadata_all.jsonl") as f, open('100ksample.jsonl', 'w') as w:

        papers_by_s2_api = [json.loads(line) for line in f if line.strip()]
        # random.shuffle(papers_by_s2_api)
        papers_by_s2_api = papers_by_s2_api[0:100000]
        for each in papers_by_s2_api:
            try:
                each["gs_citation"] = int(id2paper[each["gs_id"]][3]) / (2022 - int(id2paper[each["gs_id"]][-1]))
                w.write(json.dumps(each) + '\n')
            except:
                wrong.append(each["gs_id"])
                # print(id2paper[each["gs_id"]])
                # break
    
def generate_df():
    import pandas as pd
    df = pd.read_json('100ksample.jsonl', lines=True)
    df.drop_duplicates(subset ="title", keep = False, inplace = True)
    df = df[pd.notnull(df['abstract'])]
    df = df[pd.notnull(df['title'])]
    df.to_csv("100ksample.csv")
        # df['citationCount'] = (df['citationCount'] >= 6).astype(int)
    

