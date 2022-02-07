import random
import json
import numpy as np
from efficiency.log import show_time

# def filter():
#     with open("../paper_metadata_all.jsonl") as f, open('new.jsonl', 'w') as w:

#         papers_by_s2_api = [json.loads(line) for line in f if line.strip()]
#         # random.shuffle(papers_by_s2_api)
#         papers_by_s2_api = papers_by_s2_api[0:10000]
#         for each in papers_by_s2_api:
#             w.write(json.dumps(each) + '\n')


# def plot(dict):
#     '''
#     dict = {'male': [1222, 1333, 198, 20000, ...]
#             'female': [145, 23789, 533, 222, ...]}
#     '''

#     import numpy as np
#     import matplotlib.pyplot as plt
#     plt.style.use('seaborn-deep')

#     x = dict['male']
#     y = dict['female']
#     bins = np.linspace(0, 4000, 10)

#     plt.hist([x, y], bins, label=['male', 'female'])
#     plt.legend(loc='upper right')
#     plt.show()

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
    








generate100kdf()
generate_df()
# plot({'male': [1222, 1333, 198, 2000, 1334, 2557, 914, 345, 554], 'female': [145, 2389, 533, 222, 3721, 3441, 344, 824, 256]})