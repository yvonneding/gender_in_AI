import nltk
from nltk.corpus import words
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import math
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

class WhyCitation:
    def _load_paper2citation_data(self):
        import json
        from efficiency.log import show_time

        with open(P.file_s2_api) as f:
            papers_by_s2_api = [json.loads(line) for line in f if line.strip()]
        show_time(f'Got {len(papers_by_s2_api)} papers_by_s2_api from {P.file_s2_api}')

        keys = ['id', 'title', 'year', 'citation', 'citation_influential', 'venue', 'authors', 'abstract']
        papers = []
        for paper in papers_by_s2_api:
            clean_paper = {'gs_id': paper['gs_id']}
            for k in keys:
                clean_paper[k] = eval(U.meta_file_format['s2_api'][k])

            papers.append(clean_paper)


        self.raw_paper_df = pd.DataFrame(papers)
        self.raw_paper_df = self.raw_paper_df[pd.notnull(self.raw_paper_df['abstract'])]
        self.raw_paper_df = self.raw_paper_df[pd.notnull(self.raw_paper_df['title'])]
        self.raw_paper_df = self.raw_paper_df[pd.notnull(self.raw_paper_df['citation'])]
        content_set = []
        for index, row in self.raw_paper_df.iterrows():
            content_set.append(row[2])
            content_set.append(row[8])
        content_set = filter(None, content_set)
        cv = CountVectorizer()
        word_count_vec = cv.fit_transform(content_set).toarray()
        word_name = cv.get_feature_names()
        word_count_vec = np.array(word_count_vec)
        freq = np.sum(word_count_vec, axis=0)
        self.common_word_set = []
        for index, item in enumerate(word_name):
            if freq[index] > 1000:
                self.common_word_set.append(word_name[index])
        
        # for binary classification
        # print(self.raw_paper_df['citation'].describe())
        # self.raw_paper_df['citation'] = (self.raw_paper_df['citation'] >= 6).astype(int)

        '''
        (Pdb) self.raw_paper_df
                              gs_id  ...                                           abstract
        0       7715758236046727466  ...  This paper introduces a new P wave arrival tim...
        1      10251033406930934735  ...  Our long-term interest is in machines that con...
        2      13633406419057296907  ...  Various cell types have been investigated as c...
        3      16750122778147439803  ...  Annual low dose computed tomography (CT) lung ...
        4       2364745725011706039  ...  OBJECTIVES\nTo examine the economic burden of ...
        ...                     ...  ...                                                ...
        81696  14468527423587399910  ...  In security-sensitive applications, the succes...
        81697   1241252694255680131  ...  An attempt has been endeavored in the Analytic...
        81698  14224623745193385985  ...  5G networks are primarily designed to support ...
        81699  16494909129552116152  ...  Blockchain (BC) has become one of the most imp...
        81700   7665949997968365444  ...  Recently, caption generation with an encoder-d...
        
        [81701 rows x 9 columns]
        (Pdb) self.raw_paper_df.iloc[0]
        gs_id                                                 7715758236046727466
        id                      {'s2_hash': '1516188ff50404c845f79ce4341f73121...
        title                   A Study on the P Wave Arrival Time Determinati...
        year                                                               2011.0
        citation                                                                2
        citation_influential                                                    1
        venue
        authors                 [{'authorId': '49564745', 'name': 'K. S. Lee'}...
        abstract                This paper introduces a new P wave arrival tim...
        Name: 0, dtype: object
        '''

    def _load_paper_features(self):
        # TODO: load the hand-crafted features we come up with
        df = self.raw_paper_df
        feature_input = []
        for index, row in self.raw_paper_df.iterrows():
            if row[8] and row[2]:
                abstract_features = Abstract2Features(row[8])
                numbers_in_Abs = abstract_features.num_features()

                title_features = Title2Features(row[2])
                has_acronym, has_mark, has_colon = title_features.feature_style()
                low_freq_words = title_features.feature_novelty(self.common_word_set)
                # model_feature_input = {
                #     'has_acronym': int(has_acronym == True), # bool
                #     'has_mark': int(has_mark == True),  # bool
                #     'has_colon': int(has_colon == True), # bool
                #     'num_low_freq_words': low_freq_words, # float
                #     'numbers_in_Abs': numbers_in_Abs # list of numbers
                # }
                model_feature_input = {
                    'has_acronym': has_acronym, # bool
                    'has_mark': has_mark,  # bool
                    'has_colon': has_colon, # bool
                    'num_low_freq_words': low_freq_words, # float
                    'numbers_in_Abs': numbers_in_Abs # float
                }
                feature_input.append(model_feature_input)
        self.paper_feature_df = pd.DataFrame(feature_input)
        self.paper_feature_df = pd.concat([self.paper_feature_df, self.raw_paper_df['citation']], axis=1)

class Abstract2Features:
    def __init__(self, abstract):
        self.abstract = abstract
        self.tokens = word_tokenize(abstract)

    def num_features(self):
        numbers = [i for i in self.tokens if i.replace('.', '', 1).isdigit()]
        return len(numbers)

class Title2Features:
    def __init__(self, title):
        self.title = title
        self.tokens = word_tokenize(title)

    def feature_style(self):
        tokens = self.tokens
        has_acronym = False
        for i in tokens:
            upper = 0
            for c in i:
                if c.isupper():
                    upper += 1
            if upper >= 2:
                has_acronym = True
                break
        if ":" in tokens:
            if tokens.index(':') == 1:
                has_acronym = True


        # has_acronym = any(len([c.isupper() for c in i]) >= 2 for i in tokens) or (tokens.index(':') == 1)
        pattern_mark = '[\?\!]'
        has_mark = re.search(pattern_mark, self.title)!=None
        has_colon = True if any(i == ':' for i in tokens) else False
        # has_acronym: you check whether (1) there are words which has more than 2 capital letters, e.g., GloVe, BERT, OpinionFinder
        # OR (2) it is in the format of "Babytalk: Understanding and generating simple image descriptions"
        
        return has_acronym, has_mark, has_colon

    # def feature_topic(self):
    #     # pre_calculated_features is an instance of FeaturesByNLPModels()
    #     subareas = FeaturesByNLPModels(self.title)
    #     return subareas.title2subarea

    def feature_novelty(self, common_word_set):
        # common_word_set is what you pre-calculate on all paper title+abstract (tokenized version) of words larger than 100 occurrences
        # import nltk
        # new_text = nltk.word_tokenize(text)
        # We should avoid double tokenization: new_new_text = nltk.word_tokenize(new_text)

        low_freq_words = {i for i in self.tokens if i not in common_word_set}
        num_low_freq_words = len(low_freq_words)
        return num_low_freq_words # a number that shows how many low-frequency words are in the title.


class WhyCitation_AnswerByModeling(WhyCitation):
    def _load_exp_results(self):
        pass


class WhyCitation_AnsweredByCorrelation(WhyCitation):
    def filter_method(self):
        df = self.paper_feature_df
        # TODO: get each feature's importance by a certain criterion
        plt.figure(figsize=(12,10))
        cor = df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()
        #Correlation with output variable
        cor_target = abs(cor["citation"])
        #Selecting highly correlated features
        relevant_features = cor_target[cor_target>0.5]
        print(relevant_features)
    
    def backward_elimination(self):
        df = self.paper_feature_df
        df.dropna(inplace=True)
        X = df.drop("citation",1)   #Feature Matrix
        y = df["citation"]
        cols = list(X.columns)
        pmax = 1
        while (len(cols)>0):
            p= []
            X_1 = X[cols]
            X_1 = sm.add_constant(X_1)
            model = sm.OLS(y,X_1).fit()
            p = pd.Series(model.pvalues.values,index = cols)      
            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if(pmax>0.05):
                cols.remove(feature_with_p_max)
            else:
                break
        selected_features_BE = cols
        print(selected_features_BE)
    
    def recursive_feature_elimination(self):
        df = self.paper_feature_df
        df.dropna(inplace=True)
        X = df.drop("citation",1)   #Feature Matrix
        y = df["citation"]
        nof_list=np.arange(1,5)       
        high_score=0
        #Variable to store the optimum features
        nof=0           
        score_list =[]
        for n in range(len(nof_list)):
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
            model = LinearRegression()
            rfe = RFE(model, n_features_to_select=nof_list[n], step=1)
            X_train_rfe = rfe.fit_transform(X_train,y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]
        print("Optimum number of features: %d" %nof)
        print("Score with %d features: %f" % (nof, high_score))
    
    def pmi(self, feature_index):
        # p(label, featureX)/p(featureX) p(label)
        label_num = 0
        label_feature_num = 0
        df = self.paper_feature_df
        for index, row in df.iterrows():
            if bool(row[feature_index]):
                label_num += 1
                if row[5]:
                    label_feature_num += 1
        p_label = label_num / len(df)
        p_label_feature = label_feature_num / len(df)
        pmi = math.log10(p_label_feature / (0.5 * p_label))
        print(pmi)
    
    def chi_squared(self, row_index, col_index):
        from scipy.stats import chi2_contingency
        from scipy.stats import chi2
        df = self.paper_feature_df
        row_true_col_false = 0
        row_false_col_false = 0
        row_true_col_true = 0
        row_false_col_true = 0
        for index, row in df.iterrows():
            if row[row_index] and not row[col_index]:
                row_true_col_false += 1
            elif not row[row_index] and not row[col_index]:
                row_false_col_false += 1
            elif row[row_index] and row[col_index]:
                row_true_col_true += 1
            elif not row[row_index] and row[col_index]:
                row_false_col_true += 1
        contingency_table = [[row_false_col_false, row_false_col_true], [row_true_col_false, row_true_col_true]]
        print(contingency_table)
        stat, p, dof, expected = chi2_contingency(contingency_table)
        print('dof=%d' % dof)
        print(expected)
        # interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
        if abs(stat) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
        # interpret p-value
        alpha = 1.0 - prob
        print('significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
    
    def Pearsons(self, index1, index2):
        from scipy.stats import pearsonr
        df = self.paper_feature_df
        data1 = []
        data2 = []
        for index, row in df.iterrows():
            if not np.isnan(row[index1]) and not np.isnan(row[index2]):
                data1.append(row[index1])
                data2.append(row[index2])
        corr, _ = pearsonr(data1, data2)
        print('Pearsons correlation: %.3f' % corr)

    def selectKBest(self, k):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        df = self.paper_feature_df
        df = df.dropna()
        X = df.drop("citation",1)   #Feature Matrix
        y = df["citation"]
        select = SelectKBest(score_func=chi2, k=k)
        z = select.fit_transform(X,y)
        print(z)


def main():
    why_citation = WhyCitation_AnsweredByCorrelation()
    why_citation._load_paper2citation_data()
    why_citation._load_paper_features()
    why_citation.selectKBest(2)



if __name__ == '__main__':
    from file_paths import Paths
    from prepare_scholar_data import Utils

    P = Paths()
    U = Utils()

    # P.file_s2_api = P.file_s2_api.format('all')

    main()
