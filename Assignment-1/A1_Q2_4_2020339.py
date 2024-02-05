import numpy as np
import pandas as pd
import A1_Q2_3_2021475 as Modified_LM
import A1_Q2_1_2021565 as LM
from utils import emotion_scores
import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
import sklearn

NUM_EXAMPLES = 50

raw = open('Dataset/corpus.txt', 'r').read().replace('\n', ' ')
training_text = open('Dataset/corpus.txt', 'r').read().splitlines()
training_labels = open('Dataset/labels.txt', 'r').read().splitlines()
emotions = {'sadness':0,'joy':1,'love':2,'anger':3,'fear':4,'surprise':5}

test_files = [
    'generated_examples/gen_joy.txt',
    'generated_examples/gen_anger.txt',
    'generated_examples/gen_fear.txt',
    'generated_examples/gen_love.txt',
    'generated_examples/gen_sadness.txt',
    'generated_examples/gen_surprise.txt'
]

params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000],
    'random_state': [42],
}

def main():
    num_merges = 800
    lm = LM.BiGramLM(raw,smoothing='kneser-ney',num_merges=num_merges)

    emotion_token = {}
    for i in lm.tokenizer.tokens:
        emotion_token[i] = emotion_scores(i)

    #delete the generated_examples folder if it exists
    if os.path.exists(os.path.join(os.getcwd(), "generated_examples")):
        shutil.rmtree(os.path.join(os.getcwd(), "generated_examples"))
    os.mkdir(os.path.join(os.getcwd(), "generated_examples"))

    for emotion in emotions.keys():
        lm_e = Modified_LM.ModifiedBiGramLM(emotion,raw,emotion_token=emotion_token,num_merges=num_merges)
        if os.path.exists(os.path.join(os.getcwd(), "generated_examples", "gen_{emotion}.txt".format(emotion=emotion))):
            os.remove(os.path.join(os.getcwd(), "generated_examples", "gen_{emotion}.txt".format(emotion=emotion)))
        
        with open(os.path.join(os.getcwd(), "generated_examples", "gen_{emotion}.txt".format(emotion=emotion)), 'w') as fp:
            for _ in range(NUM_EXAMPLES):
                generated = lm_e.generate(10, start='i') + '\n'
                fp.write(generated)

    test_text = []
    test_labels = []

    for file_path in test_files:
        with open(file_path, 'r') as file:
            test_text += file.read().splitlines()
            label = file_path.split('_')[-1].split('.')[0]
            test_labels += [label] * NUM_EXAMPLES


    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_text)
    X_test = vectorizer.transform(test_text)

    svm = SVC() 
    grid_search = GridSearchCV(svm, params, n_jobs=-1)
    grid_search.fit(X_train, training_labels)
    best_params = grid_search.best_params_

    y_test_pred = grid_search.predict(X_test)

    #Print sklearn classification report
    print(classification_report(test_labels, y_test_pred))

    with open('params.txt', 'w') as param_file:
        param_file.write(str(best_params))

    with open('metric.txt', 'w') as metric_file:
        metric_file.write(f"classification_report(test_labels, y_test_pred)")

if __name__ == '__main__':
    main()