import pandas as pd
import json

class GenderMap:
    def __init__(self,path):
        json_file = json.loads(open(path).read())
        self.map = dict()
        for word in json_file:
            try:
                if word['gender'] == 'm':
                    m_word = word['word']
                    f_word = word['gender_map']['f'][0]['word']
                    self.map[m_word] = f_word
                    self.map[f_word] = m_word
                elif word['gender'] == 'f':
                    f_word = word['word']
                    m_word = word['gender_map']['m'][0]['word']
                    self.map[m_word] = f_word
                    self.map[f_word] = m_word
            except KeyError:
                pass

gm = GenderMap('gendered_words/gendered_words.json')
print(gm.map['her'])