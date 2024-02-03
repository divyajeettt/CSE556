from typing import Union
from A1_Q2_1_2021565 import BiGramLM,log
from utils import emotion_scores
from A1_Task1_2021529 import Tokenizer

class ModifiedBiGramLM(BiGramLM):  
           
    def __init__(self, emotion, corpus :str,emotion_token,smoothing : Union[str,None] = None, num_merges=400):
        self.indices = {'sadness':0,'joy':1,'love':2,'anger':3,'fear':4,'surprise':5}
        self.emotion = emotion
        self.emotion_token = emotion_token
        super().__init__(corpus, smoothing, num_merges)   
        print("Calculating modified probabilites")
        self.modified_calculate_probs()    

    def modified_conditional_prob(self, word1 : str, word2 : str) ->   float:
        prob = super().getfreq(word1,word2)/super().getfreq(word1)
        score = self.emotion_token[word2][self.indices[self.emotion]]['score']
        prob += 0.1*score
        return log(prob)
    
    def modified_calculate_probs(self):
        for word1 in self.tokenizer.tokens:
            self.conditional_log_probs[word1] = []
            for word2 in self.tokenizer.tokens:
                self.conditional_log_probs[word1].append(
                        (word2,self.modified_conditional_prob(word1,word2))
                )
            self.conditional_log_probs[word1].sort(key=lambda x:x[1],reverse=True)
    

    def generate_emotion(self, num_words: int, start : Union[None,str] = None) -> str:
        return super().generate(num_words,start)
         