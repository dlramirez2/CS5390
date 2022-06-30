import OpenAttack
import nltk
import datasets
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from tqdm import tqdm
from contextlib  import contextmanager

@contextmanager
def no_ssl_verify():
    import ssl
    from urllib import request

    try:
        request.urlopen.__kwdefaults__.update({'context': ssl.SSLContext()})
        yield
    finally:
        request.urlopen.__kwdefaults__.update({'context': None})

def make_model():
    class MyClassifier(OpenAttack.Classifier):
        def __init__(self):
            try:
                self.model = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon')
                self.model = SentimentIntensityAnalyzer()
        
        def get_pred(self,input_):
            return self.get_prob(input_).argmax(axis=1)
            
        def get_prob(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 1e-6)
                ret.append(np.array([1 - prob, prob]))
            return np.array(ret)
    return MyClassifier()


def main():
    with no_ssl_verify():
        def dataset_mapping(x):
            return{
                "x":x["sentence"],
                "y":1 if x["label"] > 0.5 else 0,
            }

        print("Char-Level Attacks")
        attacker = OpenAttack.attackers.VIPERAttacker()

        print("Build model")
        victim = make_model() 

        dataset = datasets.load_dataset("sst",split="train[:20]").map(function=dataset_mapping)
        print("Start attack")

        attack_eval = OpenAttack.AttackEval(attacker,victim,metrics=[
            OpenAttack.metric.Fluency(),
            OpenAttack.metric.GrammaticalErrors(),
            OpenAttack.metric.SemanticSimilarity(),
            OpenAttack.metric.EditDistance(),
            OpenAttack.metric.ModificationRate()

        ])
        attack_eval.eval(dataset,visualize=True,progress_bar=True)
      
    
    
if __name__ == "__main__":
    main()
