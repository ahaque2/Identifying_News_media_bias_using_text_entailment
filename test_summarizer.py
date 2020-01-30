import pandas as pd
from wrapper import summarizer

#data = pd.read_csv('data/entailment_negative_training_sample.csv')
#sent1 = data['sent1']
#sent2 = data['sent2']
sent = "This is a sentence that I want to test my new text summarizer on."
s = summarizer()
summarized_sent = s.summarize(sent)
print(summarized_sent)
print(len(summarized_sent))