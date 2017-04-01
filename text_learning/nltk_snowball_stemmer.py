from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

print stemmer.stem("responsiveness")
print stemmer.stem("unresponsive")