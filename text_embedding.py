import pandas as pd
import numpy as np

# edit the csv file
df1 = pd.read_csv('./unigram_freq.csv')
#df['babbage_search'] = df.babbage_search.apply(eval).apply(np.array)
ids = np.arange(1,len(df1)+1)
df1['Id'] = ids
df1.to_csv('unigram_freq_add_id.csv')

# prepare the embedding
df = pd.read_csv('./unigram_freq_add_id.csv', index_col=-1)
df = df[['word', 'count']]
df['combined'] = "a photo of a " + df.word.str.strip()

df = df.dropna()

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))

from openai.embeddings_utils import get_embedding

df['babbage_similarity'] = df.combined.apply(lambda x: get_embedding(x, engine='text-similarity-babbage-001'))
df['babbage_search'] = df.combined.apply(lambda x: get_embedding(x, engine='text-search-babbage-doc-001'))
df.to_csv('output/embedded_words.csv')


# calculate the similarity
import pandas as pd
import numpy as np

df = pd.read_csv('./embedded_words.csv')
df['babbage_search'] = df.babbage_search.apply(eval).apply(np.array)

from openai.embeddings_utils import get_embedding, cosine_similarity
# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    embedding = get_embedding(product_description, engine='text-search-babbage-query-001')
    df['similarities'] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n).combined
    if pprint:
        for r in res:
            print(r[:200])
            print()
    return res

res = search_reviews(df, 'delicious beans', n=3)