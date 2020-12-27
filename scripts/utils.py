def ravel(list_): return [j for sub in list_ for j in sub]


def generate_wcloud(topics_list, max_words=100, stopwords_list=['learning', 'neural', 'computer', 'algorithm', 'network', 'Artificial', 'model']):
    '''
    args : 1d topics list (1 x n)
    output : word cloud image 
    '''
    text = ' '.join(topics_list)

    # lower max_font_size, change the maximum number of word and lighten the background:
    stopwords = set(stopwords_list)
    wordcloud = WordCloud(max_font_size=100, stopwords=stopwords,
                          max_words=max_words, background_color="white").generate(text)
    print(wordcloud)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
