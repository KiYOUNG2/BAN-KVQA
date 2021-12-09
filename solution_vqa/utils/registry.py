dictionary_dict = {
    'bert': {'dict': '', # BERT by Google
             'tokenizer': 'sp'},
    'bertrnn': {'dict': '', # BERT by Google
                'tokenizer': 'sp'},
    'fasttext-pkb': {'path': 'fasttext/ko.vec', # Word2vec by Kyubyong Park
                     'dict': 'dictionary_kkma.kvqa.pkl',
                     'tokenizer': 'kkma',
                     'embedding': 'ft_init.kvqa.npy',
                     'format': 'fasttext'}
}
