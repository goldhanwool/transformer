import numpy as np
from global_args import PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX, toks_and_inds, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class BuildDataset:
    
    def __init__(self):
        pass

    def build_vocab(self, dataset, toks_and_inds, min_freq):
        def build_language_vocab(language_data):
            vocab = toks_and_inds.copy() #내부 함수를 사용시 vocab가 전역변수처럼 사용된다는 점 염두하기
            vocab_freqs = {}
            for word in language_data:
                vocab_freqs[word] = vocab_freqs.get(word, 0) + 1

            for word, freq in vocab_freqs.items():
                if freq >= min_freq and word not in vocab:
                    vocab[word] = len(vocab) # 인덱스 순서로 주기.
            
            return vocab

        en_data = [word for example in dataset for word in example['en']]
        de_data = [word for example in dataset for word in example['de']]

        en_vocab = build_language_vocab(en_data)
        de_vocab = build_language_vocab(de_data)

        print("+------------[build_vocab] > return------------+")
        print("\nen_vocab: ", en_vocab)
        print("\nde_vocab: ", de_vocab)
        
        return en_vocab, de_vocab
    
    def add_tokens(self, dataset, batch_size):
        for example in dataset:
            example['en'] = [SOS_TOKEN] + example['en'] + [EOS_TOKEN]
            example['de'] = [SOS_TOKEN] + example['de'] + [EOS_TOKEN]
        
        data_batches = np.array_split(dataset, np.arange(batch_size, len(dataset), batch_size))

        for batch in data_batches:
            max_en_seq_len = max(len(example['en']) for example in batch)
            max_de_seq_len = max(len(example['de']) for example in batch)
            for example in batch:
                example['en'] += [PAD_TOKEN] * (max_en_seq_len - len(example['en']))
                example['de'] += [PAD_TOKEN] * (max_de_seq_len - len(example['de']))
        
        return data_batches
    
    def build_dataset(self, dataset, vocabs):
        source, target = [], []
        for batch in dataset:
            source_tokens, target_tokens = [], []

            for sentence_dic in batch: #{'en': ['<sos>', 'a', 'teacher', 'explaining', 'a', 'lesson', 'to', 'students', 'in', 'a', 'classroom', '<eos>'], 'de': ['<sos>', '교실에서', '학생들에게', '수업을', '설명하는', '선생님', '<eos>', '<pad>']}
                """
                    **vocabs -> 
                    
                    (
                        {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3, 'a': 4, 'teacher': 5, 'explaining': 6, 'lesson': 7, 'to': 8, 'students': 9, 'in': 10, 'classroom': 11, 'two': 12, 'friends': 13, 'sharing': 14, 'meal': 15, 'at': 16, 'restaurant': 17, 'cyclist': 18, 'racing': 19, 'against': 20, 'time': 21, 'competition': 22, 'photographer': 23, 'taking': 24, 'pictures': 25, 'wedding': 26, 'child': 27, 'learning': 28, 'play': 29, 'the': 30, 'piano': 31, 'team': 32, 'of': 33, 'scientists': 34, 'conducting': 35, 'research': 36, 'laboratory': 37, 'mountain': 38, 'climber': 39, 'reaching': 40, 'summit': 41, 'group': 42, 'tourists': 43, 'guided': 44, 'tour': 45, 'museum': 46, 'street': 47, 'performer': 48, 'juggling': 49, 'front': 50, 'crowd': 51, 'barista': 52, 'making': 53, 'coffee': 54, 'cafe': 55}, 
                        {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3, '교실에서': 4, '학생들에게': 5, '수업을': 6, '설명하는': 7, '선생님': 8, '레스토랑에서': 9, '식사를': 10, '함께': 11, '나누는': 12, '두': 13, '친구': 14, '경기에서': 15, '시간과': 16, '경쟁하며': 17, '자전거를': 18, '타는': 19, '사람': 20, '결혼식에서': 21, '사진을': 22, '찍고': 23, '있는': 24, '사진작가': 25, '피아노': 26, '치는': 27, '법을': 28, '배우고': 29, '아이': 30, '실험실에서': 31, '연구를': 32, '수행하고': 33, '과학자': 34, '팀': 35, '정상에': 36, '도달한': 37, '산악': 38, '등반가': 39, '박물관에서': 40, '가이드': 41, '투어를': 42, '하고': 43, '관광객': 44, '그룹': 45, '관중들': 46, '앞에서': 47, '저글링을': 48, '하는': 49, '거리': 50, '공연자': 51, '카페에서': 52, '커피를': 53, '만들고': 54, '바리스타': 55}
                    )
                
                    
                    **sentence_dic ->
                    
                    [
                        {
                            'en': ['<sos>', 'a', 'teacher', 'explaining', 'a', 'lesson', 'to', 'students', 'in', 'a', 'classroom', '<eos>'], 
                            'de': ['<sos>', '교실에서', '학생들에게', '수업을', '설명하는', '선생님', '<eos>', '<pad>']
                        },
                        {
                            'en': ['<sos>', 'a', 'teacher', 'explaining', 'a', 'lesson', 'to', 'students', 'in', 'a', 'classroom', '<eos>'], 
                            'de': ['<sos>', '실험실에서', '연구를', '수행하고', '있는', '과학자', '팀', '<eos>']
                        }
                    ]
                    
                    1. vocab에서 sentence_dic에 있는 단어들이 있는지 찾은 뒤 있으면 키로 되어있는 단어의 값을 할당하고 없으면 unk_index를 할당
                    2. 그리고 sentence를 새로운 리스트에 인덱스가 담겨진 리스트로 반환
                    3. 전체 리스트를 array로 만들어서 배치별로 소스 데이터와 번역 데이터를 각각 신규 리스트에 담는다. 
                    
                """

                en_inds = [vocabs[0].get(word, UNK_INDEX) for word in sentence_dic['en']] 
                de_inds = [vocabs[1].get(word, UNK_INDEX) for word in sentence_dic['de']]
                
                """
                    train_source -> 인풋 데이터 ex) 영어 
                    train_target -> 아웃풋 데이터 ex) 한국어

                """
                source_tokens.append(en_inds)
                target_tokens.append(de_inds)
            
            source.append(np.asarray(source_tokens))
            target.append(np.asarray(target_tokens))
        
        return source, target


