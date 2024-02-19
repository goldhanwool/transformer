#clear dataset
class DataPreprocess:

    chars2remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    
    def clear_dataset(self, *data): #함수가 임의의 개수의 인자를 튜플 형태로 받음 (train_data, val_data, test_data)
        cleaned_datasets = []
        for dataset in data: # [train_data], [val_data], [test_data]
            self.fillering_dataset(dataset) #[{'en': 'Two young, White males are outside near many bushes.', 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
        
        return data
        
    def fillering_dataset(self, dataset):
        for sentence in dataset: #{'en': 'Two young, White males are outside near many bushes.', 'de': 'Zwei junge we...
            sentence['en'] = ''.join([char for char in sentence['en'] if char not in self.chars2remove]).lower().split()
            sentence['de'] = ''.join([char for char in sentence['de'] if char not in self.chars2remove]).lower().split()
                  
        return dataset
        
        
