# token_embedding = Embedding(INPUT_DIM, OUTPUT_DIM, data_type=np.float32)
from global_args import D_MODEL, DATA_TYPE
import numpy as np

"""
 vocab에 있는 단어들을 embedding 벡터로 만들기.
 vocab가 embedding벡터의 크기가 되며, 
 Output_dim은 d_model으로 벡터의 차원을 형성하게 된다.
     
"""

src_vocabs = vocabs[0]
trg_vocabs = vocabs[1]

input_dim = len(src_vocabs)#64
output_dim = D_MODEL


"""
    임베딩 순서
    가중치 형성 -> 원핫인코딩 -> 가중치 행렬과의 행렬 곱을 통해 임베딩 벡터 생성
    
"""

d_model = D_MODEL
input_dim = len(src_vocabs)#64
output_dim = d_model
data_type = DATA_TYPE


# 임베딩 행렬 만들기
def build(input_dim, output_dim):
    # Initialize weights and optimizer-related parameters
    
    """
    (vocabs.length, d_model)
    (64, 128) -> 총 64개의 단어 중 단어 한개 당 128개의 숫자로 표시됨 -> 차 후 행렬곱을 통해 차원을 변경하기 위함
    
    [
        [-0.13909371 -0.00494262 -0.04885064  0.12339354  0.24833478 -0.01550754
         -0.07263125 -0.13907956 -0.04903608 -0.05489263 -0.04621615  0.2981122
         -0.0738701   0.17770159  0.08714287  0.01635077  0.33333808 -0.00672233
          0.15200676  0.07497328 -0.25422883  0.03457352 -0.06668603 -0.08481509
          0.07594035  0.08715848 -0.34895572 -0.11775529  0.02324624 -0.03319979
         -0.0403923  -0.17475133  0.07940577  0.08783524 -0.1759314  -0.11330465
         -0.10199197  0.17205916  0.1874646  -0.05619537 -0.13816069 -0.13634555
          0.08525768 -0.12116446 -0.21173294  0.0529008  -0.2617494  -0.00610838
          0.09034323 -0.00594079  0.08814585 -0.05338065  0.08603407  0.06245302
          0.03024847  0.16254406  0.11350104  0.12725215  0.01842347  0.2424168
         -0.16149186  0.07628524 -0.18584116  0.2408927  -9.50129254  5.90684526
          0.09034323 -0.00594079  0.08814585 -0.05338065  0.08603407  0.06245302
          0.03024847  0.16254406  0.11350104  0.12725215  0.01842347  0.2424168
         -0.16149186  0.07628524 -0.18584116  0.2408927  -9.50129254  5.90684526 ...],
         [-9.50129256e-02  5.90684526e-02 -7....]
     ]
    
    
    """
    """
        ** w 가중치 행렬 만들기 **
        
        np.random.normal(0, pow(input_dim, -0.5), (input_dim, output_dim))
    
        -np.random.normal: 넘파이(Numpy) 라이브러리의 함수로, 정규 분포(normal distribution)에서 무작위 샘플을 생성합니다. 
        이 함수는 세 가지 주요 매개변수를 받습니다: 평균(mean), 표준편차(std), 그리고 출력될 샘플의 형태(size).

        -0: 정규 분포의 평균(mean) 값입니다. 여기서 0을 지정함으로써, 생성되는 랜덤 값들의 평균이 0이 되도록 합니다.

        -pow(self.input_dim, -0.5): 정규 분포의 표준편차(standard deviation)입니다. self.input_dim은 임베딩 계층의 입력 
        차원 크기(즉, 어휘 사전의 크기)를 나타냅니다. pow(self.input_dim, -0.5)는 self.input_dim의 -0.5승을 계산하는 것으로, 
        입력 차원의 크기에 반비례하는 표준편차를 설정합니다. 이는 초기 가중치의 분산을 입력 차원 크기에 따라 조정하여, 
        너무 크거나 작은 값이 초기 가중치로 설정되는 것을 방지합니다. 
        이러한 방식은 가중치 초기화 시 과도한 분산이 모델 학습에 부정적 영향을 미치는 것을 방지하기 위해 사용됩니다.

        -(self.input_dim, self.output_dim): 생성될 샘플의 형태(size)입니다. 이는 임베딩 행렬의 차원을 나타내며, 
        self.input_dim은 어휘 사전의 크기(입력 차원), self.output_dim은 임베딩 벡터의 크기(출력 차원)를 의미합니다.


    """
    w = np.random.normal(0, pow(input_dim, -0.5), (input_dim, output_dim)).astype(data_type)
 
    """
     ** 변수 설명 **

        -m (모멘텀): 이는 과거 그래디언트의 지수적 가중 평균입니다. m은 매개변수의 업데이트 방향을 결정하는 데 사용되며, 
        이는 최적화 과정에서 관성의 역할을 하여 지역 최소값(local minima)에 빠지는 것을 방지하고, 최적화 과정을 안정화시킵니다.

        -v (스케일된 스퀘어 그래디언트 변수): 이는 과거 그래디언트의 제곱의 지수적 가중 평균입니다. 
        v는 각 매개변수에 대해 적응적으로 학습률을 조정하는 데 사용되며, 이를 통해 학습률을 매개변수별로 다르게 적용할 수 있습니다.

        -m_hat과 v_hat: Adam 최적화에서는 m과 v의 편향 보정(bias correction)을 위해 사용됩니다. 
        초기 단계에서 m과 v는 0으로 초기화되어 있기 때문에, 학습 초기에는 이들의 추정치가 실제 값보다 작게 나타납니다. 
        m_hat과 v_hat는 이러한 초기 편향을 보정하기 위해 계산되며, 보다 정확한 학습률 조정과 최적화 과정을 가능하게 합니다.


        v, m, v_hat, m_hat ->
        
        [
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 0. 0. 0. 0. 0. 0. 0.],
         [0. 0. ...                                                              ],
         
         ...
        ]

    """    
    v, m = np.zeros_like(w).astype(data_type), np.zeros_like(w).astype(data_type)
    v_hat, m_hat = np.zeros_like(w).astype(data_type), np.zeros_like(w).astype(data_type)

    return w, v, m, v_hat, m_hat

    
#prepare_labels 메서드 배치에서 제공된 레이블(단어 인덱스)을 one-hot 인코딩 형태로 변환합니다.
def prepare_labels(src, batch_size, current_input_length):
    # Prepare one-hot encoded labels
    """
    ** src **
    
        array([[ 1., 11.,  3.,  3.,  3.,  6.,  3., 16., 11.,  3.,  2.],
               [ 1., 11.,  3.,  3.,  3.,  6., 11.,  3.,  2.,  0.,  0.]],
    
    """
    
    """
    src의 float를 int32로 변환
    ->
    
    batch_labels:  [[ 1 11  3  3  3  6  3 16 11  3  2][ 1 11  3  3  3  6 11  3  2  0  0]]
    
    """
    batch_labels = src.astype(np.int32) #float 타입을 정수로 변환 

    
    """
        ** 원핫인코딩 **
        
        np.zeros((batch_labels.size, input_dim))
        batch_labels.size 전체 단어의 수
        
        batch_labels 배열의 전체 요소 수에 해당하는 행의 수를 가진 새로운 배열을 생성합니다. 
        여기서 input_dim은 이 새 배열의 열 수를 지정합니다. 
        결과적으로, 이 코드는 batch_labels의 모든 요소에 대해 각각 input_dim 크기의 원-핫 인코딩 벡터를 생성할 수 있는 
        0으로 채워진 배열을 만듭니다.
    
    """
    #batch_labels.size -> batch_labels 배열의 전체 요소 수 
    """
        
        총 22개의 리스트들. 
            리스트 안에는 vocabs의 개수를 가진 0으로 초기화한 후 
            -> 22개의 단어들을 인덱스를 구하고 
            -> 리스트 내 해당 인덱스를 1로 표시한다.   

        [
            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
             0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
             0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.],
            [0. 0. 0. 0. 0....     ]
        ]
    
    """    
    prepared_batch_labels = np.zeros((batch_labels.size, input_dim))    

    
    """
    [Numpy 고급인덱싱]
    
    행렬 A가 아래와 같을 경우
    [[0, 1, 2],
     [3, 4, 5],
     [6, 7, 8]]
    
    A[[1], [[0]]] => [[3]]이 된다. 
    -> 이는 원본 배열의 2번째 행, 첫 번째 열에 해당하는 요소를 2차원 배열 형태로 반환

    ** np.arange **
    
    1. np.arange(batch_labels.size) : 주어진 시작 값에서 종료 값 전까지, 지정된 간격으로 숫자들의 배열을 생성
        
        -> [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21]
    
    
    ** batch_labels.reshape **
    
    2. batch_labels.reshape(1, -1) : 1은 새로운 배열의 행의 수를, -1은 열의 수를 자동으로 계산하여 배열이 전체 요소를 포함할 수 있도록 설정
                                     1을 첫 번째 인자로 사용함으로써, 결과 배열은 1행짜리 2차원 배열
        
        
        origin -> [[ 1 11  3  3  3  6  3 16 11  3  2][ 1 11  3  3  3  6 11  3  2  0  0]]
        reshape(1, -1) -> [[ 1 11  3  3  3  6  3 16 11  3  2  1 11  3  3  3  6 11  3  2  0  0]]
    
    
    ** [np.arange(batch_labels.size), batch_labels.reshape(1, -1)] **
    
    3. prepared_batch_labels[np.arange(batch_labels.size), batch_labels.reshape(1, -1)] = 1  
        
        -> Numpy 고급 인덱싱에 따라  
            prepared_batch_labels[[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21], [[ 1 11  3  3  3  6  3 16 11  3  2  1 11  3  3  3  6 11  3  2  0  0]]]
            리스트 자리에 1을 할당
            
    
         ** 결과 ** 
        
        [
         [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
          0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.], 
         [0. 0. 1. 0....]
        ...
        ]
    
    """
    prepared_batch_labels[np.arange(batch_labels.size), batch_labels.reshape(1, -1)] = 1 #(22, 64)

    return prepared_batch_labels.reshape(batch_size, current_input_length, input_dim).astype(data_type)

    
def forward(src, w):
    # Forward pass through the embedding layer
    if not all([np.equal(len(src[0]), len(arr)).all() for arr in src]):
        raise ValueError("Input sequences must be of the same length")

    """
        

        input_data:  (22, 64)
        w:  (64, 128)

        m x k * k x n = m x n
    
        => (src 내 모든 단어의 수, 단어의 표현수(d_model))
    
    """
    
    input_data = src
    
    current_input_length = len(input_data[0])
    print("current_input_length: ", current_input_length)
    
    batch_size = len(input_data)
    print("batch_size: ", batch_size)

    input_data = prepare_labels(input_data, batch_size, current_input_length)
    
    output_data = np.dot(input_data, w)
    
    """
        ** embedding 최종 벡터 **
        2차원 / 22개 리스트들 / 128개의 벡터들 / <class 'numpy.ndarray'>
    
    22개
    [
        128개
        [-0.01754851 -0.02039202  0.03705394 -0.05842575 -0.04335693 -0.09243767
         -0.02292279  0.1649055  -0.23684667 -0.09028699 -0.13731095 -0.08329499
         -0.1539625   0.03145679 -0.02831026 -0.01981316  0.13953198  0.23046137
          0.11754691  0.11977625 -0.2048956  -0.22407499  0.05049831  0.09267972
         -0.28039318  0.14498429 -0.11362207  0.1257344  -0.05583074  0.20342787
          0.01950205 -0.10378047  0.18143481 -0.08829035  0.15565318  0.01062319
         -0.10667782 -0.0788945  -0.3130156  -0.0778164  -0.07208032 -0.03936708
          0.10513901  0.07858507 -0.22845136  0.09986452 -0.19954075 -0.07029081
         -0.01430234  0.0306071  -0.04272318 -0.06421797 -0.01900979 -0.02070363
         -0.10371406  0.1379824  -0.04348351 -0.05357485 -0.0742117  -0.00678039
         -0.06185937 -0.12180065 -0.05522724 -0.07674956 -0.04551932 -0.11485524
         -0.05761339  0.2525947   0.20954578  0.14501038 -0.2929794   0.06387337
          0.09428639  0.0665492   0.07598448 -0.00121802 -0.02703046 -0.08394213
         -0.00092731  0.01218953  0.23703417  0.19902468  0.0663708   0.21338741
          0.3086231   0.23671836 -0.04345555 -0.06494432 -0.14651255 -0.01834501
          0.03750806  0.15845391  0.22131018  0.18024859  0.05381922  0.25223953
          0.02940506  0.13870372  0.02031351 -0.12297809 -0.21689643  0.11214298
          0.13774447 -0.11734725  0.0777282  -0.14302221  0.05022501  0.0856167
          0.15016966 -0.02298907 -0.10908934  0.08416137  0.17010488 -0.0212655
         -0.20138791 -0.21280202  0.01435964 -0.11012035 -0.00096831 -0.03000556
          0.19578189  0.00517127 -0.03915082  0.08026774  0.20363002  0.00213619
         -0.02959533  0.19515276],
         [0.19578189  0.00517127 ... ]
    ...
    
    ]
    """

    return output_data   


#w, v, m, v_hat, m_hat = build(input_dim, output_dim) #가중치 행렬 초기화

#token_embedding.forward(src)
# embedding_nparray = forward(src, w)

# print("embedding 단어 개수: ", len(embedding_nparray))
# print("embedding 단어별 벡터표현 개수: ", len((embedding_nparray[0])))
# print("embedding.shape: " , embedding_nparray.shape)
# print("\nembedding_nparray: " , embedding_nparray)

"""
 embedding 남은 과제

"""

#     def backward(self, error):
#         # Backward pass through the embedding layer
#         self.grad_w = np.matmul(np.transpose(self.input_data, axes=(0, 2, 1)), error).sum(axis=0)
#         # No error propagation to previous layers as the embedding layer is usually the first layer
#         return None

#     def update_weights(self, layer_num):
#         # Update weights using the optimizer
#         self.w, self.v, self.m, self.v_hat, self.m_hat = self.optimizer.update(
#             self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)
#         return layer_num + 1
