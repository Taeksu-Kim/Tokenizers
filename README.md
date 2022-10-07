### Tokenizers

한국어 Tokenizer를 직접 만드는 것에 대한 repo

Tokenizer가 잘 쪼개졌는지는 NER이나 MRC 데이터셋을 이용하여 Tokenizer의 Offsetmapping을 이용하여 Token index와 원문 index를 이용, 정답 Token들을 다시 Decode한 후 원문 과 비교하는 방식 등으로 확인해볼 수 있을 것으로 보임

잘 나누어지지 않는 패턴으로는 은,는,이,가,을,를,으로 등이 있음
ex) 3시30분이 되었다 -> '분'/'이' 가아니라 '분이'로 나는 경우 등

Vocab 숫자가 늘어날수록 이러한 문제점이 더 커짐

huggingface의 Tokenizer 라이브러리를 이용하면 비교적 간단하게 토크나이저를 만들 수 있음

작업 자체는 시간이 그렇게까지 오래걸리지는 않기 때문에 PLM에 비하여 큰 데이터를 하드웨어 영향을 상대적으로 덜받으면서 사용해볼 수 있음 

조사 분리와 관련하여서는 형태소 분석기를 이용하여 먼저 형태소 분석기로 나눠준 후 이후 토크나이저를 적용하는 방법을 사용해볼 수 있음
tokenizer_example.ipynb 참고

BPE -> 두 글자씩 묶어서 등장 빈도를 센 후 등장 빈도 수가 높은 것을 사용

WordPiece -> BPE와 비슷하지만 등장 빈도 수가 아니라 Likelihood를 사용. 'a','b','ab'인 경우, p(ab) / p(a)p(b)

BBPE -> 문자 단위가 아닌 바이트 단위에서 BPE 수행. ex) BPE : b e s t, BBPE :62 65 73 74
다국어, OOV 단어 처리에 효과적

기타 word level, char level, unigram Tokenizer등이 있음 

Huggingface의 Tokenizer Source들은 아래에서 확인해볼 수 있음
https://github.com/huggingface/tokenizers/tree/main/bindings/python/py_src/tokenizers/implementations

Sentencepiece를 이용하여 순서대로 Tokenizer를 만들고 싶을 경우 아래 링크 참고
https://github.com/upskyy/upskyy.github.io/blob/76a6d9a9bb2d4049c5b2552b3c73fee538023f13/_posts/2021-08-19-thirds.md

## custom_tokenizer.py
Huggingface의 Tokenizer를 약간 커스텀한 파일

## tokenizer_example.ipynb
커스텀한 Tokenizer + Mecab까지 사용한 Tokenizer 예제.
