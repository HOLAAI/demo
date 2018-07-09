## scikit learn 을 사용한 베이즈 분류 example

from konlpy.tag import Twitter


## example 문서들
raw_texts = [
    "카지노 대박",
    "안녕하세요 반갑습니다. 저는 철수 입니다",
    "서울 사는 친구들을 만나고 싶습니다.",
    "바카라 승률 보장",
    "한화 이글스 팬 없나요?",
    "호주에서 카지노 갔었습니다.",
    "대학교 문제로 속상합니다.",
    "여자친구가 화를 내요 상담이 필요합니다.",
    "카지노 농구 동호회 모집합니다",
    "카지노 100% 보장"
]
## 1 - spam , 0 - no spam
y = [1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
twitter = Twitter()
documents = []
for text in raw_texts :
    ## 명사만 사용
    nouns = twitter.nouns(text)
    doc = " ".join(nouns)
    documents.append(doc)

print("형태소 분석으로 재조립된 문서들 ==> ")
print(documents)
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X = count_vect.fit_transform(documents)

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

#model = BernoulliNB().fit(X, y)        ## 베르누이
model =MultinomialNB().fit(X, y)      ## 다행 분포

new_count = count_vect.transform(["카지노 승률 대박"])
proba = model.predict_proba(new_count)
print("class label : ", model.classes_)
print("예측: ",  proba)
