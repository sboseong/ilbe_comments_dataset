# Hate Comments Dataset
### ilbe 댓글 데이터를 이용한 혐오 데이터 셋

&nbsp; 
 * 원시 데이터 : 일베(일간베스트 저장소) 댓글 약 120만개 
 
&nbsp; 

  * 훈련 및 테스트 : 
 ```
	- 훈련 집합 (train set) : 구축된 데이터 셋 중 약 34만개의 문장 (None 22만개, Hate 12만개)
	 

	- 테스트 집합 (test set) : 구축된 데이터 셋 중 약 38000개의 문장 (None 25000개, Hate 13000개)
	
	
	- 정확도 : 약 75%
```

&nbsp; 
 * 데이터 셋 구축 방법 : ```Dataset_building.py``` 파일 및 [논문](https://github.com/sboseong/ilbe_comments_dataset/blob/main/%EC%9B%8C%EB%93%9C%20%EC%9E%84%EB%B2%A0%EB%94%A9%20%EA%B8%B0%EB%B2%95%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%ED%98%90%EC%98%A4%ED%91%9C%ED%98%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%20%EC%9E%90%EB%8F%99%20%EA%B5%AC%EC%B6%95.pdf) 참고  

&nbsp; 

 * 구성 파일:
&nbsp; 

	```
	- word2vec_model.zip : ilbe 댓글의 nouns만 추출하여 학습한 Word2vec 모델
		(Vector_size = 300, window = 5, Min_count = 3, sg = 0, epochs = 100)
		
		
	- Dataset_building.py : 데이터 셋 구축 파일
	
	
	- ilbe_comments_None+Hate.txt : ilbe 댓글에 대해 Hate와 None으로 label된 파일
	
	
	- data/train.dat, data/test.dat : SVM 분류기 실험에 사용된 train set과 test set
	```
