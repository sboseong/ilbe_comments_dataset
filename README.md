# Hate Comments Dataset
### ilbe 댓글 데이터를 이용한 혐오 데이터 셋

&nbsp; 
 * 원시 데이터 : 일베(일간베스트 저장소) 댓글  
 
 ```
	- 훈련 집합 (train set) : 구축된 데이터 셋 중 약 34만개의 문장 (None 22만개, Hate 12만개)
	 

	- 테스트 집합 (test set) : 구축된 데이터 셋 중 약 38000개의 문장 (None 25000개, Hate 13000만개)
	
	
	- 정확도 : 약 75%
```

&nbsp; 
 * 데이터 셋 구축 방법 : ```Dataset_building.py``` 파일 참고  

&nbsp; 
 * 훈련 및 테스트 : 
&nbsp; 
	
######

&nbsp; 
 * 구성 파일:
&nbsp; 

	```
	- model.pt : ilbe 댓글의 nouns만 추출하여 학습
		(Vector_size = 300, window = 5, Min_count = 3, sg = 0, epochs = 100)
		
		
	- Dataset_building.py : 데이터 셋 구축 파일
	
	
	- hate_label.txt : ilbe 댓글에 대한 label된 파일
	
	
	- data/train.dat, data/test.dat : SVM 분류기 실험에 사용된 train set과 test set
	```
