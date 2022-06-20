# Hate Comments Dataset
### ilbe 댓글 데이터를 이용한 혐오 데이터 셋

&nbsp; 
 * 원시 데이터 : 일베(일간베스트 저장소) 댓글  

&nbsp; 
 * 데이터 셋 구축 방법 : ```Dataset_building.py``` 파일 참고  

&nbsp; 
 * 훈련 및 테스트 : 
&nbsp; 

	```
	- 훈련 집합 : 구축된 데이터 셋 중 임의로 선택한 10만개의 문장 (None 5만개, Hate 5만개)
	 

	- 테스트 집합 : Korean Hate Speech이라는 Kaggle Competition의 데이터 5679개의 문장
			

	- 분류기 모델 : SVM (Support Vector Machine)
	
	
	- 정확도 : 약 68 %
	```
	
###### [참고] [Korean Hate Speech](https://www.kaggle.com/competitions/korean-hate-speech-detection/data)

&nbsp; 
 * 구성 파일:
&nbsp; 

	```
	- model.pt : ilbe 댓글의 nouns만 추출하여 학습
		(Vector_size = 300, window = 5, Min_count = 3, sg = 0, epochs = 100)
		
		
	- Dataset_building.py : 데이터 셋 구축 파일
	
	
	- hate_label.txt : ilbe 댓글에 대한 label된 파일
	```
