# hate_comments_datasets

ilbe 댓글 데이터를 이용한 혐오 데이터 셋

구축 방법 : Dataset_building.py 참고

구성 파일:
	model.pt : ilbe 댓글의 nouns만 추출하여 학습
		(Vector_size = 300,
		window = 5,
		Min_count = 3,
		sg = 0,
		epochs = 100)
	Dataset_building.py : 데이터 셋 구축 파일
	hate_label.txt : ilbe 댓글에 대한 label된 파일