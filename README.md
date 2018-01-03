# fundus

# fix error update list 

1.Class activation map 이 없는 tensor 들을 ensemble.py 에 같이 넣으면 에러가 나는 오류 을 수정

2.이미지 불러오고 255을 나누고 preprocessing 에서 color augmentation 할때 다시 [0,1] 로 pixel값을 limit 한다 문제는 없는가?. -->없다(확인 완료)

3.

# update 

test images의 softmax 값이 어떤 확률 분포를 가지는지 확인해보기.

t-sne 분석 업그레이드 

dense net 

vgg net 

wide resnet 

transfer learning (inception v3 완료 , inception v5 ,vgg 16 , alexent ,resnet)


# transfer learning 

inception v3 
<cache 가 저장되어 있지 않는데도 저장되어 있다는 버그 , 해당 경로에 없어도 pkl 파일이 있으면 load 한다 . 왜 그런지 모르겠다.
그리고 pkl 을 로드하면 읽어올수 없다고 한다. 뭔가가 남겨져 있어서 그런것 같은데 그게 뭐인지 모르겠다.?
>



# learning rate 을 어떻게 확인할까?
--> learinig rate 을 감소시키는 gamma 을 적용시키기
