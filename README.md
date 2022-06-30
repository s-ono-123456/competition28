# 中古マンション価格予測トレーニングコンペティション

https://www.nishika.com/competitions/13/

## data.ipynb
zipデータを解凍、データ配置する

## reference.ipynb
参考となる地価データを取得

## 利用したデータ類
地価公示データ
- https://nlftp.mlit.go.jp/ksj/old/datalist/old_KsjTmplt-L01.html

→平均地価しか出していない。
集約関数でmax-minやstd、max、min等を追加しても良いかもしれない。

国勢調査　都道府県・市区町村別の主な結果
https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00200521&tstat=000001049104&cycle=0&tclass1=000001049105&stat_infid=000032143614&tclass2val=0

駅別乗降客数
https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-v2_7.html