# AI-Illustration-Analysis
AIイラストと人手イラストの違いを見つけることが目的\
AIイラストはwife diffusion, stable diffusion, Nobel AIで作成されたものを対象にしています。
# 実験項目 
[AIイラスト顔検出(exp1)](#AIイラスト顔検出)\
[識別器の学習(exp2)](#識別器の学習)
## AIイラスト顔検出
人手イラストで学習したイラスト顔検出器(以前開発したYOLOv3ベースのもの(訓練データ1000枚で学習))をAIイラストに適用した際に正しく検出できるのかを試した。\
結果\
しっかり検出された。
![image](https://user-images.githubusercontent.com/55880071/195827632-b47b94c6-8f7d-424f-9e5e-916e5bece9e2.png)

## 識別器の学習
人手イラスト(879枚)とAIイラスト(598枚)を使って2値分類ResNetの学習をさせ、正しく認識できるかを試した。\
結果\
学習が不安定で誤判定まみれだった。\
学習のさせ方が悪いのかも。\
データ拡張はランダムクロップを使用。\
顔だけで識別可能かは今後の課題だが、普通に無理そう。
![loss](https://user-images.githubusercontent.com/55880071/199756768-7a1edf46-607c-4aa6-9c59-499ff23fa753.png)
