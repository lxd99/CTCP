# Continuous-Time Graph Learning for Cascade Popularity Prediction

This is the implementation of CTCP: [Continuous-Time Graph Learning for Cascade Popularity Prediction, IJCAI 2023].

## Requirements

- python == 3.7.0
- pytorch == 1.9.1
- dgl == 0.8.2
- scikit-learn == 1.0
- numpy == 1.21.5
- pandas == 1.1.5
- tqdm == 4.62.3

## Dataset

- Download the preprocessed dataset from [Baidu Yun](https://pan.baidu.com/s/1qBauuSOk29lQUH9Bvh6_hg) （extract code myu6）
- create a directory `./data` and put the downloaded dataset into the directory.

## Run
Create the directories to store the running results 
```sh
mkdir log results saved_models
```
Running command

```sh
#Twitter
python main.py --dataset twitter  --prefix std --gpu 0 --epoch 150 --embedding_module aggregate --use_dynamic --use_temporal --use_structural --use_static --dropout 0.6 --predictor merge --lambda 0.1
#APS
python main.py --dataset aps  --prefix std --gpu 0 --epoch 150 --embedding_module aggregate --use_dynamic --use_temporal --use_structural --use_static --dropout 0.6 --predictor merge --lambda 0.1
#Weibo
python main.py --dataset weibo  --prefix std --gpu 0 --epoch 150 --embedding_module aggregate --use_dynamic --use_temporal --use_structural --use_static --dropout 0.6 --predictor merge --lambda 0.1
```

After running, the log file, results, and trained model are saved under the directories of `log`, `saved_results,` and `saved_models`  respectively.

