# **TiDA-GCN** 

<p align="left">
  <img src='https://img.shields.io/badge/python-3.6+-blue'>
  <img src='https://img.shields.io/badge/Tensorflow-1.12+-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.16-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-0.22.0-brightgreen'>
  <img src='https://img.shields.io/badge/scipy-1.5.3-brightgreen'>
</p> 

## **Overall description** 
- Here presents the code of TiDA-GCN. As the dataset is too big for GitHub, we upload two datasets (i.e., Hvideo and Hamazon) on Bitbucket: [https://bitbucket.org/jinyuz1996/tida-gcn-data/src/master/](https://bitbucket.org/jinyuz1996/tida-gcn-data/src/master/ "https://bitbucket.org/jinyuz1996/tida-gcn-data/src/master/"). You should download them before training the GCNs. The code is attached to our paper: **"Time Interval-enhanced Graph Neural Network for Shared-account Cross-domain Sequential Recommendation" (TNNLs 2022)**. If you want to use our codes and datasets in your research, please cite our paper. Note that, our paper is still in the state of 'Early Access', if you want to see the preview version (on IEEE Xplore), please visit:[https://ieeexplore.ieee.org/document/9881215/]("https://ieeexplore.ieee.org/document/9881215/").
- You can also view the code of our previous study (DA-GCN) here, which is attached to the paper **"DA-GCN: A Domain-aware Attentive Graph Convolution Network for Shared-account Cross-domain Sequential Recommendation (IJCAI 2021)"**. If you want to use our codes and datasets in your research, please cite our paper from ijcai.org as:[https://www.ijcai.org/proceedings/2021/0342.pdf](https://www.ijcai.org/proceedings/2021/0342.pdf "https://www.ijcai.org/proceedings/2021/0342.pdf").

## **Code description** 
### **Vesion of implements and tools**
1. python 3.6
2. tensorflow 1.12.0
3. scipy 1.5.3
4. numpy 1.16.0
5. pandas 0.22.0
6. matplotlib 3.3.4
7. Keras 1.0.7
8. tqdm 4.60.0
### **Source code of TiDA-GCN**
1. the definition of recommender see: TiDA-GCN/TiDA-Module.py
2. the definition of Training process see: TiDA-GCN/TiDA_Train.py
3. the definition of Evaluating process see: TiDA-GCN/TiDA_Evaluation.py
4. the preprocess of dataset see: TiDA-GCN/TiDA_Configuration.py
5. the hyper-parameters of TiDA-GCN see: TiDA-GCN/TiDA_Settings.py
6. to run the training method see: TiDA-GCN/TiDA_Main.py and the training log printer was defined in: TiDA-GCN/TiDA_printer.py
 * The directory named Checkpoint is used to save the trained models.
 * The familiar floder structure is also apply to DA-GCN.
