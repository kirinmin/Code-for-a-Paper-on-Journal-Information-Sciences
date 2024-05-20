import numpy as np
import pandas
import pandas as pd
from sklearn.linear_model import LogisticRegression  # LR
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.svm import SVC  # SVM
from sklearn.naive_bayes import GaussianNB  # NB
from sklearn.tree import DecisionTreeClassifier  # DT
from sklearn.ensemble import RandomForestClassifier  # RF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # LDA
from xgboost.sklearn import XGBClassifier  # XGB
from sklearn.ensemble import GradientBoostingClassifier  # GBDT
from tensorly.decomposition import tucker  # tucker
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count
from datetime import datetime
import eOTD
# from sktensor import dtensor
# import sktensor
import tensorly as tl
from tensorly.tenalg import multi_mode_dot
from tensorly.decomposition import parafac
from tensorly.tenalg import khatri_rao


def metrics_cm( classifier, X_flow_train_de, X_flow_test_de, y_flow_train, y_flow_test):
    start = datetime.now()
    classifier.fit(X_flow_train_de, y_flow_train)
    prediction = classifier.predict(X_flow_test_de)
    end = datetime.now()

    timestr = str(end - start)
    datetime_obj = datetime.strptime(timestr, "%H:%M:%S.%f")
    ret_stamp = datetime_obj.microsecond / 1000000 + datetime_obj.second + datetime_obj.minute * 60 + datetime_obj.hour * 60 * 60 
    cm = confusion_matrix(y_flow_test, prediction)

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    FAR = FP / (TN + FP)
    F1_Score = 2 * (Recall * Precision) / (Recall + Precision)

    return Accuracy, Precision, Recall, FAR, F1_Score, ret_stamp

def Experiment( X_mat, y_class, kf, diffcla):
    kfold = KFold(n_splits=kf)
    scaler = StandardScaler()

    KNN = KNeighborsClassifier()
    SVM = SVC()
    NB = GaussianNB()
    LR = LogisticRegression()
    Lda = LDA()
    GBDT = GradientBoostingClassifier(n_estimators=50)
    DT = DecisionTreeClassifier()
    RF = RandomForestClassifier(n_estimators=50)
    XGB = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric=['logloss', 'auc', 'error'],
                        nthread=cpu_count())

    if diffcla == 1:
        cla_l = [GBDT, KNN, SVM, NB, LR, Lda, DT, RF, XGB]
    else:
        cla_l = [XGB]


    count = 1
    data_all = []

    for train_index, test_index in kfold.split(X_mat, y_class): 
        X_flow_train, y_flow_train = X_mat[train_index], y_class[train_index]  
        X_flow_test, y_flow_test = X_mat[test_index], y_class[test_index]  


        for cla in range(len(cla_l)):
            ml_p = metrics_cm( cla_l[cla], X_flow_train, X_flow_test, y_flow_train, y_flow_test)
            data_all.append(ml_p)
        count += 1

    data_mean = np.mean(np.array_split(data_all, kf), axis=0)
    return data_mean


if __name__ == '__main__':
    scaler = StandardScaler()
    dataset_net = 'cic'            
    if dataset_net == 'cic':
        path = "./data/CICCombined.csv"
        dataandclass=pd.read_csv(path, low_memory=False) 
        dataandclass= dataandclass.sample(frac=0.1) 
        dataandclass = dataandclass.iloc[:, :].values  
        data = dataandclass[:, :-1]
        data = scaler.fit_transform(data)  
        data_tensor = tl.fold(data, mode=2, shape=[8, 8, data.shape[0]])
        total_tensor = data_tensor[:, :, :]
        init_tensor = total_tensor[0:6, 0:6, 0: int(data.shape[0]*0.98)]
        #print("cic原始张量的大小为", np.shape(init_tensor))

    else:  #kdd
        dataandclass = pd.read_csv("./data/KDDTrain1+.csv", low_memory=False) 
        dataandclass = dataandclass.sample(frac=0.2)  
        dataandclass = dataandclass.iloc[:, :].values
        data = scaler.fit_transform(data) 
        data_tensor = tl.fold(data, mode=2, shape=[5, 8, data.shape[0]])
        total_tensor = data_tensor[:, :, :]
        init_tensor = total_tensor[0:4, 0:6, 0:int(data.shape[0]*0.98)]
        # printnp.shape(init_tensor))

    print('dataset', dataset_net, datetime.now(), 'total cores is %d' % cpu_count())
    print(init_tensor.shape)
    print(total_tensor.shape)

    arg='our'

    if arg == 'our':
        incre_tensor_tucker_result,de_stamp,paperrelaterror=\
            eOTD.KDDdtatade(init_tensor,total_tensor,a=0.6,thresholdv=0.999)
        matrix=tl.unfold(incre_tensor_tucker_result,mode=2)
    elif arg == 'HOSVD':
        rank1 = eOTD.ranks(total_tensor, 0.1)
        print(rank1)
        start = datetime.now()
        # U, G = eOTD.HOSVD(total_tensor, rank1)
        # multi_mode_dot_data = multi_mode_dot(G, U)
        core, tucker_factors = tucker(total_tensor, rank1)
        multi_mode_dot_data = multi_mode_dot(core, tucker_factors)
        end = datetime.now()
        timestr = str(end - start)
        de_stamp = datetime_obj.microsecond / 1000000 + datetime_obj.second + datetime_obj.minute * 60 + datetime_obj.hour * 60 * 60 
        paperrelaterror = eOTD.compreslt(total_tensor, multi_mode_dot_data)
        matrix=tl.unfold(multi_mode_dot_data,mode=2)
    elif arg == 'HOOI':
        rank1 = eOTD.ranks(total_tensor, 0.999)
        core, tucker_factors = sktensor.tucker_hooi(total_tensor, rank1) #, init='random'
        matrix = tl.unfold(multi_mode_dot_data, mode=2)

    elif arg == 'CP':
        rank1 = eOTD.ranks(total_tensor, 0.4)
        print(rank1)
        start = datetime.now()


        A, lbd, epoch = eOTD.cp_als(total_tensor, R=rank1[0], max_iter=4)
        full_tensor = tl.fold(np.matmul(np.matmul(A[0], np.diag(lbd)),
                                        khatri_rao(A, skip_matrix=0).T),
                              mode=0,
                              shape=total_tensor.shape)

        end = datetime.now()
       timestr = str(end - start)
        datetime_obj = datetime.strptime(timestr, "%H:%M:%S.%f")
        de_stamp = datetime_obj.microsecond / 1000000 + datetime_obj.second + datetime_obj.minute * 60 + datetime_obj.hour * 60 * 60 
        paperrelaterror = eOTD.compreslt(total_tensor, full_tensor)
        matrix= tl.unfold(full_tensor, mode=2)




    X_flow =matrix
    y_flow = dataandclass[:, -1]
    y_flow = np.array(y_flow,dtype=int)

    kf = 5  # k-fold
    result = Experiment( X_flow, y_flow, kf, 0).T 
    print('\n')
    print('Accuracy', result[0], 'Precision', result[1], 'Recall', result[2], 'FAR', result[3], 'F1_Score', result[4], result[5])
    totaltime=result[5]+de_stamp
    print([de_stamp],totaltime,'error',[paperrelaterror])  #[paperrelaterror]'error',[paperrelaterror

