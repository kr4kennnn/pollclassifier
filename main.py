import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from scipy import stats

warnings.filterwarnings("ignore", category=ConvergenceWarning)      #ignore unnecessary warnings

data = pd.read_csv(r'data.csv')                                     #dataset path

new_column_names = {                                                #rename column names with short values to get rid of long questions 
    'Yaşınız' : 'age' ,
    'Cinsiyyetiniz' : 'gender' ,
    'Ailenizin aylık geliri ne kadar ? (TL)' : 'family_income' ,
    'Öğrenci misiniz ?' : 'student_status' ,
    'Çalışıyor musunuz ?' : 'job_status' ,
    'Haftada kaç gün toplu taşıma kullıyorsunuz ?' : 'public_transport_use' ,
    'Gün içerisinde trafikte kaç saat geçiriyorsunuz ?' : 'traffic_time' ,
    'Farklı araba/motosiklet modellerine ve özelliklerine aşina mısınız?' : 'vehicle_info' ,
    'Ehliyetiniz var mı' : 'license_status' ,
    'Şu an veya geçmişte hiç motorlu taşıt sahibi oldunuz mu ?' : 'vehicle_status' ,
}
data = data.rename(columns=new_column_names)

#define data types
data['student_status'] = data['student_status'].replace('Evet',True).replace('Hayır',False)     
data['job_status'] = data['job_status'].replace('Evet',True).replace('Hayır',False)
data['vehicle_info'] = data['vehicle_info'].replace('Evet',True).replace('Hayır',False)
data['license_status'] = data['license_status'].replace('Evet',True).replace('Hayır',False)
data['vehicle_status'] = data['vehicle_status'].replace('Evet',True).replace('Hayır',False)   
data['age'] = data['age'].astype('int')
data['gender'] = data['gender'].replace('Erkek',0 ).replace('Kadın',1 )
data['family_income'] = data['family_income'].replace('0 - 10000',1).replace('10001 - 20000',2).replace('20001-30000',3).replace('30001-40000',4).replace('40000+',5)
data['public_transport_use'] = data['public_transport_use'].replace('0-2',1).replace('3-5',2).replace('6-7',3)
data['traffic_time'] = data['traffic_time'].replace('0-2',1).replace('2-3',2).replace('3+',3)


FEATURES = [                                                        #define our future and target parameters
    'age',
    'gender',
    'family_income',
    'student_status',
    'job_status',
    'public_transport_use',
    'traffic_time',
    'vehicle_info',
    'license_status',
]  

TARGET = 'vehicle_status'

X = data[FEATURES]
y = data[TARGET]

data  = preprocessing.normalize(data)                               #data normalization

# 5 classifier scores with 10 kfold value 
Log_score = cross_val_score(LogisticRegression(), X,y,cv=10)
SVC_score = cross_val_score(LinearSVC(),X,y,cv=10)
Knn_score = cross_val_score(KNeighborsClassifier(),X,y,cv=10)
RF_score = cross_val_score(RandomForestClassifier(),X,y,cv=10)
GNB_score = cross_val_score(GaussianNB(),X,y,cv=10)
print("Logistic regression: \n", Log_score)
print('Mean Accuracy:', Log_score.mean())
print("\n\n")
print("Linear SVC: \n" , SVC_score)
print('Mean Accuracy:', SVC_score.mean())
print("\n\n")
print("K-NN: \n " , Knn_score)
print('Mean Accuracy:', Knn_score.mean())
print("\n\n")
print ("Random Forest: \n", RF_score)
print('Mean Accuracy:', RF_score.mean())
print("\n\n")
print ("Gaussian NB: \n" , GNB_score)
print('Mean Accuracy:', GNB_score.mean())
print("\n")


#Selecting 5 best features
k = 5
feature_selector = SelectKBest(score_func=f_classif, k=k)
X_selected = feature_selector.fit_transform(X, y)

#applying PCA with these features
pca = PCA()
X_pca = pca.fit_transform(X_selected)
explained_variance_ratio = pca.explained_variance_ratio_
print("RESULTS AFTER FEATURE SELECTION AND PCA \n")
print("Explained Variance Ratio", explained_variance_ratio, '\n')

classifiers = [
    LogisticRegression(),
    LinearSVC(),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    GaussianNB()
]

classifier_scores = []
for clf in classifiers:
    pipeline = make_pipeline(feature_selector, pca, clf)
    scores = cross_val_score(pipeline, X, y, cv=10)
    classifier_scores.append(scores)
    print(clf.__class__.__name__)
    print(scores)
    print('Mean Accuracy:', scores.mean())
    print('\n')

#T-test for classifiers after feature selection and PCA
t_statistic, p_value = stats.ttest_rel(classifier_scores[0], classifier_scores[1])
print("Paired t-test results between Logistic Regression and Linear SVC:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[0], classifier_scores[2])
print("Paired t-test results between Logistic Regression and K-NN:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[0], classifier_scores[3])
print("Paired t-test results between Logistic Regression and Random Forest:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[0], classifier_scores[4])
print("Paired t-test results between Logistic Regression and Gaussian NB:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[1], classifier_scores[2])
print("Paired t-test results between Linear SVC and K-NN:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[1], classifier_scores[3])
print("Paired t-test results between Linear SVC and Random Forest:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[1], classifier_scores[4])
print("Paired t-test results between Linear SVC and Gaussian NB:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[2], classifier_scores[3])
print("Paired t-test results between K-NN and Random Forest:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[2], classifier_scores[4])
print("Paired t-test results between K-NN and Gaussian NB:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')

t_statistic, p_value = stats.ttest_rel(classifier_scores[3], classifier_scores[4])
print("Paired t-test results between Random Forest and Gaussian NB:")
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print('\n')