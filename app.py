
#EDA Packages
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import re
import os
import json
import requests
import SessionState
import streamlit as st
from PIL import Image

im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Sundar-Bot",
    page_icon=im,
    layout="wide",
)
st.title("Auto ModeL Sundar version")
st.header("Still in development only works for Category trget.")
@st.cache
def main(df,target,measure):
    strout = ""
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    dtypedf = pd.DataFrame()
    dtypedf['Feature'] = df.dtypes.index
    dtypedf['dtype'] = [str(x) for x in df.dtypes]
    numdf = df.loc[:,df.dtypes!=np.object]
    catdf = df.loc[:,df.dtypes==np.object]
    cat = 0
    if df[target].dtypes==np.object:
        cat=1
    for i in numdf.columns:
        df[i].fillna(df[i].mean(skipna=True),inplace=True)
    for i in catdf.columns:
        df[i].fillna(df[i].mode()[0],inplace=True)
    for i in catdf.columns:
        le = LabelEncoder()
        le.fit(df[i])
        df[i]= pd.Series(le.transform(df[i]))
    results = pd.DataFrame({"Model":[],"Accuracy":[],"AUC-ROC":[],"Precission":[],"Recall":[],"F1":[]})
    try:
        from sklearn.tree import DecisionTreeClassifier
        X = df.drop(columns = target)
        y =df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        results.loc[len(results.index)] = ['DecisionTreeClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    except:
        pass
    try:
        from sklearn.ensemble import RandomForestClassifier
        X = df.drop(columns = target)
        y =df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        clf=RandomForestClassifier(n_estimators=100)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['RandomForestClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    except:
        pass
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['ExtraTreesClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    except:
        pass
    try:
        clf = LGBMClassifier()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['LGBMClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    except:
        pass
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        clf  = GradientBoostingClassifier()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['GradientBoostingClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    except:
        pass
    try:
        from sklearn.ensemble import AdaBoostClassifier
        clf= AdaBoostClassifier(n_estimators=50)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['AdaBoostClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    except:
        pass
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    results.loc[len(results.index)] = ['QuadraticDiscriminantAnalysis', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                             metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    results.loc[len(results.index)] = ['GaussianNB', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                             metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=len(y.value_counts()))
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    results.loc[len(results.index)] = ['KNeighborsClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                             metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    from sklearn.linear_model import Ridge, Lasso
    try:
        clf = Ridge()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        y_pred = np.round(y_pred)
        results.loc[len(results.index)] = ['Ridge', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    except:
        pass
    try:
        clf = Lasso()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        y_pred = np.round(y_pred)
        results.loc[len(results.index)] = ['Lasso', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    except :
        pass
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    results.loc[len(results.index)] = ['LinearDiscriminantAnalysis', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                             metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
    results.sort_values(by =measure,inplace = True,ascending=False)
    return results
target = st.text_input("Target name Case-sensitive", "")
measure = st.sidebar.selectbox(
    "Pick measure you'd like to sort",
    ("Accuracy", "AUC-ROC", "Precission", "Recall", "F1") 
) 
uploaded_file = st.file_uploader(label="Upload an csvfile file",
                                 type=["csv"])
session_state = SessionState.get(pred_button=False)
# Create logic for app flow
if uploaded_file is not None:
    session_state.df=pd.read_csv(uploaded_file)
    session_state.target = target
    session_state.measure = measure
    pred_button = st.button("get results")
else:
    st.warning("Please upload csv file.")
    st.stop()    
# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 
# And if they did...
if session_state.pred_button:
    results= main(session_state.df, session_state.target, session_state.measure)
    st.dataframe(results)