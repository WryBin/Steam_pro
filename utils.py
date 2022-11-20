import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


def feat_kernelMedian(df, df_feature, fe, value, pr, name=""):
    def get_median(a, pr=pr):
        a = np.array(a)
        x = a[~np.isnan(a)]
        n = len(x)
        weight = np.repeat(1.0, n)
        idx = np.argsort(x)
        x = x[idx]
        if n<pr.shape[0]:
            pr = pr[n,:n]
        else:
            scale = (n-1)/2.
            xxx = np.arange(-(n+1)/2.+1, (n+1)/2., step=1)/scale
            yyy = 3./4.*(1-xxx**2)
            yyy = yyy/np.sum(yyy)
            pr = (yyy*n+1)/(n+1)
        ans = np.sum(pr*x*weight) / float(np.sum(pr * weight))
        return ans

    df_count = pd.DataFrame(df_feature.groupby(fe)[value].apply(get_median)).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
    
def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot(scale=3)进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n

# Feiyang: 1. 获得核函数 PrEp
PrOriginalEp = np.zeros((2000,2000))
PrOriginalEp[1,0] = 1
PrOriginalEp[2,range(2)] = [0.5,0.5]
for i in range(3,2000):
    scale = (i-1)/2.
    x = np.arange(-(i+1)/2.+1, (i+1)/2., step=1)/scale
    y = 3./4.*(1-x**2)
    y = y/np.sum(y)
    PrOriginalEp[i, range(i)] = y
PrEp = PrOriginalEp.copy()
for i in range(3, 2000):
    PrEp[i,:i] = (PrEp[i,:i]*i+1)/(i+1)

def adv_val(Train_data, pred_data, features):

    # 对抗验证
    df_adv =  pd.concat([Train_data, pred_data])

    adv_data = lgb.Dataset(
        data=df_adv[features], label=df_adv.loc[:, 'Is_Test']
    )

    # 定义模型参数
    params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 300,
    'learning_rate': 0.1,
    'metric': 'auc',
    'verbose': -1,
    'min_data_in_leaf': 6,
    'max_depth':30,
    'seed':42, 
    'sub_feature': 0.7, 
    }

    adv_cv_results = lgb.cv(
        params, 
        adv_data,
        num_boost_round=30, 
        nfold=5, 
        early_stopping_rounds=10, 
        # verbose_eval=True, 
        seed=42)

    # print('交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
    #     adv_cv_results['auc-mean'][-1], adv_cv_results['auc-stdv'][-1]))

    # print('模型最优的迭代次数为{}.'.format(len(adv_cv_results['auc-mean'])))

    # 使用训练好的模型，对所有的样本进行预测，得到各个样本属于测试集的概率
    params['n_estimators'] = len(adv_cv_results['auc-mean'])

    model_adv = lgb.LGBMClassifier(**params)
    model_adv.fit(df_adv[features], df_adv.loc[:, 'Is_Test'])

    preds_adv = model_adv.predict_proba(df_adv[features])[:, 1]
    
    return preds_adv, adv_cv_results