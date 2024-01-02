# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:43:15 2023

@author: pan
"""
#conda activate streamlit_test
#cd /d H:/git_xgbshare/xgbshare
#streamlit run predict.py
import akshare as ak
from datetime import date,timedelta
import pandas as pd 
# import tushare 
#print(tushare.__version__)

# pro = ts.pro_api('e79d0344d6ac178e4d5973c42b612c9ed776bc47117c49aa9d3d7b24')

import streamlit as st 
# import os
# st.write(os.getcwd())   #取得当前工作目录
#os.chdir(r'H:\git_xgbshare\xgbshare')
#全局配置
st.set_page_config(
    page_title="up-up-up",    #页面标题
    page_icon=":rainbow:",        #icon:emoji":rainbow:"
    layout="wide",                #页面布局
    initial_sidebar_state="auto"  #侧边栏
)


# today = datetime.today()
# today_str = today.strftime("%Y%m%d")

cte = {"日期":"trade_date","开盘":"open","收盘":"close","最高":"high","最低":"low",
          "成交量":"vol",'最新价':"close","今开":"open",'代码':'ts_code','名称':'name',
          "成交额":"amount","振幅":"pct_rag","涨跌幅":"pct_chg","涨跌额":"change","换手率":"turnover"}



def translatecolname(df,fromto):
    edited_col=df.columns.to_list()
    if fromto=='cte':
        repl=cte
    else:
        pass
    tran_col=[repl[i] if i in repl else i for i in edited_col]
    return tran_col


def getdata(symbol,startdate,enddate):
    #symbol='588200'
    if str(symbol)[0]=="5":
        historydata = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=startdate, end_date=enddate, adjust="")
    else:
        historydata = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=startdate, end_date=enddate, adjust="")
        # stock_zh_a_hist_df.info()
        
    historydata.columns=translatecolname(historydata,fromto='cte')
    # type(historydata.trade_date[0])
    #把日期设为index    
    historydata["trade_date"] = pd.to_datetime(historydata["trade_date"])
    historydata.set_index("trade_date", inplace=True, drop=True) # 把index设为索引

    return historydata

# 依据特征重要性，选择low high open来进行预测close
# 数据选择t-n, ...., t-2 t-1 与 t 来预测未来 t+m
# 转换原始数据为新的特征列来进行预测,time_window可以用来调试用前n次的数据来预测,p_day预测后m次的数据
def series_to_supervised(data,time_window,p_day):
    #data=stock_zh_a_hist_df
    #data=historydata.copy()
    #time_window=5
    #预设n(data_columns)*time_window个列名
    data_columns = ['open','high','low','close']
    data = data[data_columns]  # Note this is important to the important feature choice
    
    cols, names = list(), list()
    for i in range(time_window, -1, -1):
        # get the data
        #i 3 2 1 0
        #i=1
        cols.append(data.shift(i)) #数据偏移量
        
        # get the column name
        if ((i)<=0):
            suffix = '(t+%d)'%abs(i)
        else:
            suffix = '(t-%d)'%(i)
        names += [(colname + suffix) for colname in data_columns]
        
    pcols, pnames = list(), list()
    for j in range(p_day):
        #print(j+1)
        #j+1 1,2
        nshift=(j+1)*-1
        #print(nshift)
        pcols.append(data.shift(nshift)) #数据偏移量
        # get the column
        psuffix = '(t+%d)'%abs(j+1)
    
        pnames += [(colname + psuffix) for colname in data_columns]
           
    # concat the cols into one dataframe
    agg = pd.concat(cols,axis=1)
    aggp = pd.concat(pcols,axis=1)
    agg=pd.concat([agg,aggp],axis=1)
    agg.columns = names+pnames
    agg.index = data.index.copy()
    
    lst = ['open','high','low','close']
    
    def check_string(string, lst): 
        if any(substring in string for substring in lst):
            a= True
        else:
            a= False
        return a

    pricelist=[check_string(string,lst=lst) for string in agg.columns]
        
    pricemax=agg.loc[:,pricelist].iloc[:,:(time_window+1)*4].apply(lambda x :x.max(),axis=1)
    pricemin=agg.loc[:,pricelist].iloc[:,:(time_window+1)*4].apply(lambda x :x.min(),axis=1)
    
    agg['max']=pricemax.copy()
    agg['min']=pricemin.copy()
    
    scaled_data=pd.DataFrame()
    for col in agg.columns:
        # col='open(t+0)'
        if any(substring in col for substring in lst):
            scaled_data[col]= (agg[col]-agg['min'])/(agg['max']-agg['min'])
        else:
            scaled_data[col]=agg.loc[:,col].copy()   
    return agg,scaled_data

#模型
import xgboost as xgb
# print(xgb.__version__)
params = {
    'booster':'gbtree',
    'objective':'reg:squarederror',  # binary:logistic此处为回归预测，这里如果改成multi:softmax 则可以进行多分类
    'gamma':0.1,
    'max_depth':4,
    'lambda':3,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'verbosity':1,
    'eta':0.1,
    'seed':1000,
    'nthread':4,
}

def predicty(data_set_process,scaled_data,yname,timew,p_day):
    # yname='close'
    # p_day=1
    ycol=yname+'(t+%d)'%abs(p_day)
    
    #pd
    train_XGB= scaled_data[pd.notna(scaled_data[ycol])]
    test_XGB = scaled_data[pd.isna(scaled_data[ycol])]

    #close(t+1)
    train_XGB_X, train_XGB_Y = train_XGB.iloc[:,:(timew+1)*4],train_XGB.loc[:,ycol]
    test_XGB_X, test_XGB_Y = test_XGB.iloc[:,:(timew+1)*4],test_XGB.loc[:,ycol]

    #生成数据集格式
    xgb_train = xgb.DMatrix(train_XGB_X,label = train_XGB_Y)
    # xgb_test = xgb.DMatrix(test_XGB_X,label = test_XGB_Y)
    xgb_test = xgb.DMatrix(test_XGB_X)
    
    num_rounds =150
    #watchlist = [(xgb_test,'eval'),(xgb_train,'train')]
    watchlist = [(xgb_train,'train')]
    
    #xgboost模型训练
    model_xgb = xgb.train(params,xgb_train,num_rounds)

    # %matplotlib qt5
    # xgb.plot_importance(model_xgb)

    #对测试集进行预测
    y_pred_xgb = model_xgb.predict(xgb_test)
    
    #转换成数据框
    testy=pd.DataFrame(y_pred_xgb,index=test_XGB_Y.index,columns=['y_pred_xgb_t']).astype(float)
    
    #对训练集进行预测
    y_pred_xgb_t = model_xgb.predict(xgb_train)
    
    #将训练集的预测Y转成数据框
    y_pred_xgb_d=pd.DataFrame(y_pred_xgb_t,index=train_XGB_Y.index,columns=['y_pred_xgb_t']).astype(float)
    
    #将训练集真实Y和预测Y、测试集预测Y合并
    dd=pd.concat([train_XGB_Y,y_pred_xgb_d],axis=1)
    bb=pd.concat([data_set_process[[yname+'(t+0)',ycol,'max','min']],pd.concat([y_pred_xgb_d,testy],axis=0)],axis=1)
    
    #转换成股价
    bb['predict_'+yname]=bb['y_pred_xgb_t']*(bb['max']-bb['min'])+bb['min']
    bb.loc[pd.isna(bb[ycol]),ycol]=bb['predict_'+yname]
    # R2=r2_score(bb[ycol],bb['predict_'+yname])
    return ycol,bb.iloc[-1:,:]


def whole(symbol,startdate,enddate,timew,p_day):
    all_data_set=getdata(symbol=symbol,startdate=startdate,enddate=enddate)
    #滞后n期
    # timew=5
    data_set_process,scaled_data = series_to_supervised(all_data_set,time_window=timew,p_day=p_day) #取前3天的数据，预测后2天的数据
    predictdata=pd.DataFrame()
    for i in range(1,p_day+1):
        # i=1
        ycol,bb_close=predicty(data_set_process,scaled_data,yname='close',timew=timew,p_day=i)
        ycol,bb_high=predicty(data_set_process,scaled_data,yname='high',timew=timew,p_day=i)
        ycol,bb_low=predicty(data_set_process,scaled_data,yname='low',timew=timew,p_day=i)
        ycol,bb_open=predicty(data_set_process,scaled_data,yname='open',timew=timew,p_day=i)
        
        predict=pd.DataFrame({'open':bb_open.loc[:,'open(t+%d)'%i][-1],
                              'close':bb_close.loc[:,'close(t+%d)'%i][-1],
                                'low':bb_low.loc[:,'low(t+%d)'%i][-1],
                                'high':bb_high.loc[:,'high(t+%d)'%i][-1],
                                'trade_date':data_set_process.index[-1]+timedelta(days=i),
                                },index=[0]).set_index('trade_date')
        predictdata=pd.concat([predictdata,predict],axis=0)

    df=pd.concat([all_data_set[['open','close','low','high']],predictdata],axis=0)[-30:]
    
    return df


with open("./symparams.txt",encoding='utf-8') as file:
    symparamsfile =file.read()
    dictFinal =eval(symparamsfile)
    symparams =pd.DataFrame.from_dict(dictFinal, orient='columns')

# @st.cache_data
# def get_realtimedata(code):
#     #ts.get_realtime_quotes('002370')
#     realtimedata = tushare.get_realtime_quotes(code)[['name','time','code']]
    
#     return realtimedata

# codedf=get_realtimedata(code=symparams['ts_code'])
def getname():
    ashare=ak.stock_zh_a_spot_em()[["代码","名称"]]#,'最新价',"今开","最高","最低","换手率"   
    fund_etf=ak.fund_etf_spot_em()[["代码","名称"]]#,'最新价',"今开","最高","最低","换手率"
    realtimedata=pd.concat([ashare,fund_etf],axis=0)
    realtimedata.columns=translatecolname(realtimedata,fromto='cte')
    return realtimedata

realtimedata=getname()
codedf=realtimedata.loc[lambda x:x["ts_code"].isin(symparams['ts_code'])]

from pyecharts.charts import Candlestick,Grid
import streamlit_echarts
from pyecharts import options as opts


def candleplot(symbol,startdate,enddate,timew,p_day):
    df =whole(symbol,startdate,enddate,timew,p_day)#
    #symbol='600839'
    name=codedf[codedf['code']==symbol]['name'].sum()
    candle=(Candlestick()
        .add_xaxis(xaxis_data=[i.strftime("%Y-%m-%d") for i in df.index])
        .add_yaxis(series_name=str(symbol)+' '+str(name), y_axis=[list(row) for row in df.values])
        .set_series_opts()
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(
                is_scale=True,#坐标轴自适应最大最小值
                splitline_opts=opts.SplitLineOpts(
                    is_show=True, linestyle_opts=opts.LineStyleOpts(width=1)
                )
            )
    ))
    
    grid=Grid()
    grid.add(candle,grid_opts=opts.GridOpts(pos_left='8%'))
    streamlit_echarts.st_pyecharts(grid,key=symbol)
    return df 



import time
import random

tab1, tab2= st.tabs(["趋势","价格"])

with tab1:
    date_choose=st.date_input(label="choose",value=date.today(),label_visibility="collapsed")
    today_str=date_choose.strftime("%Y%m%d")
    if st.button('更新图'): 
        col1, col2, col3= st.columns(3)
        for i in range(symparams.shape[0]):
            # i=7
            ncol=i%3
            with eval('col'+str(ncol+1)):
                df=candleplot(symbol=symparams.iloc[i,0],startdate=symparams.iloc[i,1],enddate=today_str,timew=symparams.iloc[i,2],p_day=symparams.iloc[i,3])
                st.dataframe(df.tail(5),
                            column_config={
                                "trade_date": st.column_config.DateColumn(format="YYYY-MM-DD")
                    })
                time.sleep(random.uniform(1,5))
    else:
        st.write("")  
    
with tab2:  
    st.markdown("https://tsstock.streamlit.app/")
        
      




