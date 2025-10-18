import pandas as pd
import numpy as np
import os

def get_bureau_processed(bureau):
    """原始信贷数据预处理，生成衍生特征"""
    bureau['BUREAU_ENDDATE_FACT_DIFF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_FACT_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_ENDDATE_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
  
    bureau['BUREAU_CREDIT_DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']
    bureau['BUREAU_CREDIT_DEBT_DIFF'] = bureau['AMT_CREDIT_SUM_DEBT'] - bureau['AMT_CREDIT_SUM']
    
    bureau['BUREAU_IS_DPD'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)
    bureau['BUREAU_IS_DPD_OVER120'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 120 else 0)
    
    return bureau

def aggregate_bureau(bureau, filter_condition=None, prefix='BUREAU'):
    """
    通用的bureau数据聚合函数
    
    参数:
        bureau: 预处理后的bureau DataFrame
        filter_condition: 筛选条件（布尔Series），None表示不筛选
        prefix: 聚合特征的前缀名
        
    返回:
        按SK_ID_CURR聚合后的DataFrame
    """
    # 应用筛选条件
    if filter_condition is not None:
        filtered_bureau = bureau[filter_condition].copy()
    else:
        filtered_bureau = bureau.copy()
    
    # 定义聚合字典（所有聚合函数共用）
    agg_dict = {
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],

        'BUREAU_ENDDATE_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
    }
    
    # 按用户ID分组聚合
    grouped = filtered_bureau.groupby('SK_ID_CURR').agg(agg_dict)
    
    # 重命名列名
    grouped.columns = [f'{prefix}_{"_".join(col).upper()}' for col in grouped.columns.ravel()]
    
    # 重置索引，将SK_ID_CURR从索引变为列
    return grouped.reset_index()

def get_bureau_bal_agg(bureau, bureau_bal):
    """处理信贷余额月度记录的聚合"""
    # 关联获取用户ID
    bureau_bal = bureau_bal.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], on='SK_ID_BUREAU', how='left')
    
    # 衍生逾期特征
    bureau_bal['BUREAU_BAL_IS_DPD'] = bureau_bal['STATUS'].apply(lambda x: 1 if x in ['1','2','3','4','5'] else 0)
    bureau_bal['BUREAU_BAL_IS_DPD_OVER120'] = bureau_bal['STATUS'].apply(lambda x: 1 if x == '5' else 0)
    
    # 定义聚合字典
    agg_dict = {
        'SK_ID_CURR': ['count'],
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'BUREAU_BAL_IS_DPD': ['mean', 'sum'],
        'BUREAU_BAL_IS_DPD_OVER120': ['mean', 'sum']
    }
    
    # 分组聚合
    grouped = bureau_bal.groupby('SK_ID_CURR').agg(agg_dict)
    grouped.columns = [f'BUREAU_BAL_{"_".join(col).upper()}' for col in grouped.columns.ravel()]
    
    return grouped.reset_index()

def get_bureau_agg(bureau, bureau_bal):
    """整合所有聚合特征"""
    # 预处理数据
    processed_bureau = get_bureau_processed(bureau)
    
    # 使用通用聚合函数生成各类聚合特征
    bureau_day_amt_agg = aggregate_bureau(processed_bureau, prefix='BUREAU')
    bureau_active_agg = aggregate_bureau(
        processed_bureau, 
        filter_condition=processed_bureau['CREDIT_ACTIVE'] == 'Active',
        prefix='BUREAU_ACT'
    )
    bureau_days750_agg = aggregate_bureau(
        processed_bureau, 
        filter_condition=processed_bureau['DAYS_CREDIT'] > -750,
        prefix='BUREAU_DAYS750'  # 修正了原代码中的命名问题
    )
    
    # 处理余额数据聚合
    bureau_bal_agg = get_bureau_bal_agg(processed_bureau, bureau_bal)
    
    # 合并所有特征
    bureau_agg = bureau_day_amt_agg.merge(bureau_active_agg, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_bal_agg, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_days750_agg, on='SK_ID_CURR', how='left')
    
    # 衍生比例特征
    bureau_agg['BUREAU_ACT_IS_DPD_RATIO'] = bureau_agg['BUREAU_ACT_BUREAU_IS_DPD_SUM'] / bureau_agg['BUREAU_SK_ID_BUREAU_COUNT']
    bureau_agg['BUREAU_ACT_IS_DPD_OVER120_RATIO'] = bureau_agg['BUREAU_ACT_BUREAU_IS_DPD_OVER120_SUM'] / bureau_agg['BUREAU_SK_ID_BUREAU_COUNT']
    
    return bureau_agg

def process_pos_bal_features(pos_bal):
    """预处理POS余额数据，生成衍生特征"""
    # 生成逾期标记特征（仅保留有用的特征衍生）
    pos_bal['POS_IS_DPD'] = pos_bal['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    pos_bal['POS_IS_DPD_UNDER_120'] = pos_bal['SK_DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
    pos_bal['POS_IS_DPD_OVER_120'] = pos_bal['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)
    return pos_bal

def aggregate_pos_data(pos_data, filter_condition=None, prefix='POS'):
    """通用的POS数据聚合函数"""
    # 应用筛选条件
    if filter_condition is not None:
        filtered_data = pos_data[filter_condition].copy()
    else:
        filtered_data = pos_data.copy()
    
    # 定义聚合字典（只定义一次）
    agg_dict = {
        'SK_ID_CURR': ['count'],
        'MONTHS_BALANCE': ['min', 'mean', 'max'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
        'POS_IS_DPD': ['mean', 'sum'],
        'POS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'POS_IS_DPD_OVER_120': ['mean', 'sum']
    }
    
    # 分组聚合
    grouped = filtered_data.groupby('SK_ID_CURR').agg(agg_dict)
    
    # 重命名列名（修复重复前缀问题）
    grouped.columns = [f'{prefix}_{"_".join(col).upper()}' for col in grouped.columns.ravel()]
    
    return grouped.reset_index()

def get_pos_bal_agg(pos_bal):
    """优化后的POS余额聚合函数，消除了冗余代码"""
    # 预处理数据（仅保留必要的特征衍生）
    processed_pos = process_pos_bal_features(pos_bal)
    
    # 使用通用聚合函数进行全量数据聚合
    pos_bal_agg = aggregate_pos_data(processed_pos, prefix='POS')
    
    # 使用通用聚合函数进行近20个月数据聚合
    cond_months = processed_pos['MONTHS_BALANCE'] > -20
    pos_bal_m20_agg = aggregate_pos_data(processed_pos, cond_months, 'POS_M20')
    
    # 合并聚合结果
    pos_bal_agg = pos_bal_agg.merge(pos_bal_m20_agg, on='SK_ID_CURR', how='left')
    
    return pos_bal_agg

def get_install_processed(install):
    """预处理分期付款数据，生成衍生特征"""
    # 计算还款差异和比例
    install['AMT_DIFF'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
    install['AMT_RATIO'] = (install['AMT_PAYMENT'] + 1) / (install['AMT_INSTALMENT'] + 1)  # +1避免除零
    
    # 计算逾期天数
    install['SK_DPD'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']
    
    # 生成逾期标记特征
    install['INS_IS_DPD'] = install['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    install['INS_IS_DPD_UNDER_120'] = install['SK_DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
    install['INS_IS_DPD_OVER_120'] = install['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)
    
    return install

def aggregate_installments(install, filter_condition=None, prefix='INS'):
    """
    通用的分期付款数据聚合函数
    
    参数:
        install: 预处理后的分期付款DataFrame
        filter_condition: 筛选条件（布尔Series），None表示不筛选
        prefix: 聚合特征的前缀名
        
    返回:
        按SK_ID_CURR聚合后的DataFrame
    """
    # 应用筛选条件
    if filter_condition is not None:
        filtered_install = install[filter_condition].copy()
    else:
        filtered_install = install.copy()
    
    # 定义聚合字典（所有场景共用）
    agg_dict = {
        'SK_ID_CURR': ['count'],
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DAYS_ENTRY_PAYMENT': ['mean', 'max', 'sum'],
        'DAYS_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_PAYMENT': ['mean', 'max', 'sum'],
        'AMT_DIFF': ['mean', 'min', 'max', 'sum'],
        'AMT_RATIO': ['mean', 'max'],
        'SK_DPD': ['mean', 'min', 'max'],
        'INS_IS_DPD': ['mean', 'sum'],
        'INS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'INS_IS_DPD_OVER_120': ['mean', 'sum']
    }
    
    # 分组聚合
    grouped = filtered_install.groupby('SK_ID_CURR').agg(agg_dict)
    
    # 重命名列名（修复原代码中重复前缀的问题）
    grouped.columns = [f'{prefix}_' + '_'.join(col).upper() for col in grouped.columns.ravel()]
    
    # 重置索引
    return grouped.reset_index()

def get_install_agg(install):
    """整合所有分期付款数据的最终聚合函数"""
    # 预处理数据
    processed_install = get_install_processed(install)
    
    # 使用通用聚合函数生成特征
    # 1. 全量数据聚合
    install_agg = aggregate_installments(processed_install, prefix='INS')
    
    # 2. 近365天数据聚合
    cond_day = processed_install['DAYS_ENTRY_PAYMENT'] >= -365
    install_d365_agg = aggregate_installments(
        processed_install, 
        filter_condition=cond_day, 
        prefix='INS_D365'
    )
    
    # 合并聚合结果
    install_agg = install_agg.merge(install_d365_agg, on='SK_ID_CURR', how='left')
    
    return install_agg

def process_card_bal_features(card_bal):
    """预处理信用卡数据，生成衍生特征"""
    # 避免除以0的风险（当额度为0时用1替代）
    limit = card_bal['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, 1)
    
    # 计算额度使用比例特征
    card_bal['BALANCE_LIMIT_RATIO'] = card_bal['AMT_BALANCE'] / limit
    card_bal['DRAWING_LIMIT_RATIO'] = card_bal['AMT_DRAWINGS_CURRENT'] / limit
    
    # 生成逾期标记特征
    card_bal['CARD_IS_DPD'] = card_bal['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    card_bal['CARD_IS_DPD_UNDER_120'] = card_bal['SK_DPD'].apply(
        lambda x: 1 if (x > 0) & (x < 120) else 0
    )
    card_bal['CARD_IS_DPD_OVER_120'] = card_bal['SK_DPD'].apply(
        lambda x: 1 if x >= 120 else 0
    )
    
    return card_bal

def aggregate_card_data(card_data, filter_condition=None, prefix='CARD'):
    """通用的信用卡数据聚合函数"""
    # 应用筛选条件
    if filter_condition is not None:
        filtered_data = card_data[filter_condition].copy()
    else:
        filtered_data = card_data.copy()
    
    # 聚合字典（只定义一次，所有场景共用）
    agg_dict = {
        'SK_ID_CURR': ['count'],
        'AMT_BALANCE': ['max'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'BALANCE_LIMIT_RATIO': ['min', 'max'],
        'DRAWING_LIMIT_RATIO': ['min', 'max'],
        'CARD_IS_DPD': ['mean', 'sum'],
        'CARD_IS_DPD_UNDER_120': ['mean', 'sum'],
        'CARD_IS_DPD_OVER_120': ['mean', 'sum']
    }
    
    # 分组聚合
    grouped = filtered_data.groupby('SK_ID_CURR').agg(agg_dict)
    
    # 重命名列名（修复重复前缀问题）
    grouped.columns = [f'{prefix}_{"_".join(col).upper()}' for col in grouped.columns.ravel()]
    
    return grouped.reset_index()

def get_card_bal_agg(card_bal):
    """优化后的信用卡余额聚合函数，消除冗余代码"""
    # 预处理数据
    processed_card = process_card_bal_features(card_bal)
    
    # 全量数据聚合
    card_bal_agg = aggregate_card_data(processed_card, prefix='CARD')
    
    # 近3个月数据聚合
    cond_month = processed_card['MONTHS_BALANCE'] >= -3
    card_bal_m3_agg = aggregate_card_data(processed_card, cond_month, 'CARD_M3')
    
    # 合并聚合结果（仅需一次reset_index）
    card_bal_agg = card_bal_agg.merge(card_bal_m3_agg, on='SK_ID_CURR', how='left')
    
    return card_bal_agg

def get_prev_processed(prev):
    """
    feature engineering 
    for previouse application credit history
    """
    prev['PREV_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
    prev['PREV_CREDIT_APPL_RATIO'] = prev['AMT_CREDIT']/prev['AMT_APPLICATION']
    # prev['PREV_ANNUITY_APPL_RATIO'] = prev['AMT_ANNUITY']/prev['AMT_APPLICATION']
    prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE']/prev['AMT_APPLICATION']

    # Data Cleansing
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    # substraction between DAYS_LAST_DUE_1ST_VERSION and DAYS_LAST_DUE
    prev['PREV_DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']

    # 1.Calculate the interest rate
    all_pay = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['PREV_INTERESTS_RATE'] = (all_pay/prev['AMT_CREDIT'] - 1)/prev['CNT_PAYMENT']

    return prev

def get_prev_amt_agg(prev):
    """
    feature engineering for the previous credit appliction
    """

    agg_dict = {
      'SK_ID_CURR':['count'],
      'AMT_CREDIT':['mean', 'max', 'sum'],
      'AMT_ANNUITY':['mean', 'max', 'sum'], 
      'AMT_APPLICATION':['mean', 'max', 'sum'],
      'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
      'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
      'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
      'DAYS_DECISION': ['min', 'max', 'mean'],
      'CNT_PAYMENT': ['mean', 'sum'],
        
      'PREV_CREDIT_DIFF':['mean', 'max', 'sum'], 
      'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
      'PREV_GOODS_DIFF':['mean', 'max', 'sum'],
      'PREV_GOODS_APPL_RATIO':['mean', 'max'],
      'PREV_DAYS_LAST_DUE_DIFF':['mean', 'max', 'sum'],
      'PREV_INTERESTS_RATE':['mean', 'max']
    }

    prev_group = prev.groupby('SK_ID_CURR')
    prev_amt_agg = prev_group.agg(agg_dict)

    # multi index 
    prev_amt_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]

    return prev_amt_agg

def get_prev_refused_appr_agg(prev):
    """
    PREV_APPROVED_COUNT : Credit application approved count
    PREV_REFUSED_COUNT :  Credit application refused count
    """
    prev_refused_appr_group = prev[prev['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])].groupby([ 'SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
    # unstack() 
    prev_refused_appr_agg = prev_refused_appr_group['SK_ID_CURR'].count().unstack()

    # rename column 
    prev_refused_appr_agg.columns = ['PREV_APPROVED_COUNT', 'PREV_REFUSED_COUNT' ]

    # NaN
    prev_refused_appr_agg = prev_refused_appr_agg.fillna(0)

    return prev_refused_appr_agg


# DAYS_DECISION
def get_prev_days365_agg(prev):
    """
    DAYS_DESCISION means How many days have been take since the previous credit application made.
    Somehow this feature is important.
    """
    cond_days365 = prev['DAYS_DECISION'] > -365
    prev_days365_group = prev[cond_days365].groupby('SK_ID_CURR')
    agg_dict = {
      'SK_ID_CURR':['count'],
      'AMT_CREDIT':['mean', 'max', 'sum'],
      'AMT_ANNUITY':['mean', 'max', 'sum'], 
      'AMT_APPLICATION':['mean', 'max', 'sum'],
      'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
      'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
      'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
      'DAYS_DECISION': ['min', 'max', 'mean'],
      'CNT_PAYMENT': ['mean', 'sum'],
      
      'PREV_CREDIT_DIFF':['mean', 'max', 'sum'], 
      'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
      'PREV_GOODS_DIFF':['mean', 'max', 'sum'],
      'PREV_GOODS_APPL_RATIO':['mean', 'max'],
      'PREV_DAYS_LAST_DUE_DIFF':['mean', 'max', 'sum'],
      'PREV_INTERESTS_RATE':['mean', 'max']
    }

    prev_days365_agg = prev_days365_group.agg(agg_dict)

    # multi index 
    prev_days365_agg.columns = ["PREV_D365_"+ "_".join(x).upper() for x in prev_days365_agg.columns.ravel()]

    return prev_days365_agg

def get_prev_agg(prev):
    prev = get_prev_processed(prev)
    prev_amt_agg = get_prev_amt_agg(prev)
    prev_refused_appr_agg = get_prev_refused_appr_agg(prev)
    prev_days365_agg = get_prev_days365_agg(prev)
    
    # prev_amt_agg
    prev_agg = prev_amt_agg.merge(prev_refused_appr_agg, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_days365_agg, on='SK_ID_CURR', how='left')
    # SK_ID_CURR APPROVED_COUNT REFUSED_COUNT
    prev_agg['PREV_REFUSED_RATIO'] = prev_agg['PREV_REFUSED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT']
    prev_agg['PREV_APPROVED_RATIO'] = prev_agg['PREV_APPROVED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT']
    # 'PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT' drop 
    prev_agg = prev_agg.drop(['PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT'], axis=1)
    
    return prev_agg

# --------------------------------------------------------------------
# -------- 以下是新增和修改的部分 --------
# --------------------------------------------------------------------

def load_and_aggregate_all_data(data_dir='data/'):
    """
    一键加载所有CSV数据并调用所有聚合函数。
    
    参数:
        data_dir (str): 存放所有CSV文件的目录路径。
        
    返回:
        dict: 一个字典，包含了所有聚合后的DataFrame。
    """
    print("--- 开始加载数据 ---")
    try:
        # 使用 os.path.join 确保路径在不同操作系统上都正确
        bureau = pd.read_csv(os.path.join(data_dir, 'bureau.csv'))
        bureau_bal = pd.read_csv(os.path.join(data_dir, 'bureau_balance.csv'))
        prev_app = pd.read_csv(os.path.join(data_dir, 'previous_application.csv'))
        pos_bal = pd.read_csv(os.path.join(data_dir, 'POS_CASH_balance.csv'))
        install = pd.read_csv(os.path.join(data_dir, 'installments_payments.csv'))
        card_bal = pd.read_csv(os.path.join(data_dir, 'credit_card_balance.csv'))
    except FileNotFoundError as e:
        print(f"错误：加载文件失败。{e}")
        print(f"请确保所有CSV文件都在 '{data_dir}' 目录下。")
        return None
    
    print("--- 数据加载完毕，开始聚合 ---")
    
    # 调用您脚本中的主聚合函数
    # 注意：我们调用 get_bureau_agg，它内部会调用 get_bureau_bal_agg
    bureau_agg = get_bureau_agg(bureau, bureau_bal)
    print("Bureau data 聚合完毕。")
    
    prev_agg = get_prev_agg(prev_app)
    print("Previous application 聚合完毕。")
    
    pos_bal_agg = get_pos_bal_agg(pos_bal)
    print("POS_CASH_balance 聚合完毕。")
    
    install_agg = get_install_agg(install)
    print("Installments 聚合完毕。")
    
    card_bal_agg = get_card_bal_agg(card_bal)
    print("Credit card 聚合完毕。")
    
    print("--- 所有聚合完成 ---")
    
    # 以字典形式返回所有结果，方便后续使用
    return {
        'bureau_agg': bureau_agg,
        'prev_agg': prev_agg,
        'pos_bal_agg': pos_bal_agg,
        'install_agg': install_agg,
        'card_bal_agg': card_bal_agg
    }

# 仅当此脚本被直接运行时，才执行以下代码
if __name__ == "__main__":
    
    # 定义数据目录
    DATA_DIRECTORY = 'data/'
    
    # 调用主函数，获取所有聚合数据
    aggregated_data = load_and_aggregate_all_data(DATA_DIRECTORY)
    
    if aggregated_data:
        # 打印原始脚本末尾要求的形状信息
        print("\n--- 各聚合结果形状 ---")
        print(f"bureau_agg: {aggregated_data['bureau_agg'].shape}")
        print(f"prev_agg: {aggregated_data['prev_agg'].shape}")
        print(f"pos_bal_agg: {aggregated_data['pos_bal_agg'].shape}")
        print(f"install_agg: {aggregated_data['install_agg'].shape}")
        print(f"card_bal_agg: {aggregated_data['card_bal_agg'].shape}")
        
        # 您还可以查看某个聚合结果的头部
        # print("\n--- Bureau Agg Head ---")
        # print(aggregated_data['bureau_agg'].head())