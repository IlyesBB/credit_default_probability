# %% [markdown]
# %%

# %% [markdown]
# # Préparation

# %%
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from joblib import load
import os
from sklearn.linear_model import LinearRegression

# %% [markdown]
# %%

# %%
def agg_numeric(df, group_var, name_df):
    """
    Agrège les variables quantitatives d'une base de donnée, en ignorant les variables de nom contenant 'SK_ID'
    df: Base de données à agréger
    group_var: Variable selon laquelle grouper les données. Si elle est de type object, la fonction ne marche plus
    name_df: Préfixe à donner aux nouvelles variables
    """
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
    df = df.select_dtypes(include=['int', 'float'])  # On récupère les variables entières et flottantes
    df[group_var] = df[group_var]
    df = df.groupby(group_var).agg(['min', 'max', 'mean', 'sum']) # On prend les extremums, la moyenne et la somme
    # On va redéfinir les noms de colonnes de la table
    columns = []
    # Iterate through the variables names
    for var in df.columns.levels[0]:
        # Iterate through the stat names
        for stat in df.columns.levels[1]:
            # Make a new column name for the variable and stat
            columns.append('%s_%s_%s' % (name_df, var, stat))
    df.columns = columns
    return df

# %% [markdown]
# %%

# %%
def agg_categorical(df, group_var, name_df):
    """
    Agrège les variables catégorielles d'une base de donnée, en ignorant les variables de nom contenant 'SK_ID'
    df: Base de données à agréger
    group_var: Variable selon laquelle grouper les données. Si elle est de type object, la fonction ne marche plus
    name_df: Préfixe à donner aux nouvelles variables
    """
    categorical = pd.get_dummies(df.select_dtypes('object'))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])  # On garde la somme et la moyenne
    # On va redéfinir les noms de colonnes de la table
    column_names = []
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        for stat in ['count', 'count_norm']:
            column_names.append('%s_%s_%s' % (name_df, var, stat))
    categorical.columns = column_names
    return categorical

# %% [markdown]
# %%

# %%
def impute_annuity_prev_app(prev_app):
    """Se base sur la corrélation entre la rente et le montant du crédit pour imputer la rente"""
    prev_app_ = prev_app.dropna(subset=['AMT_ANNUITY', 'AMT_CREDIT'])
    X = prev_app_['AMT_CREDIT'].to_numpy()
    X = X.reshape(-1, 1)
    y = prev_app_['AMT_ANNUITY'].to_numpy()
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    credit_nan = prev_app['AMT_CREDIT'][prev_app['AMT_ANNUITY'].isna()]
    annuity_nan = lin_reg.predict(credit_nan.to_numpy().reshape(-1,1))
    annuity_nan = pd.Series(annuity_nan, index=credit_nan.index)
    prev_app.loc[annuity_nan.index, 'AMT_ANNUITY'] = annuity_nan

# %% [markdown]
# %%

# %% [markdown]
# # Jointures   
# Si les noms des colonnes des tables changent, il est possible qu'il y ai des dysfonctionnements

# %%
def join_prev_app_cash(prev_app: pd.DataFrame, cash: pd.DataFrame):
    """
    Ajoute les informations contenues la table POS_cash_balance pour compléter la table previous_application
    Ne prends en compte que les durées de retard de paiement, ainsi que les status à chaque mois
    """
    cash = cash.drop(columns=['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE', 'SK_DPD_DEF'])
    # Sélection des infos les plus récente pour chaque crédit
    last_cash = cash.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False)
    last_cash.drop_duplicates(subset=['SK_ID_PREV'], keep='first', inplace=True)
    # Sélection des crédit actifs
    active_cash = last_cash[(last_cash['NAME_CONTRACT_STATUS']=='Active') | (last_cash['NAME_CONTRACT_STATUS']=='Amortized debt')]
    active_cash.set_index('SK_ID_PREV', inplace=True)
    # Days pas due actifs
    active_dpd = active_cash.loc[:, 'SK_DPD']
    active_dpd.name = 'cash_SK_DPD_ACTIVE'
    prev_app = prev_app.join(active_dpd)
    prev_app['cash_SK_DPD_ACTIVE'].replace(np.nan, 0, inplace=True)
    # dernier statut = actif
    is_active = pd.Series([1]*len(active_cash), index=active_cash.index)
    is_active.name = 'cash_IS_ACTIVE'
    prev_app = prev_app.join(is_active)
    prev_app['cash_IS_ACTIVE'].replace(np.nan, 0, inplace=True)
    # nb_entries
    nb_entries = cash[['SK_ID_PREV', 'SK_ID_CURR']].groupby('SK_ID_PREV').count()['SK_ID_CURR']
    nb_entries.name = 'cash_CNT_ENTRIES'
    prev_app = prev_app.join(nb_entries)
    # Agglomérations systématiques
    cash_agg_quanti = agg_numeric(cash, 'SK_ID_PREV', 'cash')
    cash_agg_categ = agg_categorical(cash, 'SK_ID_PREV', 'cash')
    cash_agg_categ.drop(columns=cash_agg_categ.columns[[bool(re.search('XNA', col)) for col in cash_agg_categ.columns]], inplace=True)
    prev_app = prev_app.join(cash_agg_categ)
    prev_app = prev_app.join(cash_agg_quanti)
    return prev_app

# %% [markdown]
# %%

# %%
def join_prev_app_card(prev_app: pd.DataFrame, card: pd.DataFrame):
    """
    Ajoute les informations contenues la table card_balance pour compléter la table previous_application.
    Ne prends en compte que les durées de retard de paiement, ainsi que les status à chaque mois
    """
    card = card[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'NAME_CONTRACT_STATUS', 'SK_DPD']]
    last_card = card.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False)
    last_card.drop_duplicates(subset=['SK_ID_PREV'], keep='first', inplace=True)
    active_card = last_card[(last_card['NAME_CONTRACT_STATUS']=='Active') | (last_card['NAME_CONTRACT_STATUS']=='Amortized debt')]
    active_card.set_index('SK_ID_PREV', inplace=True)
    # dernier statut = actif
    active_dpd = active_card.loc[:, 'SK_DPD']
    active_dpd.name = 'card_SK_DPD_ACTIVE'
    prev_app = prev_app.join(active_dpd)
    prev_app['card_SK_DPD_ACTIVE'].replace(np.nan, 0, inplace=True)
    # dernier statut = actif
    is_active = pd.Series([1]*len(active_card), index=active_card.index)
    is_active.name = 'card_IS_ACTIVE'
    prev_app = prev_app.join(is_active)
    prev_app['card_IS_ACTIVE'].replace(np.nan, 0, inplace=True)
    # nb_entries
    nb_entries = card[['SK_ID_PREV', 'SK_ID_CURR']].groupby('SK_ID_PREV').count()['SK_ID_CURR']
    nb_entries.name = 'card_CNT_ENTRIES'
    prev_app = prev_app.join(nb_entries)
    # Agglomérations systématiques
    card_agg_quanti = agg_numeric(card, 'SK_ID_PREV', 'card')
    card_agg_categ = agg_categorical(card, 'SK_ID_PREV', 'card')
    prev_app = prev_app.join(card_agg_categ)
    prev_app = prev_app.join(card_agg_quanti)
    return prev_app

# %% [markdown]
# %%

# %%
def prev_app_combine_card_cash(prev_app: pd.DataFrame, card, cash):
    """
    Certaines variables étant les mêmes dans POS_cash_balance et card_balance, on en réuni une partie.
    Notamment, certains status, le nombre de crédits actifs, les durées des retards de paiement...
    """
    status_in_both = [status for status in card['NAME_CONTRACT_STATUS'].unique() if status in cash['NAME_CONTRACT_STATUS'].unique()]
    for status in status_in_both:
        suffix = 'NAME_CONTRACT_STATUS_%s_count' % status
        prev_app[suffix] = prev_app[('card_%s' % suffix)] + prev_app[('cash_%s' % suffix)]
        prev_app.drop(columns=['card_%s' % suffix, 'cash_%s' % suffix], inplace=True)
        suffix_norm = 'NAME_CONTRACT_STATUS_%s_count_norm' % status
        prev_app[suffix_norm] = prev_app[suffix]/(prev_app['card_CNT_ENTRIES'] + prev_app['cash_CNT_ENTRIES'])
        prev_app.drop(columns=['card_%s' % suffix_norm, 'cash_%s' % suffix_norm], inplace=True)
    # is_active
    prev_app['IS_ACTIVE'] = prev_app['card_IS_ACTIVE']+prev_app['cash_IS_ACTIVE']
    prev_app.drop(columns=['card_IS_ACTIVE', 'cash_IS_ACTIVE'], inplace=True)
    prev_app['AMT_ANNUITY_ACTIVE'] = prev_app['IS_ACTIVE']*prev_app['AMT_ANNUITY']
    prev_app['SK_DPD_ACTIVE'] = prev_app['card_SK_DPD_ACTIVE'].combine_first(prev_app['cash_SK_DPD_ACTIVE'])
    prev_app.drop(columns=['card_SK_DPD_ACTIVE', 'cash_SK_DPD_ACTIVE'], inplace=True)
    prev_app['SK_DPD_ACTIVE'].replace(np.nan, 0, inplace=True)
    # Days per due values
    prev_app['SK_DPD_min'] = prev_app['card_SK_DPD_min'].combine(prev_app['cash_SK_DPD_min'], min)
    prev_app.drop(columns=['card_SK_DPD_min', 'cash_SK_DPD_min'], inplace=True)
    prev_app['SK_DPD_max'] = prev_app['card_SK_DPD_max'].combine(prev_app['cash_SK_DPD_max'], max)
    prev_app.drop(columns=['card_SK_DPD_max', 'cash_SK_DPD_max'], inplace=True)
    prev_app['SK_DPD_sum'] = prev_app['card_SK_DPD_sum'] + prev_app['cash_SK_DPD_sum']
    prev_app.drop(columns=['card_SK_DPD_sum', 'cash_SK_DPD_sum'], inplace=True)
    prev_app['SK_DPD_mean'] = prev_app['SK_DPD_sum']/(prev_app['card_CNT_ENTRIES'] + prev_app['cash_CNT_ENTRIES'])
    prev_app.drop(columns=['card_SK_DPD_mean', 'cash_SK_DPD_mean'], inplace=True)
    prev_app.drop(columns=['card_CNT_ENTRIES', 'cash_CNT_ENTRIES'], inplace=True)
    return prev_app

# %% [markdown]
# %%

# %%
def new_features_from_prev_app(prev_app):
    """Agrège les donnée des précédents crédit dans Home Credit pour chaque candidature"""
    count_credits = prev_app[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()['SK_ID_PREV']
    prev_app.set_index('SK_ID_PREV')
    count_credits.name = 'prev_CNT_CREDITS'
    app_supp_numeric = agg_numeric(prev_app, 'SK_ID_CURR', 'prev')
    app_supp_categorical = agg_categorical(prev_app, 'SK_ID_CURR', 'prev')
    app_supp = app_supp_numeric.join(app_supp_categorical)
    return app_supp.join(count_credits)

# %% [markdown]
# %%

# %%
def new_features_from_bureau(bureau):
    """Agrège certaines infos dans bureau pour chaque candidat"""
    bureau=bureau.drop(columns=['CREDIT_CURRENCY', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM',
    'AMT_CREDIT_SUM_LIMIT', 'DAYS_CREDIT_UPDATE', 'CREDIT_TYPE'])
    bureau_active = bureau[bureau['CREDIT_ACTIVE']=='Active']
    bureau_active = bureau_active[bureau_active['DAYS_CREDIT_ENDDATE']>0]
    bureau_active = bureau_active[bureau_active['AMT_CREDIT_SUM_DEBT']>0]
    bureau = bureau.drop(columns=['AMT_CREDIT_SUM_DEBT'])
    bureau_active = bureau_active.drop(columns=['AMT_CREDIT_SUM_DEBT'])
    bureau_active['AMT_ANNUITY'].replace(np.nan, 0, inplace=True)
    # last_status = active
    is_active = pd.Series([1]*len(bureau_active), index=bureau_active.index)
    is_active.name = 'IS_ACTIVE'
    bureau = bureau.join(is_active)
    bureau['IS_ACTIVE'].replace(np.nan, 0, inplace=True)
    # SK_DPD
    days_overdue = bureau_active['CREDIT_DAY_OVERDUE']
    days_overdue.name = 'SK_DPD_ACTIVE'
    bureau = bureau.join(days_overdue)
    bureau['SK_DPD_ACTIVE'].replace(np.nan, 0, inplace=True)
    # AMT_ANNUITY
    annuity = bureau_active['AMT_ANNUITY']
    annuity.name = 'AMT_ANNUITY_ACTIVE'
    bureau = bureau.join(annuity)
    bureau['AMT_ANNUITY_ACTIVE'].replace(np.nan, 0, inplace=True)
    # Agglomérations systématiques
    bureau_agg_quanti = agg_numeric(bureau, 'SK_ID_CURR', 'bureau')
    bureau_agg_categ = agg_categorical(bureau, 'SK_ID_CURR', 'bureau')
    app_supp_bureau = bureau_agg_quanti.join(bureau_agg_categ)
    count_credits = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
    count_credits.name = 'bureau_CNT_CREDITS'
    return app_supp_bureau.join(count_credits)

# %% [markdown]
# %%

# %%
def add_new_features(app, app_supp):
    """Réunis certaines variables dans app_supp, et ajoute ses variables aux candidatures"""
    for annuity_status in ['_ACTIVE', '']:
        suffix = 'AMT_ANNUITY%s_sum' % annuity_status
        app_supp[suffix] = app_supp[('prev_%s' % suffix)]+app_supp[('bureau_%s'% suffix)]
        app_supp.drop(columns=['prev_%s' % suffix, 'bureau_%s'% suffix], inplace=True)
        suffix = 'AMT_ANNUITY%s_max' % annuity_status
        app_supp[suffix] = app_supp[('prev_%s' % suffix)].combine(app_supp[('bureau_%s'% suffix)], max)
        app_supp.drop(columns=['prev_%s' % suffix, 'bureau_%s'% suffix], inplace=True)
        suffix = 'AMT_ANNUITY%s_min' % annuity_status
        app_supp[suffix] = app_supp[('prev_%s' % suffix)].combine(app_supp[('bureau_%s'% suffix)], min)
        app_supp.drop(columns=['prev_%s' % suffix, 'bureau_%s'% suffix], inplace=True)
    app_supp['CNT_CREDITS'] = app_supp['prev_CNT_CREDITS']+app_supp['bureau_CNT_CREDITS']
    app_supp.drop(columns=['prev_CNT_CREDITS', 'bureau_CNT_CREDITS'], inplace=True)
    app_supp['CNT_ACTIVE'] = app_supp['prev_IS_ACTIVE_sum']+app_supp['bureau_IS_ACTIVE_sum']
    for calculation in ['max', 'min', 'sum', 'mean']:
        app_supp.drop(columns=['prev_IS_ACTIVE_%s' % calculation, 'bureau_IS_ACTIVE_%s' % calculation], inplace=True)
    app_supp['SK_DPD_max'] = app_supp['prev_SK_DPD_max_max'].combine(app_supp['bureau_CREDIT_DAY_OVERDUE_max'], max)
    app_supp.drop(columns=['bureau_CREDIT_DAY_OVERDUE_max'], inplace=True)
    app_supp['SK_DPD_min'] = app_supp['prev_SK_DPD_min_min'].combine(app_supp['bureau_CREDIT_DAY_OVERDUE_min'], min)
    app_supp.drop(columns=['bureau_CREDIT_DAY_OVERDUE_min'], inplace=True)
    app_supp['SK_DPD_sum'] = app_supp['prev_SK_DPD_sum_sum'] + app_supp['bureau_CREDIT_DAY_OVERDUE_sum']
    app_supp['SK_DPD_mean'] = (app_supp['prev_SK_DPD_mean_sum']+app_supp['bureau_CREDIT_DAY_OVERDUE_sum'])/app_supp['CNT_CREDITS']
    app_supp.drop(columns=['bureau_CREDIT_DAY_OVERDUE_sum'], inplace=True)
    for calculation in ['max', 'min', 'sum', 'mean']:
        for calculation2 in ['max', 'min', 'sum', 'mean']:
            app_supp.drop(columns=['prev_SK_DPD_%s_%s' % (calculation, calculation2)], inplace=True)
    # dpd active
    app_supp['SK_DPD_ACTIVE_max'] = app_supp['prev_SK_DPD_ACTIVE_max'].combine(app_supp['bureau_SK_DPD_ACTIVE_max'], max)
    app_supp.drop(columns=['SK_DPD_ACTIVE_max', 'bureau_SK_DPD_ACTIVE_max'], inplace=True)
    app_supp['SK_DPD_ACTIVE_min'] = app_supp['prev_SK_DPD_ACTIVE_min'].combine(app_supp['bureau_SK_DPD_ACTIVE_min'], min)
    app_supp.drop(columns=['prev_SK_DPD_ACTIVE_min', 'bureau_SK_DPD_ACTIVE_min'], inplace=True)
    app_supp['SK_DPD_ACTIVE_sum'] = app_supp['prev_SK_DPD_ACTIVE_sum'] + app_supp['bureau_SK_DPD_ACTIVE_sum']
    app_supp.drop(columns=['prev_SK_DPD_ACTIVE_sum', 'bureau_SK_DPD_ACTIVE_sum'], inplace=True)
    app = app.set_index('SK_ID_CURR')
    app = app.join(app_supp)
    app['AMT_ANNUITY_ACTIVE_TOTAL'] = app['AMT_ANNUITY']+app['AMT_ANNUITY_ACTIVE_sum']
    return app

# %% [markdown]
# %% [markdown]<br>
# Traitons à présent les variables catégorielles à valeurs dans object. On va utiliser un encodeur binaire pour les variables à 2 valeurs et un encodeur onehot pour les autres.

# %% [markdown]
# %%

# %%
def encode_labels(col, application, encoder=None):
    """Réalise l'encodage d'une variable catégorielle à 2 valeurs et retourne l'encodeur ainsi que le résultat de l'encadage"""
    application[col].replace(np.nan,'NaN', inplace=True)
    if not encoder:  # Si c'est la table de test
        encoder = LabelEncoder()
        encoder.fit(application[col])
    col_values = pd.Series(encoder.transform(application[col]), index=application.index)
    return encoder, col_values

# %% [markdown]
# %%

# %%
def encode_onehot(cols, application, encoder=None):
    """Réalise l'encodage onehot d'une variable catégorielle et retourne l'encodeur ainsi que le résultat de l'encadage"""
    application.loc[:, cols].replace(np.nan,'NaN', inplace=True)
    if not encoder:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoder.fit(application[cols])
    mat_cols = encoder.transform(application[cols])
    application_categorical = pd.DataFrame(data=mat_cols, columns=encoder.get_feature_names_out(), index=application.index)
    application = pd.concat([application, application_categorical], axis=1)
    application.drop(columns=cols, inplace=True)
    return encoder, application

# %% [markdown]
# %%

# %%
def less_filled(features, percentage):
    """Sélectionne la variable avec le pourcentage le plus faible entre deux variables."""
    feature1, feature2 = features.iloc[0], features.iloc[1]
    if percentage[feature1] > percentage[feature2]:
        aux = feature1
        feature1 = feature2
        feature2 = aux
    if feature1 in ['AMT_INCOME_TOTAL', 'AMT_ANNUITY']:
        aux = feature1
        feature1 = feature2
        feature2 = aux
    return feature1

# %% [markdown]
# %% 

# %% [markdown]
# # Mise en commun

# %%
def engineer_features(app, bureau, prev_app, card, cash):
    """Compile l'ensemble des fonctions nécessaire au feature engineering"""
    impute_annuity_prev_app(prev_app)
    prev_app = join_prev_app_cash(prev_app, cash)
    prev_app = join_prev_app_card(prev_app, card)
    prev_app = prev_app_combine_card_cash(prev_app, card, cash)
    app_supp = new_features_from_prev_app(prev_app)
    app_supp_bureau = new_features_from_bureau(bureau)
    app_supp = app_supp.join(app_supp_bureau)
    app = add_new_features(app, app_supp)
    return app

# %% [markdown]
# %%

# %%
def preprocess_data(app_test, bureau, prev_app, card, cash, onehot_enc, label_encs, mean_imputer, freq_imputer, scaler):
    """Compile l'ensemble des pré-traitements réalisés"""
    app_test = engineer_features(app_test, bureau, prev_app, card, cash)
    client_id = app_test.index
    for feature, label_encoder in label_encs.items():
        _, app_test[feature] = encode_labels(feature, app_test, encoder=label_encoder)
    _, app_test = encode_onehot(onehot_enc.feature_names_in_, app_test, encoder=onehot_enc)
    regexp1 = "nan"
    regexp2 = "XNA"
    cols_nan = np.array([bool(re.search(regexp1, element)) or bool(re.search(regexp2, element)) for element in app_test.columns])
    cols_nan = app_test.columns[cols_nan]
    app_test.drop(columns=cols_nan, inplace=True)
    mean_input, freq_input = mean_imputer.feature_names_in_.tolist(), freq_imputer.feature_names_in_.tolist()
    to_drop = app_test.columns[~app_test.columns.isin(mean_input+freq_input)]
    app_test.drop(columns=to_drop, inplace=True)
    app_test.replace(to_replace=np.inf, value=np.nan, inplace=True)
    app_test.loc[:,mean_input] = mean_imputer.transform(app_test[mean_input])
    app_test.loc[:,freq_input] = freq_imputer.transform(app_test[freq_input])
    mat_test = scaler.transform(app_test)
    app_test_new = pd.DataFrame(mat_test, columns=app_test.columns, index=client_id)
    return app_test_new

# %% [markdown]
# %%

# %%
def load_joblibs(job_dir):
    """Permet de charger les encodeurs, les imputeurs et le scaler depuis le dossier en argument"""
    onehot_encoder = load(job_dir+'/onehot_enc.joblib')
    mean_imputer = load(job_dir+'/mean_imputer.joblib')
    freq_imputer = load(job_dir+'/freq_imputer.joblib')
    regexp = 'label_encoder-'
    label_encoders = {}
    for file_name in os.listdir(os.getcwd()+'/joblib'):
        if re.match(regexp, file_name):
            feature = file_name.split('-')[1].split('.')[0]
            label_encoders[feature] = load(job_dir+'/'+file_name)
    scaler = load(job_dir+'/scaler.joblib')
    return onehot_encoder, label_encoders, mean_imputer, freq_imputer, scaler

# %% [markdown]
# %

# %%
def preprocess_data_from_joblib(app, bureau, joblib_dir):
    """Enchaîne le chargement des modèles et le pré-traitement"""
    onehot_enc, label_enc, mean_imputer, freq_imputer, scaler = load_joblibs(joblib_dir)
    app = preprocess_data(app, bureau, onehot_enc, label_enc, mean_imputer, freq_imputer, scaler)
    return app

# %% [markdown]
# %

# %%
def drop_outliers(data, col, n_iqr=3):
    """Supprime les lignes contenant des outliers selon une colonne"""
    q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    iqr = q3-q1
    not_outlier = np.logical_and(q1-n_iqr*iqr <= data[col], data[col] <= q3+3*n_iqr)
    not_delete = np.logical_or(not_outlier, data[col].isna())
    data.drop(index=data.index[~not_delete], inplace=True)

# %% [markdown]
# %

# %%
def drop_outliers_all(data, cols, n_iqr=3):
    """Supprime les lignes contenant des outliers selon plusieurs colonnes"""
    for col in cols:
        drop_outliers(data, col, n_iqr)
