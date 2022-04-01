# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from joblib import load
from sklearn.impute import SimpleImputer
import os
from preprocessing_functions import load_joblibs, preprocess_data


# Chargement des données, des encodeurs et des imputeurs
data_dir = './home-credit-default-risk'
most_freq_imputer: SimpleImputer
mean_imputer: SimpleImputer
scaler: StandardScaler
onehot_encoder, label_encoders, mean_imputer, most_freq_imputer, scaler = load_joblibs(
    './joblib')
model: LogisticRegression = load('./joblib/adaboost.joblib')

# Chargement des données pré-traitées, et re-mise à l'échelle
# app: Données de test
app = pd.read_csv(data_dir+'/application_test_preprocessed.csv')
app.set_index('SK_ID_CURR', inplace=True)
# app_: Données de test à l'échelle d'origine
app_ = scaler.inverse_transform(app)
app_ = pd.DataFrame(app_, index=app.index, columns=app.columns)
# app_train: Données d'entraînement
app_train = pd.read_csv(data_dir+'/application_train_preprocessed.csv')
app_train.set_index('SK_ID_CURR', inplace=True)
target = app_train['TARGET']
app_train.drop(columns=['TARGET'], inplace=True)
# app_train_: Données d'entraînement à l'échelle d'origine
app_train_ = scaler.inverse_transform(app_train)
app_train_ = pd.DataFrame(app_train_, index=app_train.index, columns=app_train.columns)

# curr_id: Identifiant de crédit initial
curr_id = app.index[0]


def predict_score(input_data):
    """Utilise les données formatées d'un crédit pour retourner le score associé à partir de l'API Mlflow."""
    # res: Va contenir le score final
    res = os.popen('curl http://127.0.0.1:5000/invocations -H \'Content-Type: application/json\' -d \'{"data": [%s]}\'' % input_data.tolist())
    res = res.read()
    res = res.split(',')[1]
    res = float(res[:-2])
    return res
    
# On enregistre les targets d'entraînement dans la table 'd'origine'
app_train_['PREDICTION'] = target

dashboard = Dash(__name__)

# colors: Dictionnaire des couleurs de base de l'application
colors = {
    'background': 'rgb(125,150,125)',
    'text': '#FFFFFF',
    'cell': 'rgb(175,220,125)'
}
font = 'Verdana, sans-serif'

# On va sélectionner les features les plus importantes et les rendre modulables
# features, coefs: Liste des features et liste de leurs coefficients
features, coefs = app.columns, model.feature_importances_
# coef_feature: Liste des tuples (feature, coefficient)
coef_feature = [(feature, float(coef)) for feature, coef in zip(features, coefs)]
coef_feature.sort(key=lambda x: abs(float(x[1])), reverse=True)
# dict_coefs: On enregistre l'ensemble des coefficients dans dictionnaire
dict_coefs = {feature: float(coef) for feature, coef in coef_feature}
# On sélectionne les 15 features les plus importantes
coef_feature = coef_feature[:min(len(coef_feature),15)]
# features: Liste des features les plus importantes
features = [tup[0] for tup in coef_feature]


"""
Ajout des éléments d'introduction: Le titre principal, les couleurs, Les KPIS, l'identifiant du crédit notamment
"""

# kpis: Liste des éléments html des KPIS (score, rente et revenus annuels)
kpis = [
    html.Div(children=[
        html.Label('', id='score', title='Probability of default', style={'font-family':font})
    ], style={'padding': 10, 'flex': 1}),

    html.Div(children=[
        html.Label('Annuity: ', title="Feature importance: %.2f" % dict_coefs['AMT_ANNUITY'], style={'font-family':font}),
        dcc.Input('', id='annuity', type='number')
    ], style={'padding': 10, 'flex': 1}),

    html.Div(children=[
        html.Label('Income: ', title="Feature importance: %.2f" % dict_coefs['AMT_INCOME_TOTAL'], style={'font-family':font}),
        dcc.Input('', id='income', type='number')
    ], style={'padding': 10, 'flex': 1})
]

# children_layout: Liste des éléments html d'introduction (titre, id, KPIS)
children_layout = [
    html.H1(
        children='CREDIT DASHBOARD',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-family':font
        }
    ),

    html.Div(children='Evaluate a client\'s probability of credit default', style={
        'textAlign': 'center',
        'color': colors['text'],
        'font-family':font
    }),

    html.Datalist(children=app.index.map(int).tolist(), id='id_list'),

    html.Div(children=[
        html.Label('Loan ID: ', style={'font-family':font}),
        dcc.Input(value=curr_id, type='number', id='id', list='id_list')
    ], style={'padding': 10, 'flex': 1, 'margin-top':'2em'}),

    html.H2(children='KPIS', style={'text-align': 'center', 'color':colors['text'], 'font-family':font, 'margin-top':'2em'}),

    html.Div(children=kpis, style={'border-style': 'solid', 'border-radius':10, 'border-width':0.2, 'box-shadow': '5px  5px 5px gray', 'display':'flex', 'flex':'1 1 1', 'flex-direction':'row',
        'flex-wrap': 'wrap', 'background-color':colors['cell']})
]


"""
Ajout du bloc de paramétrage des features
"""

# categ_feature: Liste des variables catégorielles de notre sélection de features
categ_feature = [feature for feature in features if feature in most_freq_imputer.feature_names_in_]
# quanti_feature: Liste des variables quantitatives de notre sélection de features
quanti_feature = [feature for feature in features if feature in mean_imputer.feature_names_in_]

# children_features_input: Va contenir les éléments html paramétrables
children_features_input = []

# On boucle sur les variables catégorielles, en ajoutant un block de paramétrage pour chacune
for feature in categ_feature:
    children_features_input.append(html.Div(
        children=[
            html.Label('%s: ' % feature, title="Feature importance: %.2f" % dict_coefs[feature], style={'font-family':font}),
            dcc.Dropdown([0, 1], id=(feature+'_in'))  # Il n'y à que 2 possibilités pour une catégorie
        ],
        style = {'padding': 10, 'flex':  'auto'}
    ))
# On boucle sur les variables quantitatives, en ajoutant un block de paramétrage pour chacune
for feature in quanti_feature:
    children_features_input.append(html.Div(
        children=[
            html.Label('%s: ' % feature, title="Feature importance: %.2f" % dict_coefs[feature], style={'font-family':font}), 
            html.Br(),
            dcc.Input(type='number', id=(feature+'_in'), style={'margin-top':'5px'})],  # On peux entrer n'importequel nombre
        style = {'padding': 10, 'flex': '1 1 1'}
    ))
# features_per_row: Nombre de features dans une colonne du bloc de paramétrage
feature_per_row = 4
# children_features_input_wrap: Va contenir les colonnes de features
children_features_input_wrap = []
# On boucle sur les pacquets de features, chacun dans une colonne
for i in range(0, len(children_features_input), feature_per_row):
    i_max = min(len(children_features_input), i+feature_per_row)
    children_features_input_wrap.append(
        html.Div(children=children_features_input[i:i_max])
    )
# Ajout du titre précédent le bloc de paramétrage
children_layout.append(
    html.H2(children='FEATURES',
        style={'text-align': 'center', 'color':colors['text'], 'font-family':font, 'margin-top':'2em'}
    )
)
# Ajout du bloc de paramétrage
children_layout.append(
    html.Table(
        children=children_features_input_wrap, 
        style={'border-style': 'solid', 'border-radius':10, 'border-width':0.2, 'box-shadow': '5px  5px 5px gray', 'display':'flex', 'flex':'1 1 1', 'flex-direction':'row',
        'flex-wrap': 'wrap', 'background-color':colors['cell']}
    )
)


"""
Ajout des graphes (Importance locale, distribution des features sur la target)
Pour la distribution des features, on représente un diagramme en barre pour les variables catégorielles et 
un diagramme en boîte pour les variables quantitatives.
"""

# Ajout du titre précédent le graphe des importances locales des features
children_layout.append(html.H2(children='LOCAL IMPORTANCE', style={'text-align': 'center', 'color':colors['text'], 'font-family':font, 'margin-top':'2em'}))
# Ajout du bloc qui va contenir le graphe des importances locale. Il sera rempli dans un callback
children_layout.append(html.Div(id='local_importance'))

# children_features_graph: Liste contenant un graphe pour chaque feature
children_features_graphs = []

# On boucle sur les variables catégorielles
for feature in categ_feature:
    app_train_[feature] = app_train_[feature].map(round).map(int)  # On formatte les valeurs de la feature
    title = ''
    fig = px.histogram(app_train_, x='PREDICTION', color=feature, opacity=0.8, title=title, barmode='group', histnorm='percent',
    text_auto=True, category_orders={feature: [0, 1]})

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    graph = dcc.Graph(
        id=feature,
        figure=fig
    )
    children_features_graphs.append(graph)

# On boucle sur les variables quantitatives
for feature in quanti_feature:
    title = ''
    fig = px.box(app_train_, x='PREDICTION', y=feature, title=title)

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    graph = dcc.Graph(
        id=feature,
        figure=fig
    )
    children_features_graphs.append(graph)

# Ajout du titre précédent les graphes comparatifs
children_layout.append(html.H2(children='COMPARISON', style={'text-align': 'center', 'color':colors['text'], 'font-family':font, 'margin-top':'2em'}))
# Ajout de l'ensemble des graphes à notre layout
children_layout.append(
    html.Div(children=children_features_graphs, style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap':'wrap'})
)

# Finalement, on ajoute l'ensemble de nos éléments dans notre layout final 
dashboard.layout = html.Div(
    style={'margin-top': '10px', 'border-collapse': 'separate', 'background-color':colors['background']},
    children=children_layout
)


"""
Écriture des callbacks: 
- Les valeurs des features doivent être mises à jour en cas de modification de l'identifiant
- Le score doit être mise à jour en cas de modification de la valeur d'une feature
- Le graphe des importances locales doit être mis à jour en cas de modification des valeurs des features
"""

# inputs_score: Inputs des features les plus importantes en plus de la rente, des revenus et de l'identifiant
inputs_score = [Input(feature+'_in', 'value') for feature in features]
inputs_score.append(Input('income', 'value'))
inputs_score.append(Input('annuity', 'value'))
inputs_score.append(Input('id', 'value'))
@dashboard.callback(
    Output('score', 'children'),
    *inputs_score
)
def update_score(*args):
    """Met à jour le score en fonction des inputs à disposition"""
    # client_id: Identifiant du prêt
    client_id = args[-1]
    # args: Contient les features importantes. Les deux derniers éléments sont la rente et les revenus
    args = args[:-1]
    # client_data: Variables enregistrées pour le client. On les met à jour avec les valeurs du panneau de paramétrage
    client_data = app_.loc[client_id].copy()
    for ind, feature in enumerate(features):
        client_data[feature] = args[ind] if args[ind] is not None else client_data[feature]
    # On met à jour aussi la rente et les revenus
    client_data['AMT_INCOME_TOTAL'] = args[-2] if args[-2] is not None else client_data['AMT_INCOME_TOTAL']
    client_data['AMT_ANNUITY'] = args[-1] if args[-1] is not None else client_data['AMT_ANNUITY']
    client_data = scaler.transform([client_data])[0]
    return "Score: %.2f" % predict_score(client_data)

# outputs_features: Outputs des features les plus importantes 
outputs_features = [Output(feature+'_in', 'value') for feature in features]
@dashboard.callback(
    *outputs_features,
    Output('income', 'value'),
    Output('annuity', 'value'),
    Input('id', 'value')
)
def update_feature_values(id_client):
    """Met à jour les valeurs des features en cas de modification de l'identifiant"""
    to_return = app_.loc[id_client].copy()
    to_return[categ_feature] = to_return[categ_feature].map(round).map(int)
    to_return[quanti_feature] = to_return[quanti_feature].map(float)
    return to_return[features+['AMT_INCOME_TOTAL', 'AMT_ANNUITY']].tolist()

# features_median: Contient la médiane à l'échelle d'origine pour chaque feature importante.
features_median = app_train.median()
features_median.drop(index=features_median.index[~features_median.index.isin(features)], inplace=True)
@dashboard.callback(
    Output('local_importance', 'children'),
    *(inputs_score[:-3]+[inputs_score[-1]])  # Les inputs des KPIS ne sont pas pris en compte
)
def update_local_importances(*args):
    """Crée l'histogramme des importances des features."""
    client_id = args[-1]
    args = args[:-1]
    client_data = app_.loc[client_id].copy()
    for ind, feature in enumerate(features):
        client_data[feature] = args[ind] if args[ind] is not None else client_data[feature]
    client_data = pd.Series(scaler.transform([client_data])[0], index=client_data.index)
    loc_importance = features_median.copy()
    for feature in features:
        loc_importance.loc[feature] = abs(loc_importance.loc[feature]-client_data.loc[feature])
        loc_importance.loc[feature] = loc_importance.loc[feature]*abs(dict_coefs[feature])
    loc_importance.name = "Local importance"
    loc_importance = loc_importance.to_frame()
    loc_importance['Feature'] = loc_importance.index
    fig = px.bar(loc_importance, x='Feature', y='Local importance')
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    dashboard.run_server(debug=True)
