import streamlit as st
import pandas as pd
import joblib

st.title("Prédiction de défaut/ Crédit Populaire d'Algérie")

Impayes_confreres = st.sidebar.selectbox("Impayes_confreres", ('OUI', 'NON'))
Mouvements_confies = st.sidebar.selectbox("Mouvement_confies", ('Quasi intégral', 'Partiel'))
Activite = st.sidebar.selectbox('Activité', ('industries manufacturières', 'Pharmacetique ','industries Agroalimentaires ', 'Service ', 'Import/Export ', 'BTPH ', 'Hydrocarbures_Energie', 'Commerce'))
FR_en_jour_de_CA = st.sidebar.number_input('FR_en_jour_de_CA')
Delai_de_reglement_des_fournisseurs = st.sidebar.number_input('délai de réglement des fournisseurs')
ACT_Stocks_DCT = st.sidebar.number_input('ACT_Stocks/DCT')
Disponibilite_DCT = st.sidebar.number_input('Disponibilité_DCT')
Charges_financieresEBE = st.sidebar.number_input('Charges_financières_sur_EBE')
DCT_Total_dette = st.sidebar.number_input('DCT_Total_dette')
Resultat_net_Fonds_propres = st.sidebar.number_input('Résultat_net_sur_Fonds_propres')
EBE_CA = st.sidebar.number_input('EBE/CA')
Resultat_Financier_CA = st.sidebar.number_input('Résultat_Financier/CA')
Resultat_net_apres_impots_CA = st.sidebar.number_input('Résultat_net_après_impôts/CA')
Dette_financiere_Fonds_propres = st.sidebar.number_input('Dette_financière/Fonds_propres')
Fonds_propres_Passif_Non_courant = st.sidebar.number_input('Fonds_propres/Passif_Non_courant')
Fonds_propres_Total_actif = st.sidebar.number_input('Fonds_propres/Total_actif')
Dette_financiere_Resultat_net = st.sidebar.number_input('Dette_financière/Résultat_net')
Dette_a_long_terme_CAF = st.sidebar.number_input('Dette_à_long_terme/ CAF')

dataset = pd.read_excel('Classeur1xlsx')
dataset_numerical = dataset.drop(['Défaut ', 'Impayés_confrères', 'Mouvements_confiés ', 'Activité'], axis=1)
dataset_categorical = dataset[['Impayés_confrères', 'Mouvements_confiés ', 'Activité']]


input_dict = {'Impayés_confrères': Impayes_confreres, 'Mouvements_confiés ': Mouvements_confies, 'Activité': Activite, 'FR_en_jour_de_CA': FR_en_jour_de_CA, 'Délai_de_règlement_des_fournisseurs \n': Delai_de_reglement_des_fournisseurs, 'ACT_Stocks_DCT': ACT_Stocks_DCT,'Disponibilité_DCT': Disponibilite_DCT, 'Charges_financièresEBE': Charges_financieresEBE, 'DCT_Total_dette': DCT_Total_dette, 'Résultat_net_Fonds_propres': Resultat_net_Fonds_propres, 'EBE_CA': EBE_CA, 'Résultat_Financier_CA': Resultat_Financier_CA, 'Résultat_net_après_impôts_CA': Resultat_net_apres_impots_CA, 'Dette_financière_Fonds_propres': Dette_financiere_Fonds_propres, 'Fonds_propres_Passif_Non_courant': Fonds_propres_Passif_Non_courant, 'Fonds_propres_Total_actif': Fonds_propres_Total_actif, 'Dette_financière_Résultat_net': Dette_financiere_Resultat_net, 'Dette_à_long_terme_CAF': Dette_a_long_terme_CAF}
input_df = pd.DataFrame([input_dict])

input_df_without = input_df.drop(['FR_en_jour_de_CA','Délai_de_règlement_des_fournisseurs \n', 'ACT_Stocks_DCT','Disponibilité_DCT', 'Charges_financièresEBE', 'DCT_Total_dette','Résultat_net_Fonds_propres', 'EBE_CA', 'Résultat_Financier_CA', 'Résultat_net_après_impôts_CA', 'Dette_financière_Fonds_propres', 'Fonds_propres_Passif_Non_courant', 'Fonds_propres_Total_actif', 'Dette_financière_Résultat_net', 'Dette_à_long_terme_CAF'], axis =1)

expanded_columns = ['Impayés_confrères_NON', 'Impayés_confrères_OUI', 'Mouvements_confiés _Partiel', 'Mouvements_confiés _Quasi intégral','Activité_BTPH ', 'Activité_Commerce','Activité_Hydrocarbures, Energie, Mines et services liés','Activité_Import/Export ', 'Activité_Pharmacetique ','Activité_Service ', 'Activité_industries Agroalimentaires ','Activité_industries manufacturières']
new_df = pd.get_dummies(input_df_without).reindex(columns=expanded_columns, fill_value=0)
very_new_df = pd.concat([new_df, input_df[['FR_en_jour_de_CA', 'Délai_de_règlement_des_fournisseurs \n', 'ACT_Stocks_DCT','Disponibilité_DCT', 'Charges_financièresEBE', 'DCT_Total_dette', 'Résultat_net_Fonds_propres', 'EBE_CA', 'Résultat_Financier_CA', 'Résultat_net_après_impôts_CA', 'Dette_financière_Fonds_propres', 'Fonds_propres_Passif_Non_courant', 'Fonds_propres_Total_actif', 'Dette_financière_Résultat_net', 'Dette_à_long_terme_CAF']]], axis=1)

pt_model = joblib.load('adabostu')


def predicter():
     m = pt_model.predict(very_new_df)
     return m


predict_button = st.button('predict outcome', on_click=predicter)


if predict_button:
    result = predicter()
    st.success(result)
