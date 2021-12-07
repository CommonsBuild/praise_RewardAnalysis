# Scripts for the data reward data analysis. THIS IS FOR PROTOTYPING, THE NOTEBOOK IS THE MAIN PRODUCT
# usually the workflow is: play with the idea here with mock data -> pack into methods and have them run in the notebook


from typing import SupportsRound
import pandas as pd
import numpy as np
import analytics_toolbox as tb
from ipyfilechooser import FileChooser


fc_praise = FileChooser('./exampleFiles')

PRAISE_DATA_PATH = "exampleFiles/1500-mock-praise.csv"
SOURCECRED_DATA_PATH = "exampleFiles/1500-mock-sourcecred.csv"
REWARD_BOARD_ADDRESSES_PATH = "exampleFiles/rewardboard-addresses.csv"

NUMBER_OF_PRAISE_REWARD_TOKENS_TO_DISTRIBUTE = 1950
# Right now sourcecred rewards are calculated externally and already specified in the input. This may change.
NUMBER_OF_SOURCECRED_REWARD_TOKENS_TO_DISTRIBUTE = 1950
NUMBER_OF_REWARD_TOKENS_FOR_QUANTIFIERS = 1000
NUMBER_OF_REWARD_TOKENS_FOR_REWARD_BOARD = 100

# distribution algorithm for the praise data. simple for now. It:
#   - Adds the points for each user
#   - Calcs what percentage of the total each user has
#   - Assigns token reward depending on the number of tokens to distribute


def calc_praise_rewards(praiseData, tokensToDistribute):
    # we discard all we don't need and and calculate the % worth of each praise
    slimData = praiseData[['FROM', 'TO', 'FINAL QUANT']].copy()
    totalPraisePoints = slimData['FINAL QUANT'].sum()

    slimData['PERCENTAGE'] = slimData['FINAL QUANT']/totalPraisePoints
    slimData['TOKEN TO RECEIVE'] = slimData['PERCENTAGE'] * tokensToDistribute
    print(slimData)
    return slimData

# generates a new table with combined percentages and added token rewards
# ISSUE: We need single ids


def combine_datasets(praise_data, sourcecred_data):
    processed_praise = prepare_praise(praise_data)
    processed_praise = processed_praise[['IDENTITY', 'PERCENTAGE', 'TOKEN TO RECEIVE']].groupby(
        ['IDENTITY']).agg('sum').reset_index()
    processed_sourcecred = prepare_sourcecred(sourcecred_data)
    combined_dataset = processed_praise.append(
        processed_sourcecred, ignore_index=True)

    combined_dataset = combined_dataset.groupby(
        ['IDENTITY']).agg('sum').reset_index()
    # since we just added to percentages
    combined_dataset['PERCENTAGE'] = combined_dataset['PERCENTAGE'] / 2

    return combined_dataset

# General Helper func. Puts all the "processing we probably won't need to do later or do differently" in one place
#  -removes the '#' and following from discord names
#  -Some renaming and dropping


def prepare_praise(praise_data):
    praise_data['TO'] = (praise_data['TO'].str.split(
        '#', 1, expand=False).str[0]).str.lower()
    praise_data.rename(columns={'TO': 'IDENTITY'}, inplace=True)
    praise_data = praise_data[['IDENTITY', 'PERCENTAGE', 'TOKEN TO RECEIVE']]
    return praise_data

# General Helper func. Puts all the "processing we probably won't need to do later or do differently" in one place
#  -Some renaming and dropping
#  -changing percentages to 0.00-1.00


def prepare_sourcecred(sourcecred_data):
    sourcecred_data.rename(columns={'%': 'PERCENTAGE'}, inplace=True)
    sourcecred_data['IDENTITY'] = sourcecred_data['IDENTITY'].str.lower()
    sourcecred_data['PERCENTAGE'] = sourcecred_data['PERCENTAGE'] / 100
    sourcecred_data = sourcecred_data[[
        'IDENTITY', 'PERCENTAGE', 'TOKEN TO RECEIVE']]
    return sourcecred_data


praise_data = pd.read_csv(PRAISE_DATA_PATH)
sourcecred_data = pd.read_csv(SOURCECRED_DATA_PATH)
rewardboard_addresses = pd.read_csv(REWARD_BOARD_ADDRESSES_PATH)

# print(praise_data)

praise_distribution = calc_praise_rewards(
    praise_data.copy(), NUMBER_OF_PRAISE_REWARD_TOKENS_TO_DISTRIBUTE)

total_period_praise = combine_datasets(
    praise_distribution.copy(), sourcecred_data.copy())
print(praise_distribution)

p_vals = np.array([50, 80, 90, 95, 99])
token_rp = np.array([tb.resource_percentage(
    total_period_praise["TOKEN TO RECEIVE"], p) for p in p_vals])

my_rd_index = [("Top " + str(100 - p) + "%") for p in p_vals]
resource_distribution = pd.DataFrame({"Rewards": token_rp}, index=my_rd_index)
print("\n======= RESOURCE PERCENTAGES ========")
print(resource_distribution)

p_vals = np.array([0, 50, 80])
rewards_gc = np.array([tb.gini_gt_p(
    np.array(total_period_praise["TOKEN TO RECEIVE"]), p) for p in p_vals])

my_index = ["All", "Top 50%", "Top 20%"]
gini_coefs = pd.DataFrame({"rewards": rewards_gc}, index=my_index)
print("\n======= GINI COEFFICIENTS ========")
print(gini_coefs)

entropies_df = pd.DataFrame(data={"Rewards": tb.calc_shannon_entropies(
    total_period_praise["PERCENTAGE"])}, index=["Entropy", "Max Entropy", "% of Max"])
print("\n======= ENTROPIES ========")
print(entropies_df)

ak_coef_IH = tb.nakamoto_coeff(total_period_praise, "PERCENTAGE")
print("\n======= NAKAMOTO COEFFICIENT  ========")
print(ak_coef_IH)


print("=========")
# separate quant table
quant_only = pd.DataFrame()
quantifier_table_buf = praise_data.copy()
quantifier_table_buf.drop(['DATE', 'TO', 'FROM', 'REASON', 'SERVER', 'CHANNEL',
                           'CORRECTION ADD', 'CORRECTION SUB', 'CORRECTION COMMENT', 'FINAL QUANT'], axis=1, inplace=True)
num_of_quants = int((quantifier_table_buf.shape[1] - 1) / 4)
for i in range(num_of_quants):
    q_name = str('QUANT_' + str(i+1) + '_ID')
    q_value = str('QUANT_'+str(i+1))
    buf = quantifier_table_buf[['ID', q_name, q_value]].copy()

    buf.rename(columns={q_name: 'QUANT_ID',
               q_value: 'QUANT_VALUE', 'ID': 'PRAISE_ID'}, inplace=True)
    # print(buf)
    quant_only = quant_only.append(buf.copy(), ignore_index=True)

columnsTitles = ['QUANT_ID', 'PRAISE_ID', 'QUANT_VALUE']
quant_only.sort_values(['QUANT_ID', 'PRAISE_ID'], inplace=True)
quant_only = quant_only.reindex(columns=columnsTitles).reset_index(drop=True)

# count on the quant table:


print(quant_only)
print("=========")
print(quant_only['QUANT_ID'].value_counts())

print("===== PRAISE FLOW ===========")
praise_flow = tb.prepare_praise_flow(
    praise_distribution, n_senders=10, n_receivers=20)
print(praise_flow)
