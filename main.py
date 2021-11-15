from typing import SupportsRound
import pandas as pd 
import numpy as np 
import analytics_toolbox as tb

PRAISE_DATA_PATH = ".\exampleFiles\Praise and SourceCred output data - PRAISE.csv"
SOURCECRED_DATA_PATH = ".\exampleFiles\Praise and SourceCred output data - SOURCECRED.csv"
NUMBER_OF_TOKENS_TO_DISTRIBUTE = 1000

#distribution algorithm for the praise data. simple for now. It:
#   - Adds the points for each user
#   - Calcs what percentage of the total each user has
#   - Assigns token reward depending on the number of tokens to distribute
def calc_praise_rewards(praiseData, tokensToDistribute):
    #we discard all we don't need and add the values
    slimData = praiseData[['TO', 'FINAL QUANT']].groupby(['TO']).agg('sum').reset_index()
    totalPraisePoints = slimData['FINAL QUANT'].sum()

    slimData['PERCENTAGE'] = slimData['FINAL QUANT']/totalPraisePoints
    slimData['TOKEN TO RECEIVE'] = slimData['PERCENTAGE'] * tokensToDistribute
    return slimData

#generates a new table with combined percentages and added token rewards
# ISSUE: We need single ids
def combine_datasets(praise_data, sourcecred_data):
    processed_praise = prepare_praise(praise_data)
    processed_sourcecred = prepare_sourcecred(sourcecred_data)
    combined_dataset = processed_praise.append(processed_sourcecred, ignore_index=True)

    combined_dataset = combined_dataset.groupby(['IDENTITY']).agg('sum').reset_index()
    #since we just added to percentages
    combined_dataset['PERCENTAGE'] = combined_dataset['PERCENTAGE'] / 2


    return combined_dataset

#General Helper func. Puts all the "processing we probably won't need to do later or do differently" in one place
#  -removes the '#' and following from discord names
#  -Some renaming and dropping 
def prepare_praise(praise_data):
    praise_data['TO'] = (praise_data['TO'].str.split('#', 1, expand=False).str[0]).str.lower()
    praise_data.rename(columns = {'TO':'IDENTITY'}, inplace = True)
    praise_data = praise_data[['IDENTITY', 'PERCENTAGE', 'TOKEN TO RECEIVE']]
    return praise_data

#General Helper func. Puts all the "processing we probably won't need to do later or do differently" in one place
#  -Some renaming and dropping 
#  -changing percentages to 0.00-1.00
def prepare_sourcecred(sourcecred_data):
    sourcecred_data.rename(columns = {'%':'PERCENTAGE'}, inplace = True)
    sourcecred_data['IDENTITY'] = sourcecred_data['IDENTITY'].str.lower()
    sourcecred_data['PERCENTAGE'] = sourcecred_data['PERCENTAGE'] / 100
    sourcecred_data = sourcecred_data[['IDENTITY', 'PERCENTAGE', 'TOKEN TO RECEIVE']]
    return sourcecred_data





praise_data = pd.read_csv(PRAISE_DATA_PATH)
sourcecred_data = pd.read_csv(SOURCECRED_DATA_PATH)

praise_distribution = calc_praise_rewards(praise_data, NUMBER_OF_TOKENS_TO_DISTRIBUTE)

total_period_praise =combine_datasets(praise_distribution, sourcecred_data)
print(total_period_praise)

p_vals = np.array([50,80,90,95,99])
IH_rp = np.array([tb.resource_percentage(total_period_praise["TOKEN TO RECEIVE"], p) for p in p_vals])

my_rd_index = [("Top " + str(100 - p) +"%") for p in p_vals]
resource_distribution = pd.DataFrame({"IH": IH_rp}, index = my_rd_index)
print("\n======= RESOURCE PERCENTAGES ========")
print(resource_distribution)

p_vals = np.array([0, 50, 80])
IH_gc = np.array([tb.gini_gt_p(np.array(total_period_praise["TOKEN TO RECEIVE"]), p) for p in p_vals])

my_index = ["All", "Top 50%", "Top 20%"]
gini_coefs = pd.DataFrame({"IH": IH_gc}, index = my_index)
print("\n======= GINI COEFFICIENTS ========")
print(gini_coefs)

entropies_df = pd.DataFrame(data = {"IH" : tb.calc_shannon_entropies(total_period_praise["PERCENTAGE"]) }, index = ["Entropy", "Max Entropy", "% of Max"])
print("\n======= ENTROPIES ========")
print(entropies_df)

ak_coef_IH = tb.nakamoto_coeff(total_period_praise, "PERCENTAGE")
print("\n======= NAKAMOTO COEFFICIENT  ========")
print(ak_coef_IH)