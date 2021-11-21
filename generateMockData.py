from os import replace
from numpy.lib.npyio import save
import pandas as pd 
import numpy as np 
from numpy.random import default_rng
import matplotlib.pyplot as plt
from math import sqrt
import argparse
from datetime import datetime
from pathlib import Path


PRAISE_VALUES = [0, 13, 21, 55, 144]




rng = default_rng()


def generate_praise_dataset(number_of_users, total_number_of_praises, number_of_quants, quants_per_praise):
    praise_id = list(range(1001, (1001+total_number_of_praises)))
 
    user_id = (rng.pareto(3, int(total_number_of_praises*3/2))*100).astype(int)
    user_id = user_id[user_id<number_of_users][:total_number_of_praises]
    # alternative normal distribution
    #user_id = rng.normal(loc= number_of_users * 0.5, scale=sqrt(number_of_users*0.5*0.5), size=total_number_of_praises).astype(int)
    
    from_id = rng.integers(0, number_of_users, total_number_of_praises)
    praise_value = rng.integers(0, 5, total_number_of_praises)
    quant_id = rng.integers(0, number_of_quants, total_number_of_praises)

    #generat mock text columns for "date", "reason", "server" and "channel"
    mock_date = ["01/01/2021-00:00:00"] * total_number_of_praises
    mock_reason = ["mock reason text"] * total_number_of_praises
    mock_server= ["mock server"] * total_number_of_praises
    mock_channel= ["mock channel"] * total_number_of_praises


    df = pd.DataFrame(dict(
            PRAISE_ID= praise_id,
            DATE = mock_date,
            USER_ID = user_id,
            FROM_ID = from_id,
            REASON = mock_reason,
            SERVER = mock_server,
            CHANNEL = mock_channel,
            QUANT_1 = praise_value,
            QUANT_1_ID = quant_id
        ))


    df_output  = df.copy()
    #add  quants_per_praise - 1 columns to each row, calc the praise value
    for i in range(1, quants_per_praise):
        col1_name = "QUANT_" + str(i+1)
        col2_name = col1_name + "_ID"

        rand_modifiers= rng.choice(range(-2,3), total_number_of_praises , p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        #some nice Gandhi Nukes here... probably worth revisiting if this is to become serious
        df_output[col1_name] = (df_output['QUANT_1'] + rand_modifiers ) % len(PRAISE_VALUES)
        df_output[col2_name] = (df_output['QUANT_1_ID'] + i) % number_of_quants

        #replace with the "real" numbers
        df_output[col1_name] = df_output[col1_name].apply(lambda x: PRAISE_VALUES[x])

    #replace in the orginal column too   
    df_output['QUANT_1'] = df_output['QUANT_1'].apply(lambda x: PRAISE_VALUES[x])

    #add "avg quant" column
    list_of_averages = []
    for i in range(len(df_output)):
        score_list = []
        for j in range(1, quant_per_praise+1):
            col_name = "QUANT_" + str(j)
            score_list.append(df_output.iloc[i][col_name])
        
        #here would be the place to make more sophisticated weightings (like dismissing highest and lowest value)
        avg = (sum(score_list)/len(score_list)).astype(int)

        list_of_averages.append(avg)

    df_output['AVG QUANT'] = list_of_averages

    #generate dupliations, dismissals, correctons, etc
    df_output['DUPLICATE ID'] = ''
    df_output['DISMISSED'] = ''
    df_output['CORRECTION ADD'] = ''
    df_output['CORRECTION SUB'] = ''
    df_output['CORRECTION COMMENT'] = ''
    df_output['FINAL QUANT'] = df_output['AVG QUANT'].copy()

    #10% of the praise gets set apart for dismissal/ duplication / add / sub : 2.5% each
    sample = rng.choice(total_number_of_praises, int(total_number_of_praises/10) , replace=False)
    p1 = int(len(sample)*0.25)
    p2 = int(len(sample)*0.5)
    p3 = int(len(sample)*0.75)


    #the modification is capped: maximum is doubling the average score, minimum reducing to 0 (if avg is 0 then we add smth betweeen 0-50)
    for i in range(len(sample)):
        if i < p1 :
            #dismiss
            df_output.loc[sample[i],'DISMISSED'] = 'TRUE'
            df_output.loc[sample[i],'FINAL QUANT'] = '0'
        elif i < p2:
            #duplicate
            df_output.loc[sample[i], 'DUPLICATE ID'] = rng.choice(praise_id)
            df_output.loc[sample[i], 'FINAL QUANT'] = '0'
        elif i < p3:
            #add
            rand_add = rng.integers(0, df_output.loc[sample[i], 'AVG QUANT']) if df_output.loc[sample[i],'AVG QUANT'] != 0 else rng.integers(0, 50)
            df_output.loc[sample[i], 'CORRECTION ADD'] = rand_add
            df_output.loc[sample[i], 'FINAL QUANT'] = df_output.loc[sample[i], 'AVG QUANT'] + rand_add
            df_output.loc[sample[i], 'CORRECTION COMMENT'] = 'addition comment'
        else:
            #substract
            if df_output.loc[sample[i], 'AVG QUANT'] == 0:
                continue
            rand_sub = rng.integers(0, df_output.loc[sample[i], 'AVG QUANT']) 
            df_output.loc[sample[i],'CORRECTION SUB'] = rand_sub
            df_output.loc[sample[i], 'FINAL QUANT'] = df_output.loc[sample[i], 'AVG QUANT'] - rand_sub
            df_output.loc[sample[i], 'CORRECTION COMMENT'] = 'substraction comment'




    #rename for correct output:
    df_output.rename(columns = {'PRAISE_ID':'ID', 'USER_ID':'TO', 'FROM_ID': 'FROM' }, inplace = True)
    df_output['TO'] = df_output['TO'].apply(lambda x: "bot" + str(df_output.loc[x, 'TO']) + "#" + str(rng.integers(0,2000)))
    #print(df_output)

    return df_output

def generate_sourcecred_dataset( number_of_users = 50, number_of_tokens=1000):
    user_id = list(range(0, number_of_users))
    user_grain = list((rng.pareto(3, size=number_of_users)*1000).astype(int))
    #alternative normal distribution
    #user_grain = rng.normal(loc= 125, scale=sqrt(125*0.5*0.5), size=number_of_users).astype(int)
    

    df = pd.DataFrame(dict(
            IDENTITY= user_id,
            AMOUNT = user_grain,
        ))

    total_grain = df["AMOUNT"].sum()
    df["%"] = df["AMOUNT"]/total_grain
    df["TOKEN TO RECEIVE"] = df["%"] * number_of_tokens
    
    #rename for equivalency with praise
    df['IDENTITY'] = df['IDENTITY'].apply(lambda x: "bot" + str(df.loc[x, 'IDENTITY']) )


    return df

def save_dataset(name, df):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d-%H%M%S-")
    filename = ("mockDatasets/" + dt_string + name + ".csv")
    df.to_csv (filename, index = False, header=True)






parser = argparse.ArgumentParser(description='Create Datasets for praise Analysis testing.')
parser.add_argument("-u", "--user_num", type=int, help="Number of unique users in the system. OPTIONAL, defaults to 50")
parser.add_argument("-p", "--praise_num", type=int, help="The number of unique praises to generate. OPTIONAL, defaults to 500")
parser.add_argument("-q", "--quant_num", type=int, help="Number of quantifiers in the system. OPTIONAL, defaults to 10")
parser.add_argument("-qxp", "--quant_per_praise", type=int, help="How many different quantifiers we want to have review each praise. OPTIONAL, defaults to 3")
parser.add_argument("-t", "--token_num", type=int, help="Number of tokens to distribute with SourceCred. OPTIONAL, defaults to 1000")
parser.add_argument('--onlyPraise', action='store_true', help='Generate mock dataset only for praise')
parser.add_argument('--onlySourcecred', action='store_true', help='Generate dataset only for sourcecred')
args = parser.parse_args()


user_num = args.user_num if args.user_num is not None else 50
praise_num = args.praise_num if args.praise_num is not None else 500
quant_num = args.quant_num if args.quant_num  is not None else 10
quant_per_praise = args.quant_per_praise if args.quant_per_praise is not None else 3
token_num = args.token_num if args.token_num is not None else 1000


if args.onlyPraise:
    dataset = generate_praise_dataset(number_of_users=user_num, total_number_of_praises= praise_num, number_of_quants = quant_num, quants_per_praise = quant_per_praise).copy()
    save_dataset("praise",dataset)
elif args.onlySourcecred:
    dataset = generate_sourcecred_dataset(number_of_users = user_num, number_of_tokens= token_num).copy()
    save_dataset("sourcecred", dataset)
else:
    dataset_praise = generate_praise_dataset(number_of_users=user_num, total_number_of_praises= praise_num, number_of_quants = quant_num, quants_per_praise = quant_per_praise).copy()
    save_dataset("praise", dataset_praise)
    dataset_sourcecred = generate_sourcecred_dataset(number_of_users = user_num, number_of_tokens= token_num).copy()
    save_dataset("sourcecred", dataset_sourcecred)



