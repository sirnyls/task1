import warnings
import smatch_modified
import pandas as pd

def calc_with_warnings(true_amr, pred_amr, id):
    try:
        # redirect warnings 
        with warnings.catch_warnings(record=True) as w:
            best_match_num, test_triple_num, gold_triple_num, warning_message = smatch_modified.get_amr_match(true_amr, pred_amr, id)
            if w:
                warning_message.append(str(w[-1].message))
                
            return best_match_num, test_triple_num, gold_triple_num, warning_message

    except Exception as e:
        # if exception is raised, capture error msg
        
        error_message = str(e)

        return None, None, None, error_message


df_gpt4 = pd.read_csv("bio_amr_gpt4_invalid.csv")
data_gpt4 = df_gpt4.to_dict(orient='records')

#dfgpt35 = pd.read_csv("bio_amr_gpt3.5_invalid.csv")
#data_gpt35 = df_gpt4.to_dict(orient='records')
#true_amr_gpt35 = data_gpt4[1]['gold_amr']
#pred_amr_gpt35 = data_gpt4[1]['gpt4_0613_amr']

for amr_pair in data_gpt4: 
    true_amr_gpt4 = amr_pair['gold_amr']
    pred_amr_gpt4 = amr_pair['gpt4_0613_amr']
    id_gpt4 = amr_pair['id']


warning_message = []
best_match_num, test_triple_num, gold_triple_num, warning_message = calc_with_warnings(true_amr_gpt4, pred_amr_gpt4, id)


print("FINISH:", warning_message)






