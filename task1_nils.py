import warnings
import smatch_modified
import pandas as pd




df = pd.read_csv("bio_amr_gpt4_invalid.csv")
data = df.to_dict(orient='records')


true_amr = data[1]['gold_amr']
pred_amr = data[1]['gpt4_0613_amr']



best_match_num, test_triple_num, gold_triple_num, error_message = smatch_modified.get_amr_match(true_amr, pred_amr)


print("FINISH:", error_message)




'''  
for amr_pair in data: 

    true_amr = amr_pair['gold_amr']
    pred_amr = amr_pair['gpt4_0613_amr']
    id = amr_pair['id']
# smatch.generate_amr_lines(true_amr, pred_amr)

    best_match, test_triple_num, gold_triple_num, error_message = calc_with_warnings(true_amr, pred_amr)

    if error_message is not None:
        print("id:", id)
        print("Error:", error_message)
    else: 
        print("best match:", best_match)
        print("test_triple_num:", test_triple_num)
        print("gold_triple_num:", gold_triple_num)
'''

