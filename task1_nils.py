import warnings
import smatch_modified
import pandas as pd




df = pd.read_csv("bio_amr_gpt4_invalid.csv")
data = df.to_dict(orient='records')

true_amr = data[1]['gold_amr']
pred_amr = data[1]['gpt4_0613_amr']

warning_message = []

def calc_with_warnings(true_amr, pred_amr):
    try:
        # redirect warnings 
        with warnings.catch_warnings(record=True) as w:
            best_match_num, test_triple_num, gold_triple_num, warning_message = smatch_modified.get_amr_match(true_amr, pred_amr)
            if w:
                warning_message.append(str(w[-1].message))
                
            return best_match_num, test_triple_num, gold_triple_num, warning_message

    except Exception as e:
        # if exception is raised, capture error msg
        
        best_match_num, test_triple_num, gold_triple_num, warning_message = smatch_modified.get_amr_match(true_amr, pred_amr)
        warning_message.append(str(e))

        return None, None, None, warning_message

best_match_num, test_triple_num, gold_triple_num, warning_message = calc_with_warnings(true_amr, pred_amr)


print("FINISH:", warning_message)




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

