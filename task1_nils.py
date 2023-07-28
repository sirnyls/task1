import warnings
import smatch
import pandas as pd

def calc_with_warnings(true_amr, pred_amr):
    try:
        # redirect warnings 
        with warnings.catch_warnings(record=True) as w:
            best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(true_amr, pred_amr)
        
            if w:
                error_message = str(w[-1].message)
            else:
                error_message = None
                
            return best_match_num, test_triple_num, gold_triple_num, error_message

    except Exception as e:
        # if exception is raised, capture error msg
        error_message = str(e)
        return None, None, None, error_message

# testing
# Note: I ran the script and often got the error message "Error in parsing AMR ... " and 
# " 'NoneType' object has no attribute 'rename_node' "
# not sure if there might be something wrong in my script or if I understood the task correctly
# At https://github.com/snowblink14/smatch/blob/master/smatch.py#L644 I often see this exception...

df = pd.read_csv("test_data.csv")
data = df.to_dict(orient='records')
 
for amr_pair in data: 

    true_amr = amr_pair['true_amr']
    pred_amr = amr_pair['pred_amr']
    id = amr_pair['id']
# smatch.generate_amr_lines(true_amr, pred_amr)

    best_match, test_triple_num, gold_triple_num, error_message = calc_with_warnings(true_amr, pred_amr)

    if error_message is not None:
        print("id:", id)
        print("Error:", error_message)


