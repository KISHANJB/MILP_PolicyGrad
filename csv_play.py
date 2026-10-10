import pandas as pd
from functools import reduce
def csv_print(csv_file1, csv_file2):
    

    # 1. Load the CSV file
    print("Easy Instances:")
    df1 = pd.read_csv(csv_file1, encoding='utf-8')
    print(df1)

    print("\nHard Instances:")
    df2 = pd.read_csv(csv_file2, encoding='utf-8')
    print(df2)

def column_dropper(csv_file, col_ids):
    """
    Drops multiple columns from a CSV based on a list of 0-based column index integers.
    Args:
        csv_file (str): Path to the CSV file.
        col_ids (list): List of column indices to drop, e.g., [1, 3].
    """
    # 1. Fix: Correct pandas read function
    df = pd.read_csv(csv_file)
    
    # 2. Extract column names safely using valid indices from the input list
    valid_indices = [idx for idx in col_ids if 0 <= idx < len(df.columns)]
    cols_to_drop = [df.columns[idx] for idx in valid_indices]
    
    # Report any out-of-range indices
    invalid_indices = set(col_ids) - set(valid_indices)
    for inv in invalid_indices:
        print(f"Index {inv} is out of range for this DataFrame.")
        
    if cols_to_drop:
        # 3. Drop ALL requested columns in a single, safe operation
        df = df.drop(columns=cols_to_drop)
        print(f"Successfully dropped columns: {cols_to_drop}")
    else:
        print("No valid columns were dropped.")
        
    return df

from functools import reduce
import pandas as pd

def df_merger_with_numeric_suffixes(dfs, output_filename="merged_output.csv"):
    """
    Merges n DataFrames on their 1st column and appends _1, _2, _3... 
    to any overlapping non-key columns.
    """
    if not dfs:
        print("ERROR: Provided DataFrame list is empty.")
        return None
    
    # 1. Step 1: Pre-process each DataFrame to append _1, _2, _3... to overlapping columns
    prepared_dfs = []
    
    for idx, df in enumerate(dfs, start=1):
        df_copy = df.copy()
        join_col = df_copy.columns[0]  # The 1st column used as the merge key
        
        # Rename all non-key columns by appending _1, _2, _3, etc.
        new_columns = {}
        for col in df_copy.columns:
            if col != join_col:
                new_columns[col] = f"{col}_{idx}"
                
        df_copy = df_copy.rename(columns=new_columns)
        prepared_dfs.append(df_copy)

    # 2. Step 2: Merge helper logic for two DataFrames
    def merge_two_dfs(left_df, right_df):
        col_left = left_df.columns[0]
        col_right = right_df.columns[0]
        
        merged = pd.merge(
            left_df, 
            right_df, 
            left_on=col_left, 
            right_on=col_right, 
            how='outer'
        )
        
        # Clean up duplicate primary key column if names differed
        if col_left != col_right and col_right in merged.columns:
            merged = merged.drop(columns=[col_right])
            
        return merged

    # 3. Step 3: Continuously merge all DataFrames across the chain
    merged_df = reduce(merge_two_dfs, prepared_dfs)

    # Save output
    merged_df.to_csv(output_filename, index=False)
    print(f"SUCCESS: Merged {len(dfs)} DataFrames with numeric suffixes!")
    
    return merged_df


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.options.display.float_format = '{:.2f}'.format

    #csv_print(csv_file1="solvRes_easy.csv", csv_file2="solvRes_hard.csv")
    #from main import results,dict_to_multiindex_df
    #df = dict_to_multiindex_df(results, "KISHAN", 6)
    #df = pd.read_csv("H200ep_a0.0b1.0g1.0s0.01C.csv", encoding='utf-8')
    '''df1 = pd.read_csv("H100ep_a1.0b0.0g0.0s0.01C.csv", encoding='utf-8')
    #print(df1)
    df2 = pd.read_csv("H100ep_a0.15b0.15g0.7s0.01.csv", encoding='utf-8')

    df3 = pd.read_csv("H100ep_a1.0b1.0g1.0s0.01C.csv", encoding='utf-8')

    df4 = pd.read_csv("H200ep_a0.0b1.0g1.0s0.01C.csv", encoding='utf-8')
    #print(df2)
    #df3 = pd.read_csv("/content/H100ep_a0.15b0.15g0.7s0.01.csv", encoding='utf-8')
    df4 = pd.read_csv("merged_output.csv", encoding='utf-8')
    df5 = pd.read_csv("merged_output.csv", encoding='utf-8')
    df6 = pd.read_csv("merged_output.csv", encoding='utf-8')
    drop_colums =[6, 7, 8]
    df1 = column_dropper("H100ep_a1.0b0.0g0.0s0.01C.csv", drop_colums)
    df2 = column_dropper("H100ep_a0.15b0.15g0.7s0.01.csv", [1,2,6,7,8])
    df3 = column_dropper("H100ep_a1.0b1.0g1.0s0.01C.csv", [1,2,6,7,8])

    df4 = column_dropper("H200ep_a0.0b1.0g1.0s0.01C.csv", [1,2,6,7,8])
    #print(df2)
    #df3 = column_dropper("/content/H100ep_a0.15b0.15g0.7s0.01.csv", [1,2,6,7,8])
    #print(df1)
    merged = df_merger_with_numeric_suffixes([df1, df4], "merged_output.csv")
    print(merged)'''
    #df = pd.read_csv("sgd.csv", encoding = 'utf-8')
    df = pd.read_csv("2club200v15p5scn_tr1_TreeEp.csv", encoding = 'utf-8')
    df2 = column_dropper("2club200v15p5scn_tr1_TreeEp.csv", [8,9,15])
    #df = pd.read_csv("smoke_test.csv", encoding = 'utf-8')


    print(df2)

    
