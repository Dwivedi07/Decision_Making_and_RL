import pandas as pd

def add_episode_index(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Initialize the episode index
    episode_index = 0
    episode_indices = []
    
    # Iterate through rows and assign episode index
    for i in range(len(df)):
        # Append the current episode index to the list
        episode_indices.append(episode_index)
        
        # Check if we're at the last row or if the next row doesn't continue the episode
        if i < len(df) - 1 and df.loc[i, 'sp'] != df.loc[i + 1, 's']:
            episode_index += 1
    
    df['episode_index'] = episode_indices
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")


input_csv = "data/medium.csv"   
output_csv = "data/updated_med_data.csv"  
add_episode_index(input_csv, output_csv)
