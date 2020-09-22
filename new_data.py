import os
import requests
import pickle
import datetime
import pandas as pd
from tabulate import tabulate
# automate git push process
import subprocess as cmd
import shutil

# Input
playercol = ['date', 'Mirataro', 'Shinwan', 'ToShiroh', 'yukoron']
new_score = pd.DataFrame(columns=playercol)
new_date = datetime.datetime.now().strftime('%Y%m%d')
print('Script ran on: ', new_date)
url = "https://tenhou.net/sc/raw/dat/"+f"sca{new_date}.log.gz"
filename = f"sca{new_date}.log.gz"
print('Filename: ', filename)

# Current_dir for the folder name
current_dir = os.getcwd()
print('Current Dir: ', current_dir)

# Download gz file from the url
with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

# Convert to dataframe and add date to the dataframe
dfn = pd.read_csv(filename, usecols=[0], error_bad_lines=False, header=None)
dfn[len(dfn.columns)] = new_date

# Filter by player name:
dfs = dfn[(dfn[0].str.contains(playercol[1])) & (dfn[0].str.contains(playercol[2])) & (dfn[0].str.contains(playercol[3])) & (dfn[0].str.contains(playercol[4]))]
print('The number of data extracted: ', len(dfs[0]))
# Check if there is any data to extract.
if len(dfs[0]) > 0:
    # Split and tidy up the dataframe
    dfs[['one','two','three','four']] = dfs[0].str.split('|', 3, expand=True)
    dfs.columns = ['original', 'date', 'room', 'time', 'type', 'name1']
    dfs['date'] = pd.to_datetime(dfs['date'], format='%Y%m%d')
    dfs[['empty', 'name1', 'name2', 'name3', 'name4']] = dfs.name1.str.split(" ", n=4, expand=True)
    # Only use the important columns
    dfs = dfs[['date', 'name1', 'name2', 'name3', 'name4']]

    # Extract score of each player and put it in a dataframe players
    k=0
    for i, j in dfs.iterrows():
        dd = j[0]
        new_score.loc[k, 'date'] = dd
        for name in dfs.columns[1:]:
            s = j[name]
            player = s.split('(')[0]
            score = [p.split(')')[0] for p in s.split('(') if ')' in p][0]
            score = int(float(score.replace('+', '')))
            new_score.loc[k, player] = score
        k += 1
    print('new score board:')
    print(tabulate(new_score.head()))
    print(new_score.shape)

    # Combine with the old data
    old_score = pd.read_pickle(f"{current_dir}/players_score.pkl")
    print('old score board:')
    print(tabulate(old_score.head()))
    print(tabulate(old_score.tail()))
    print(old_score.shape)

    # Added ignore_index = True to redo index (2020/09/12)
    concat_score = pd.concat([old_score, new_score], ignore_index=True)
    print(tabulate(concat_score.head()))
    print(tabulate(concat_score.tail()))
    print(concat_score.shape)

    # Pickle the new combined data.
    concat_score.to_pickle(f"{current_dir}/players_score.pkl")

    # push the new data to github page no success. test6
    cp = cmd.run(f"cd / && cd {current_dir} && git add players_score.pkl", check=True, shell=True)
    print("Successful git add command")
    cp = cmd.run(f"cd / && cd {current_dir} && git commit -m 'scoresheet updated on {new_date}'", check=True, shell=True)
    print("Successful git commit command")
    cp = cmd.run(f"cd / && cd {current_dir} && git push origin master", check=True, shell=True)
    print('Successful upload to Github!')

    # Move the file with our data to data folder
    shutil.move(f"{current_dir}/{filename}", f"{current_dir}/data/{filename}")

# if there is no data, print no data extracted.
else:
    print('no data extracted')
    #Delete the file with no data for our match (wrong file)
    os.remove(filename)


