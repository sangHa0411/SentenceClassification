import os
import copy
import pandas as pd
import collections
import argparse
from tqdm import tqdm


def main(args) :
    files = os.listdir(args.dataset_dir)
    results = []
    for f in files :
        if f.endswith('.csv') :
            file_path = os.path.join(args.dataset_dir, f)
            df = pd.read_csv(file_path)
            results.append(df)

    size = len(results[0])

    voted_labels = []
    counter = collections.Counter()
    for i in tqdm(range(size)) :
        labels = [df.iloc[i]['label'] for df in results]
        counter.update(labels)
        counter_dict = dict(counter)

        items = sorted(counter_dict.items(), key=lambda x : x[1], reverse=True)
        selected = items[0][0]
        voted_labels.append(selected)
        counter.clear()

    voted_df = copy.deepcopy(results[0])
    voted_df['label'] = voted_labels

    voted_df.to_csv(args.output_path, index=False)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./data', help='csv results directory')
    parser.add_argument('--output_path', type=str, default='./voted.csv', help='voted csv file path')
    
    args = parser.parse_args()
    main(args)