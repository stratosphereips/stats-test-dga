import pandas as pd
import numpy as np
import datetime
import os


UMUDGA_families = ['alureon', 'chinad', 'dyre', 'gozi_luther', 'kraken_v2', 'murofet_v1', 'nymaim', 'pushdo', 'qakbot', 'ranbyus_v2', 'sisron', 'symmi', 'vawtrak_v2', 'banjori', 'corebot', 'fobber_v1', 'gozi_nasa', 'murofet_v2', 'padcrypt', 'pykspa', 'ramdo', 'rovnix', 'suppobox_1', 'tempedreve', 'vawtrak_v3', 'bedep', 'cryptolocker', 'fobber_v2', 'gozi_rfc4343', 'locky', 'murofet_v3', 'pizd', 'pykspa_noise', 'ramnit', 'shiotob', 'suppobox_2', 'tinba', 'zeus-newgoz', 'ccleaner', 'dircrypt', 'gozi_gpl', 'kraken_v1', 'matsnu', 'necurs', 'proslikefan', 'qadars', 'ranbyus_v1', 'simda', 'suppobox_3', 'vawtrak_v1'] 

umu_path = '../data/UMUDGA_families'
tranco_dataset_path = '../data/tranco_6JVNX_20230701.csv'

tranco = pd.read_csv(tranco_dataset_path, names=['ranking', 'domain'])

def get_sample_umudga(samples=100, dga_share=0.5, family='all'):
    notdga_samples = int(samples*(1 - dga_share))
    dga_samples = int(samples*dga_share)
    samples = []
    if family == 'all':
        
        for fam in UMUDGA_families:
            try:
                sample = pd.read_csv(umu_path + f'/{fam}/list/100000.txt',names=['domain']).sample(int(dga_samples/len(UMUDGA_families)))
            except FileNotFoundError:
                sample = pd.read_csv(umu_path + f'/{fam}/list/5000.txt',names=['domain']).sample(int(dga_samples/len(UMUDGA_families)))
            sample['label'] = 'dga'
            samples.append(sample)
    else:
        try:
            sample = pd.read_csv(umu_path + f'/{family}/list/100000.txt',names=['domain']).sample(int(dga_samples))
        except FileNotFoundError:
            sample = pd.read_csv(umu_path + f'/{family}/list/5000.txt',names=['domain']).sample(int(dga_samples/len(UMUDGA_families)))
        sample['label'] = 'dga'
        samples.append(sample)
    sample = tranco.iloc[:500000].sample(notdga_samples)
    sample['label'] = 'notdga'
    samples.append(sample.drop(columns=['ranking']))
    return pd.concat(samples).reset_index().drop(columns=['index'])

def inference(sample):
    # pred = ['dga' if (x>0.5) else 'notdga' for x in np.random.random(len(sample))]
    pred = []
    # iterate over the domains and do a prediction for each
    for domain in sample.domain:
        if np.random.random() > .5:
            pred.append('dga')
        else:
            pred.append('notdga')
    # return the same dataset with a extra column 'pred'
    # if domains are queried in bulk, then be sure the answers preserve the ordering of queried domains!
    sample['pred'] = pred
    return sample

def run_once(family):
    sample = get_sample_umudga(family=family)
    sample = inference(sample)
    return sample

def run_experiment():
    # create the results dir
    family = 'all'
    destdir = f'results/experimet_{family}_{str(datetime.datetime.now()).split(".")[0].replace(" ","_").replace(":","-")}'
    os.makedirs(destdir)
    # run the experiment 30 times for statistical purposes
    for run in range(30):
        result = run_once(family)
        result.to_csv(destdir + f'/results_{run}.csv', index=False)


if __name__ == '__main__':
    run_experiment()
