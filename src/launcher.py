import argh, os, pickle
from testbench import launch_experiment
from simulator import main
from configs import *
import configs

parser = argh.ArghParser()

def launch(exp_id):
    exp, runs = eval(f'config{exp_id}()')
    print(f'Preparing to launch {len(runs)}...')
    assert exp['n_runs'] == len(runs), '# of runs does not match n_runs'
    launch_experiment(exp)

def run(exp_id='', run_id=0, runpath=''):
    _, runs=eval(f'config{exp_id}()')
    config=runs[run_id]
    if runpath != '':
        config['runpath']=runpath 
        os.chdir(runpath)
    with open('config.pickle', 'wb') as pkl_file:
        pickle.dump(config, pkl_file)
    main(config)

parser = argh.ArghParser()
parser.add_commands([launch,run])

if __name__ == '__main__':
    parser.dispatch()

