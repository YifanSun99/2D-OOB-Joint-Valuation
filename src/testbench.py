'''
testbench script for running experiments
'''
import os, json, subprocess, time

def _exp_setup(exp):
    exp_path = f"{exp['out_path']}/expno{exp['expno']}"
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
        
    os.chdir(exp_path)
    exp['exp_path'] = exp_path
    json.dump(exp, open('experiment.json','w'))    

def _job_cmd_maker(exp):
    r_cmds, r_paths = [], []
    for n in range(exp['n_runs']): 
        runpath = _run_setup(exp,n)
        cmd = f"python {exp['script_path']} run -e {exp['expno']}"
        cmd = f"{cmd} --run-id {n}"
        cmd = f"{cmd} --runpath {runpath}"
        cmd = f"{cmd} 2>&1 | tee {runpath}/run.log"
        r_cmds.append(cmd)
        r_paths.append(runpath)
    return r_cmds, r_paths

def _run_setup(exp, run):
    runpath = f"{exp['exp_path']}/run{run}"
    if not os.path.exists(runpath):
        os.mkdir(runpath)
    return runpath    

def _setup_rscript(run, expno, r_cmd, path):
    run_sh = f"r{run}-e{expno}_.sh"
    os.chdir(path)
    with open(run_sh,'w') as F:
        F.write("#!/bin/sh\n")
        F.write(r_cmd)
    return run_sh

def slurm_cmd_maker(exp):
    cmd = "sbatch"
    for slurm_p in exp['slurm']:
        cmd = f"{cmd} --{slurm_p}={exp['slurm'][slurm_p] }"
    return cmd

def launch_experiment(exp):
    _exp_setup(exp)
    r_cmds, r_paths = _job_cmd_maker(exp)
    exp['run_cmds'] = r_cmds
    for run in range(exp['n_runs']):
        slurm_cmd = slurm_cmd_maker(exp)
        rsh = _setup_rscript(run, exp['expno'], r_cmds[run], r_paths[run])
        subprocess.Popen(f"{slurm_cmd} {rsh}", shell=True)
        time.sleep(0.1) # YOU MAY WANT TO CHANGE HERE
        os.chdir(f"{exp['exp_path']}")
        
