import numpy as np
import copy, socket, getpass
import os

def generate_config(expno_name,
                     dataset='covertype',
                     problem='clf',
                     n_test=3000,
                     model_family='Tree',
                     n_runs=30,
                     experiment=None):    
    '''
    This function creates experiment configurations.
    '''
    
    # Experiment configuration
    exp = dict()
    exp['expno']=expno_name
    exp['n_runs']=n_runs
    exp['script_path'] = '/burg/stats/users/ys3600/test_ginsburg/src/launcher.py'
    exp['clf_path'] = '/burg/stats/users/ys3600/test_ginsburg'
    exp['openml_clf_path'] = '/burg/stats/users/ys3600/test_ginsburg/openml_dataset'
    exp['openml_reg_path'] = '/burg/stats/users/ys3600/test_ginsburg/openml_dataset'
    # exp['out_path'] = '/burg/stats/users/ys3600/test_ginsburg/spreadout/%s'%(experiment) 
    exp['out_path'] = '/burg/stats/users/ys3600/test_ginsburg/%s_0.25_0.75'%(experiment) 
    # if not os.path.exists(exp['out_path']):
    #     os.makedirs(exp['out_path'])
    exp['slurm'] = {'account':'stats','time':'24:00:00','ntasks-per-node':'4','cpus-per-task':'1','mem-per-cpu':'10G','nodes':1,'exclude':'g[189-194]'}

    # Run configuration
    run_temp = dict()
    run_temp['problem']=problem
    run_temp['dataset']=dataset
    
    runs=[]
    if expno_name != '000CR':
        if problem == 'clf':
            for run_id in range(n_runs):
                run = copy.deepcopy(run_temp) 
                run['run_id'] = run_id
                dargs_list=[]
                is_noisy = 0.1 if experiment == 'noisy' else None
                for n_train in [1000]:
                    for n_trees in [3000]:
                        dargs_list.append({'experiment':experiment,
                                           'n_train':n_train, 
                                            'n_val':(n_train//10), 
                                            'n_test':n_test,
                                            'n_trees':n_trees,
                                            'clf_path':exp['clf_path'],
                                            'openml_clf_path':exp['openml_clf_path'],
                                            'openml_reg_path':exp['openml_reg_path'],
                                            'is_noisy':is_noisy,
                                            'model_family':model_family,
                                            'run_id':run_id})
                run['dargs_list'] = dargs_list
                runs.append(run)
        elif problem == 'reg':
            raise NotImplementedError('Reg problem not implemented yet!')
        else:
            assert False, f'Check Problem: {problem}'

    else:
        for run_id in range(n_runs):
            run = copy.deepcopy(run_temp) 
            run['run_id'] = run_id
            dargs_list=[]
            
            if experiment == 'noisy':
                is_noisy = 0.1
                error_row_rate, base = None, None
                mask_ratio_list, error_mech_list, error_col_rate_list = [None], [None], [None]
 
            elif experiment == 'error':
                raise NotImplementedError('Check!!!')
                error_row_rate = 0.1
                error_col_rate_list = [0.1]
                error_mech_list = ['adv']
                mask_ratio_list, is_noisy, base = [None], None, None
            
            elif experiment == 'normal' or experiment == 'outlier':
                error_col_rate_list, error_mech_list, mask_ratio_list, is_noisy, base, \
                    error_row_rate = [None], [None], [None], None, None, None
            else:
                raise NotImplementedError('Check Experiment')            
                
            for n_train in [1000]:
                for input_dim in [20, 100]:
                    for rho in [0, 0.2, 0.6]:
                        for n_trees in [3000]:
                            for mask_ratio in mask_ratio_list:
                                for error_mech in error_mech_list:
                                    for error_col_rate in error_col_rate_list:
                                        dargs_list.append({'experiment':experiment,
                                                             'n_train':n_train, 
                                                            'n_val':(n_train//10), 
                                                            'n_test':n_test,
                                                            'input_dim':input_dim,
                                                            'n_trees':n_trees,
                                                            'rho':rho,
                                                            'is_noisy':is_noisy,
                                                            'mask_ratio':mask_ratio,
                                                            'base':base,
                                                            'error_row_rate':error_row_rate,
                                                            'error_col_rate':error_col_rate,
                                                            'error_mech':error_mech,
                                                            'model_family':model_family,
                                                            'run_id':run_id,                                                    
                                                            # 'clf_path':exp['clf_path'],# Note here
                                                            # 'openml_path':exp['openml_path'],                                                    
                                                            })
            run['dargs_list'] = dargs_list
            runs.append(run)

    return exp, runs 

'''
Classification
'''

# def config000CR(experiment = 'error'):
#     exp, runs=generate_config(expno_name='000CR', problem='clf', dataset='gaussian', n_runs=30,
#                               experiment=experiment)
#     return exp, runs  

def config001CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='001CR', problem='clf', dataset='pol',experiment=experiment)
    return exp, runs  

def config002CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='002CR', problem='clf', dataset='jannis',experiment=experiment)
    return exp, runs  

def config003CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='003CR', problem='clf', dataset='lawschool',experiment=experiment)
    return exp, runs  

def config004CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='004CR', problem='clf', dataset='fried',experiment=experiment)
    return exp, runs  

def config005CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='005CR', problem='clf', dataset='vehicle_sensIT',experiment=experiment)
    return exp, runs  

def config006CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='006CR', problem='clf', dataset='electricity',experiment=experiment)
    return exp, runs  

def config007CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='007CR', problem='clf', dataset='2dplanes',experiment=experiment)
    return exp, runs  

def config008CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='008CR', problem='clf', dataset='creditcard',experiment=experiment)
    return exp, runs  

def config009CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='009CR', problem='clf', dataset='gas-drift',experiment=experiment)
    return exp, runs 

def config010CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='010CR', problem='clf', dataset='nomao',experiment=experiment)
    return exp, runs 

def config011CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='011CR', problem='clf', dataset='musk',experiment=experiment)
    return exp, runs 

def config012CR(experiment = 'normal'):
    exp, runs=generate_config(expno_name='012CR', problem='clf', dataset='MiniBooNE',experiment=experiment)
    return exp, runs 


def config101CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='101CR', problem='clf', dataset='pol',experiment=experiment)
    return exp, runs  

def config102CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='102CR', problem='clf', dataset='jannis',experiment=experiment)
    return exp, runs  

def config103CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='103CR', problem='clf', dataset='lawschool',experiment=experiment)
    return exp, runs  

def config104CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='104CR', problem='clf', dataset='fried',experiment=experiment)
    return exp, runs  

def config105CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='105CR', problem='clf', dataset='vehicle_sensIT',experiment=experiment)
    return exp, runs  

def config106CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='106CR', problem='clf', dataset='electricity',experiment=experiment)
    return exp, runs  

def config107CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='107CR', problem='clf', dataset='2dplanes',experiment=experiment)
    return exp, runs  

def config108CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='108CR', problem='clf', dataset='creditcard',experiment=experiment)
    return exp, runs  

def config109CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='109CR', problem='clf', dataset='gas-drift',experiment=experiment)
    return exp, runs 

def config110CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='110CR', problem='clf', dataset='nomao',experiment=experiment)
    return exp, runs 

def config111CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='111CR', problem='clf', dataset='musk',experiment=experiment)
    return exp, runs 

def config112CR(experiment = 'outlier'):
    exp, runs=generate_config(expno_name='112CR', problem='clf', dataset='MiniBooNE',experiment=experiment)
    return exp, runs 


def config201CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='201CR', problem='clf', dataset='pol',experiment=experiment)
    return exp, runs  

def config202CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='202CR', problem='clf', dataset='jannis',experiment=experiment)
    return exp, runs  

def config203CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='203CR', problem='clf', dataset='lawschool',experiment=experiment)
    return exp, runs  

def config204CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='204CR', problem='clf', dataset='fried',experiment=experiment)
    return exp, runs  

def config205CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='205CR', problem='clf', dataset='vehicle_sensIT',experiment=experiment)
    return exp, runs  

def config206CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='206CR', problem='clf', dataset='electricity',experiment=experiment)
    return exp, runs  

def config207CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='207CR', problem='clf', dataset='2dplanes',experiment=experiment)
    return exp, runs  

def config208CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='208CR', problem='clf', dataset='creditcard',experiment=experiment)
    return exp, runs  

def config209CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='209CR', problem='clf', dataset='gas-drift',experiment=experiment)
    return exp, runs 

def config210CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='210CR', problem='clf', dataset='nomao',experiment=experiment)
    return exp, runs 

def config211CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='211CR', problem='clf', dataset='musk',experiment=experiment)
    return exp, runs 

def config212CR(experiment = 'noisy'):
    exp, runs=generate_config(expno_name='212CR', problem='clf', dataset='MiniBooNE',experiment=experiment)
    return exp, runs 