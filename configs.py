import numpy as np
import copy, socket, getpass

def generate_config(expno_name,
                     dataset='covertype',
                     problem='clf',
                     n_test=3000,
                     model_family='Tree',
                     n_runs=10,
                     experiment=None):    
    '''
    This function creates experiment configurations.
    '''
    
    # Experiment configuration
    exp = dict()
    exp['expno']=expno_name
    exp['n_runs']=n_runs
    exp['clf_path'] = r'C:\Users\yf-su\Desktop\XAI\df_oob\openml_dataset'
    exp['openml_clf_path'] = r'C:\Users\yf-su\Desktop\XAI\df_oob\openml_dataset'
    exp['openml_reg_path'] = r'C:\Users\yf-su\Desktop\XAI\df_oob\openml_dataset'

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
                    for n_trees in [1000]:
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
            # for run_id in range(n_runs):
            #     run = copy.deepcopy(run_temp) 
            #     run['run_id'] = run_id
            #     dargs_list=[]
            #     for n_train in [1000, 10000]:
            #         for n_trees in [1000]:
            #             dargs_list.append({'experiment':experiment,
            #                                'n_train':n_train, 
            #                                 'n_val':(n_train//10), 
            #                                 'n_test':n_test,
            #                                 'n_trees':n_trees,
            #                                 'clf_path':'clf_path',
            #                                 'openml_clf_path':'openml_clf_path',
            #                                 'openml_reg_path':'openml_reg_path',
            #                                 'is_noisy':0,
            #                                 'model_family':model_family,
            #                                 'run_id':run_id})
            #     run['dargs_list'] = dargs_list
            #     runs.append(run)
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
            elif experiment in ['mask&rank']:
                mask_ratio_list = [0.5, 0.8]
                base = 3
                is_noisy, error_row_rate = None, None
                error_mech_list, error_col_rate_list = [None], [None]
            # elif experiment in ['feature_removal']:
            #     mask_ratio_list = [0]
            #     base = 0
            #     is_noisy, error_row_rate = None, None
            #     error_mech_list, error_col_rate_list = [None], [None]
            elif experiment == 'error':
                error_row_rate = 0.1
                error_col_rate_list = [0.1,0.3]
                error_mech_list = ['noise','adv']
                mask_ratio_list, is_noisy, base = [None], None, None
            else:
                raise NotImplementedError('Check Experiment')            
                
            for n_train in [1000, 5000]:
                for input_dim in [20, 100]:
                    for n_trees in [1000]:
                        for rho in [0, 0.2, 0.6]:
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

def config000CR(experiment):
    exp, runs=generate_config(expno_name='000CR', problem='clf', dataset='gaussian', n_runs=10,
                              experiment=experiment)
    return exp, runs  

def config001CR(experiment):
    exp, runs=generate_config(expno_name='001CR', problem='clf', dataset='pol',experiment=experiment)
    return exp, runs  

def config002CR(experiment):
    exp, runs=generate_config(expno_name='002CR', problem='clf', dataset='jannis',experiment=experiment)
    return exp, runs  

def config003CR(experiment):
    exp, runs=generate_config(expno_name='003CR', problem='clf', dataset='lawschool',experiment=experiment)
    return exp, runs  

def config004CR(experiment):
    exp, runs=generate_config(expno_name='004CR', problem='clf', dataset='fried',experiment=experiment)
    return exp, runs  

def config005CR(experiment):
    exp, runs=generate_config(expno_name='005CR', problem='clf', dataset='vehicle_sensIT',experiment=experiment)
    return exp, runs  

def config006CR(experiment):
    exp, runs=generate_config(expno_name='006CR', problem='clf', dataset='electricity',experiment=experiment)
    return exp, runs  

def config007CR(experiment):
    exp, runs=generate_config(expno_name='007CR', problem='clf', dataset='2dplanes',experiment=experiment)
    return exp, runs  

def config008CR(experiment):
    exp, runs=generate_config(expno_name='008CR', problem='clf', dataset='creditcard',experiment=experiment)
    return exp, runs  

def config009CR(experiment):
    exp, runs=generate_config(expno_name='009CR', problem='clf', dataset='covertype',experiment=experiment)
    return exp, runs    

def config010CR(experiment):
    exp, runs=generate_config(expno_name='010CR', problem='clf', dataset='nomao',experiment=experiment)
    return exp, runs 

def config011CR(experiment):
    exp, runs=generate_config(expno_name='011CR', problem='clf', dataset='webdata_wXa',experiment=experiment)
    return exp, runs 

def config012CR(experiment):
    exp, runs=generate_config(expno_name='012CR', problem='clf', dataset='MiniBooNE',experiment=experiment)
    return exp, runs 


