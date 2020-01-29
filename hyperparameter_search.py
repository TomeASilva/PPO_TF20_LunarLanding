from PPO_tf_2_0_Lunar_Landing import GlobalAgent, WorkerAgent
from hyperparameter_control import refresh_search
from multiprocessing import Manager, Process, Queue
import tensorflow as tf
import pickle
import csv

### this script performs grid search over the hyperparameters defined in hyperparameter_control.py


def run_search (agent_configuration, hyperparameters):
    number_of_workers = 2
    params_queue = Manager().Queue(number_of_workers)
    episode_queue = Manager().Queue()
    current_iter = Manager().Value("i", 0)
    average_reward_queue = Queue(1)
    

    
    global_agent = GlobalAgent(**agent_configuration, 
                               **hyperparameters,
                            episode_queue=episode_queue,
                            parameters_queue = params_queue,
                            current_iter=current_iter,
                            name="Global Agent", 
                            record_statistics=False,
                            average_reward_queue=average_reward_queue,
                            number_of_childs=number_of_workers,
                            save_checkpoints=False)
    
  
    workers = [WorkerAgent(**agent_configuration, **hyperparameters, episode_queue=episode_queue, parameters_queue = params_queue, current_iter=current_iter, name=f"Worker_{_}", record_statistics=False) for _ in range(number_of_workers)]
    
    processes = []
    p1 = Process(target=global_agent.training_loop)
    processes.append(p1)
    p1.start()

    for worker in workers:
        
        p = Process(target=worker.training_loop)
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
   
    print("Simulation Run")
    return average_reward_queue.get()

if __name__ == "__main__":
    refresh_search()
    with open("./parameter_search/current_hyperparameter.pickle", "rb") as file:
        current_hyperparameter = pickle.load(file)
    with open("./parameter_search/parameter_range.pickle", "rb") as file :
        hyperparameter_range = pickle.load(file)

    with open("./parameter_search/current_hyperparameter_index.pickle", "rb") as file:
        current_hyperparameter_index = pickle.load(file)
    with open("./parameter_search/max_reward.pickle", "rb") as file:
        max_reward_index = pickle.load(file)
    with open ("./parameter_search/hyperparameter_list.pickle", "rb") as file:
        hyperparameter_list = pickle.load(file)
        
    # the index of the current parameter to changed != current_hyperparameter_index which represents the index
    #with the range of possible values for the current hyperparameter
    parameter_index = 0
    
    
    best_reward = -100000
    agent_configuration = {
                    "action_space_bounds":[-1, 1],
                    "action_space_size":2,
                    "number_iter":100,
                    "max_steps":2000,
                    "n_episodes_per_cycle":1,
                    "env_name":"LunarLanderContinuous-v2",
                    "state_space_size":8,
                }
    while not (current_hyperparameter == "end"):
        if len(hyperparameter_range[current_hyperparameter]) == 1:
            parameter_index += 1
            current_hyperparameter = hyperparameter_list[parameter_index]
            current_hyperparameter_index = 0
        
    
                
        else:       
            hyperparameters = {}
            
            for key, hyperparameter in hyperparameter_range.items():
                if key != current_hyperparameter:
                    hyperparameters[key] = hyperparameter_range[key][max_reward_index[key]]
                else: 
                    hyperparameters[key] = hyperparameter_range[key][current_hyperparameter_index]
                
            reward = run_search(agent_configuration, hyperparameters)
            
            #prepare data to store into csv file
            hyperparameter_store_reward = {}
            hyperparameter_store_reward["Average_reward"] = reward
            
            for key, value in hyperparameters.items():
                if key == "actor_optimizer":
                    hyperparameter_store_reward[key] = str(value)
                    hyperparameter_store_reward["actor_learning_rate"] = str(value.learning_rate)
                elif key =="critic_optimizer":
                    hyperparameter_store_reward[key] = str(value)
                    hyperparameter_store_reward["critic_learning_rate"] = str(value.learning_rate)
                
                else:
                    hyperparameter_store_reward[key] = value
            

            #store data into csv file
            with open("./parameter_search/search_output.csv", "a") as file:
                writer = csv.writer(file, delimiter=":")
                for key, val in hyperparameter_store_reward.items():
                    writer.writerow([key, val])
            
            #check reward to decide what is the index for the current hyperameter that resulted in the highest results
            
            if reward > best_reward:
                
                
                max_reward_index[current_hyperparameter] = current_hyperparameter_index
                # with open("./parameter_search/max_reward.pickle", "wb") as file:
                #     pickle.dump(max_reward_index, file)
                best_reward = reward
            
            #Check if we are at the end of current hyperparameter range for the current hyperparameter
            
            if current_hyperparameter_index == len(hyperparameter_range[current_hyperparameter]) - 1:
                parameter_index += 1
                current_hyperparameter = hyperparameter_list[parameter_index]
                current_hyperparameter_index = 1 # by default the first element of the next hyperparameter was already tested in the previous hyperparameter resulting search
            else: 
                current_hyperparameter_index += 1
        
        # with open ("./parameter_search/search_output.json", "w", encoding='utf-8') as file: 
        #     json.dump(hyperparameter_store_reward, file)
            
        
    
    
    
                    
