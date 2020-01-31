from PPO_tf_2_0_Lunar_Landing import GlobalAgent, WorkerAgent
from hyperparameter_control import refresh_search
from multiprocessing import Manager, Process, Queue
import tensorflow as tf
import pickle
import csv

### this script performs grid search over the hyperparameters defined in hyperparameter_control.py
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
# print(tf.__version__)

def see_configuration_details():
    with open("./parameter_search/current_hyperparameter.pickle", "rb") as file:
        current_hyperparameter = pickle.load(file)
        print(f"Current hyperparameter {current_hyperparameter} \n \n")
    
    with open("./parameter_search/parameter_range.pickle", "rb") as file :
        hyperparameter_range = pickle.load(file)
        
    with open("./parameter_search/current_hyperparameter_index.pickle", "rb") as file:
        current_hyperparameter_index = pickle.load(file)
        if current_hyperparameter == "actor_optimizer" or hyperparameter == "critic_optimizer":
            print(f"Value for the current hyperparameter: {hyperparameter_range[current_hyperparameter][current_hyperparameter_index]} \n \n")
            print(f"Learning_rate: {hyperparameter_range[current_hyperparameter][current_hyperparameter_index].learning_rate} \n \n")
        else:
            print(f"Value for the current hyperparameter: {hyperparameter_range[current_hyperparameter][current_hyperparameter_index]} \n \n")
        

   
    with open("./parameter_search/max_reward.pickle", "rb") as file:
        max_reward_index = pickle.load(file)
        
        for hyperparameter, value in max_reward_index.items():
            # for actor optimizer show the algorithm and the learning rate
            if hyperparameter == "actor_optimizer" or hyperparameter == "critic_optimizer":
                print(f"{hyperparameter}: {hyperparameter_range[hyperparameter][value]}")
                print(f"{hyperparameter} learning rate : {hyperparameter_range[hyperparameter][value].learning_rate}")   
            else:
                print(f"{hyperparameter}: {hyperparameter_range[hyperparameter][value]}")


    with open("./parameter_search/current_best_reward.pickle", "rb") as file:
        best_reward = pickle.load(file)
        print(f"Best reward : {best_reward}")
     
    

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
    p1 = Process(target=global_agent.training_loop, name="Global")
    processes.append(p1)
    p1.start()

    for worker in workers:
        
        p = Process(target=worker.training_loop, name=worker.name)
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
   
    print("Simulation Run")
    return average_reward_queue.get()

if __name__ == "__main__":
    user_option = input(f"Do you want to continue the previous search (yes), or start over (no)")

    if user_option == "yes":
        pass
    elif user_option == "no":
        refresh_search()
    else:
        print("Option not accepted the program will terminate")
        exit()
        
 
    with open("./parameter_search/current_hyperparameter.pickle", "rb") as file:
        current_hyperparameter = pickle.load(file)
    with open("./parameter_search/parameter_range.pickle", "rb") as file:
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
                    "number_iter":100,# <--------------------------
                    "max_steps":2000,
                    "n_episodes_per_cycle":1,
                    "env_name":"LunarLanderContinuous-v2",
                    "state_space_size":8,
                }
    # If current hyperparameter is not "end" continue the parameter search
    while not (current_hyperparameter == "end"):
        # If for the current hyperparameter there is only one configuration to search, then there is not need to search within the current hyperparameter
        if len(hyperparameter_range[current_hyperparameter]) == 1:
            parameter_index += 1 # Go to next hyperparameter within the hyperparameter_list
            current_hyperparameter = hyperparameter_list[parameter_index]
            current_hyperparameter_index = 0 # Start the search of the next hyperparameter from the start 
        
    
        # If there is more than one configuration possible for the current hyperparameter
        # Let's build the hyperparameter search     
        else:       
            hyperparameters = {}
            
            for key, hyperparameter in hyperparameter_range.items():
                #For the hyperparameters already searched we want to set them for the configuraion that already led to the highest reward
                if key != current_hyperparameter:
                    hyperparameters[key] = hyperparameter_range[key][max_reward_index[key]]
                #For the hyperparameter being searched we want to search from where we left out    
                else: 
                    hyperparameters[key] = hyperparameter_range[key][current_hyperparameter_index]
            
            # Run the configuration of hyperparameters
            reward = run_search(agent_configuration, hyperparameters)
            
            #prepare data to store into csv file
            #This dictionary will contain the summary of the search with established configuration
            hyperparameter_store_reward = {}
            #Store the reward
            hyperparameter_store_reward["Average_reward"] = reward
            
            for key, value in hyperparameters.items():
                if key == "actor_optimizer":
                    #segregate the gradient descent algorithm from the learning rate for the actor
                    hyperparameter_store_reward[key] = str(value)
                    hyperparameter_store_reward["actor_learning_rate"] = str(value.learning_rate)
                elif key =="critic_optimizer":
                    #segregate the gradient descent algorithm from the learning rate for the actor
                    hyperparameter_store_reward[key] = str(value)
                    hyperparameter_store_reward["critic_learning_rate"] = str(value.learning_rate)
                
                else:
                    #for the rest of the hyperparameters the dictionary to store is equal to hyperparameters used in the last run
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
                with open("./parameter_search/current_best_reward.pickle", "wb") as f :
                    pickle.dump(best_reward, f)
       
            
            #Check if we are at the end of current hyperparameter range for the current hyperparameter
            
            if current_hyperparameter_index == len(hyperparameter_range[current_hyperparameter]) - 1:
                # We will advance the search to the next hyperparemeter
                parameter_index += 1 
                current_hyperparameter = hyperparameter_list[parameter_index]
                current_hyperparameter_index = 1 # by default the first element of the next hyperparameter was already tested in the previous hyperparameter resulting search
            else: 
                #Continue the search within the current hyperparameter, by advancing on next value possible for the current hyperparameter
                current_hyperparameter_index += 1
            
        # Store the next hyperparameter to be searched
        with open("./parameter_search/current_hyperparameter.pickle", "wb") as file:
            pickle.dump(current_hyperparameter, file)
       # Where whitin the the hyperparameter did we left out
        with open("./parameter_search/current_hyperparameter_index.pickle", "wb") as file:
            pickle.dump(current_hyperparameter_index, file)
        # For all the hyperparameters already searched, store the best configuration found
        with open("./parameter_search/max_reward.pickle", "wb") as file:
            pickle.dump(max_reward_index, file)
     
       
       
        # with open ("./parameter_search/search_output.json", "w", encoding='utf-8') as file: 
        #     json.dump(hyperparameter_store_reward, file)
            
        
    
    
    
                    
