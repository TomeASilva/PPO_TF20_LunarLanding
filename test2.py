from hyperparameter_search import see_configuration_details
import pickle

see_configuration_details()
hyperparameter = "gradient_steps_per_episode"
with open ("./parameter_search/current_hyperparameter.pickle", "wb") as f:
    pickle.dump(hyperparameter, f)

index_current_hyperparameter = 
with open("./parameter_search/current_hyperparameter_index.pickle", "wb") as f:
    pickle.dump(index_current_hyperparameter, f)
    
best_reward = -125
with open("./parameter_search/current_best_reward.pickle", "wb") as f:
    pickle.dump(best_reward, f)


