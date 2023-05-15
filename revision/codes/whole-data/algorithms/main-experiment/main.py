import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter

pd.options.display.float_format = '{:.2f}'.format


#######################################################################algorithms#####################################################################


##################################################################initialization##########################################################

num_states=13
num_actions=5
epsilon=0.01/(num_actions-1)
#discount factor
gamma=0.3
#theta
theta=0.0001

#################################################Policy Copy#################################


####creating transition_probability and reward matrix
def transition_probability_reward(train_set):
    """
        transition_probability_reward: get transition probability and reward for all transitions from s to s' while applying action a
        
        @param train_set: data partitioned into training set
        @return transition probability: transition probability during the transitionfrom one state to others
        return reward mean: reward mean during the transition from one state to another
    """
    #trans_count is a dictionary of {(state, action): {next_state: count}}
    #reward_sum is a matrix that counts the sum of total reward going to s' from s taking action a

    trans_counts={}
    reward_sum = {}
    groups = train_set.groupby('RID')
    for patient, rows in groups:
        for i, row in rows.iterrows():
            cur_state=row['states']
            action=row['action']
            reward=row['reward']
            try:
                next_state =train_set.loc[i+1,'states']
                #add this transition to trans_counts
                if (cur_state, action) in trans_counts:
                    if next_state in trans_counts[(cur_state,action)]:
                        trans_counts[(cur_state,action)][next_state] += 1
                        reward_sum[(cur_state,action)][next_state] += reward
                    else:
                        trans_counts[(cur_state,action)][next_state] = 1
                        reward_sum[(cur_state,action)][next_state] = reward
                else:
                    trans_counts[(cur_state,action)] = {next_state: 1}
                    reward_sum[(cur_state,action)]={next_state: reward} 

            except KeyError:
                pass
            
            
    #normalize the transition counts
    trans_prob = {}   #transition probability from s to s' given action a
    reward_mean = {}
    count = 0
    for state_action, next_state_count in trans_counts.items():
        # print("\nstate_action",state_action)
        # print("\nnext_state_count",next_state_count)
        count += 1
        norm_const = sum(next_state_count.values())
        # print("norm_const", norm_const)
        trans_prob[state_action] = {}
        for next_state, count in next_state_count.items():
            trans_prob[state_action][next_state] = float(count)/float(norm_const)
    # print("\ntrans_prob",trans_prob)

    for state_action, next_state_reward in reward_sum.items():
        reward_mean[state_action] = {}
        for next_state, sums in next_state_reward.items():
            reward_mean[state_action][next_state] = float(sums)/float(trans_counts[state_action][next_state]) 

    return trans_prob, reward_mean

def policy_evaluation(policy,state_values, trans_mat, reward_mean):
    """
        policy_evaluation:evaluate policy by estimating values for each states
        return: state values of the given policy
    """
    delta=theta*2
    while (delta > theta):
        delta = 0
        for state in range(num_states):
            new_value_for_state=0
            for action in range(num_actions):
                try:
                    prob_dist=trans_mat[(state,action)]
                except:
                    continue
                for next_state,t_probability in prob_dist.items():
                    reward=reward_mean[(state,action)][next_state]
                    ################################from pseudocode#######################
                    try:
                        new_value_for_state+=policy[state,action]*t_probability*(reward+gamma*state_values[next_state]) 
                    except:
                        print("index error")
            delta = max(delta, np.abs(new_value_for_state - state_values[state])) 
            state_values[state] = new_value_for_state
    # print("state_values",state_values)
    return state_values

def policy_improvement( policy, state_values, trans_mat, reward_mean):
    """
        policy_evaluation:improve policy
        return: improved policy
    """
    
    policy_stable = True
    for state in range(num_states):
        # actions from current state
        old_action = np.argmax(policy[state])
        temp_array = np.zeros((num_actions))
        for action in range(num_actions):
            try:
                prob_dist=trans_mat[(state,action)]
            except:
                continue
            for next_state,t_probability in prob_dist.items():
                reward=reward_mean[(state,action)][next_state]
                try:
                    temp_array[action] += t_probability * (reward + gamma * state_values[next_state])
                    # print(temp_array)
                except:
                    print("index error")
        
        policy[state] = np.ones((num_actions))*epsilon
        policy[state, np.argmax(temp_array)] = 1-epsilon*(num_actions-1)
        if old_action != np.argmax(policy[state]): 
            policy_stable = False
    # print(policy_stable)
            
    return policy_stable, policy


def policy_iteration(trans_mat, reward_mean):

    #initialize value for each state
    state_values = np.zeros(num_states)
    #initialize policy
    policy = np.ones((num_states, num_actions))*epsilon

    policy_stable=False
    while not policy_stable:
        # policy evaluation
        state_values = policy_evaluation(policy, state_values, trans_mat, reward_mean)
        # policy improvement
        policy_stable,policy = policy_improvement(policy, state_values, trans_mat, reward_mean)
        # print(policy)
        
    # return state_values, policy,another_copy
    return state_values, policy

def policy_copy(trans_mat, reward_mean):

    #initialize value for each state
    state_values = np.zeros(num_states)
    #initialize policy
    policy = np.ones((num_states, num_actions))*epsilon

    # policy evaluation
    state_values = policy_evaluation(policy, state_values, trans_mat, reward_mean)
    # policy copy
    policy_stable,policy = policy_improvement(policy, state_values,trans_mat, reward_mean)
        
    # return state_values, policy,another_copy
    return state_values, policy


# print(behavior_policy)
######################################testing######################################
def weighted_importance_sampling(policy, test_set, behavior_policy):
    
    test_set['timestep']=''
    test_set['rho']=''
    
    groups = test_set.groupby('RID')
    for rid, rows in groups:
        timestep=0
        rho_cum=1
        #each_row
        for index, row in rows.iterrows():
            state=int(row['states'])
            action=int(row['action'])
            test_set.loc[index, 'timestep']=timestep
            rho_cum = rho_cum*policy[state,action]/behavior_policy[state, action]
            #test_set.iloc[i,7]=rho_cum
            test_set.loc[index, 'rho']=rho_cum

            timestep = timestep+1
            
            

    values=[]
    value_sum =0
    for rid, rows in groups:
        # value_sum =0
        sum_of_rho=rows['rho'].sum()
        for index, row in rows.iterrows():
            reward = row['reward']
            timestep= row['timestep']
            rho=row['rho']
            value_sum = value_sum + (gamma**timestep)*reward*rho/sum_of_rho
        values.append(value_sum)
    # print(value_sum)    
    return(value_sum)
    # return(values)
    

####################################################################Q_learning#######################################################################
alpha=0.05

def derive_policy(q_main, epsilon=epsilon):
    # print("hi", q_main)
    policy = np.ones((num_states, num_actions))*epsilon
    for state in range(num_states):
        best_action = np.argmax(q_main[state,:])
        policy[state, best_action]=1-epsilon*(num_actions-1)
    return policy

def make_zero_policy(epsilon=epsilon):
    policy = np.ones((num_states, num_actions), dtype=np.float32)*epsilon
    policy[:, 0]=1-epsilon*(num_actions-1)
    return policy

def make_random_policy(epsilon=epsilon):
    policy = np.ones((num_states, num_actions))*epsilon
    random_actions = np.random.choice(num_actions, size=num_states)
    for state, action in enumerate(random_actions):
        policy[state, action] = 1-epsilon*(num_actions-1)
    return policy

def QLearning(q_table_main, train_data_main):

    groups=train_data_main.groupby('RID')
    for rid, rows in groups:
        for index, row in rows.iterrows():
            state=int(row['states'])
            action=int(row['action'])
            reward = row['reward']
            try:
                next_state = rows.loc[index+1, 'states']
                # max_action_value = q_table[next_state,np.argmax(q_table[next_state])]
                max_action_value=np.max(q_table_main[next_state, :])
                
                q_table_main[state, action] = q_table_main[state, action] + \
                            alpha * (reward + gamma * max_action_value - q_table_main[state,action])
                
                
            except KeyError:
                pass
    return q_table_main



######################################################################################################################mainsplit###############################################

data_path="/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/revision/codes/whole-data/data/main-experiment/"

data1= pd.read_csv(data_path+'whole-data-with-states.csv', low_memory=False)


taken =['RID', 'AGE','CDRSB','MMSE', 'ADAS13','MOCA','CMMED','action','states']

data = data1.loc[:, taken]
# print("data \n", data)

# print("unique values count for CMMED",data.CMMED.value_counts())

##fill empty ADAS13 and MOCA score
data['ADAS13']=data.groupby('RID')['ADAS13'].ffill().bfill()
data['MOCA']=data.groupby('RID')['MOCA'].ffill().bfill()

data['terminal']=0

# print(len(data.index))

for ind in data.index:
  if ind+1<len(data.index):
     if (data['RID'][ind] != data['RID'][ind+1]):
         data.loc[ind,'terminal']=1
  else:
         data.loc[ind,'terminal']=1

# print(data.terminal.value_counts())


obs=['states']
data['reward']=0


for ind in data.index[1:]:
  if (data['terminal'][ind-1]!=1):
     data.loc[ind,'reward']= data.loc[ind,'MMSE']- data.loc[ind-1,'MMSE']

#####################################splitting data###########################################
#split into train/validation/test

weighted_group=np.zeros((100,5),dtype=np.float32)
weighted_group_eighty=np.zeros((100,5),dtype=np.float32)
weighted_group_thirty=np.zeros((100,5),dtype=np.float32)
weighted_group_fifty=np.zeros((100,5),dtype=np.float32)
arr=[]
for repetition in range(100):
    unique_ids = data.RID.unique()
    random.shuffle(unique_ids)
    train_sample=0.8
    test_sample=0.2
    train_num = int(len(unique_ids) * train_sample) 
    train_ids = unique_ids[:train_num]
    test_ids = unique_ids[train_num:]
    
    train_set = pd.DataFrame()
    train_set = data.loc[data['RID'].isin(train_ids)].reset_index(drop=True)

    test_set = pd.DataFrame()
    test_set = data.loc[data['RID'].isin(test_ids)].reset_index(drop=True)
    

    old_p_reward=-100
    old_q_reward=-100
    old_copy_reward=-100
    
    
    old_p_eighty_reward=-100
    old_q_eighty_reward=-100
    old_copy_eighty_reward=-100

    old_p_fifty_reward=-100
    old_q_fifty_reward=-100
    old_copy_fifty_reward=-100

    old_p_thirty_reward=-100
    old_q_thirty_reward=-100
    old_copy_thirty_reward=-100

    for internal_repitition in range(50):

        unique_train_ids=train_set.RID.unique()
        random.shuffle(unique_train_ids)
        full_train_sample=0.8
        val_sample=0.2

        full_train_num = int(len(unique_ids)*full_train_sample) 
        full_train_ids = unique_ids[:train_num]

        full_train_set = pd.DataFrame()
        full_train_set = data.loc[data['RID'].isin(full_train_ids)].reset_index(drop=True)

        val_num = int(len(unique_ids)*val_sample) 
        val_ids=unique_train_ids[-val_num:]
        
        val_set = pd.DataFrame()
        val_set = data.loc[data['RID'].isin(val_ids)].reset_index(drop=True)

        unique_full_train_ids=full_train_set.RID.unique()
        random.shuffle(unique_full_train_ids)
        eighty=0.8
        fifty=0.5
        thirty=0.3

        eighty_num = int(len(unique_full_train_ids) * eighty) 
        fifty_num = int(len(unique_full_train_ids) * fifty) 
        thirty_num = int(len(unique_full_train_ids) * thirty) 
        eighty_ids = unique_full_train_ids[:eighty_num]
        fifty_ids = unique_full_train_ids[:fifty_num]
        thirty_ids = unique_full_train_ids[:thirty_num]


        eighty_set = pd.DataFrame()
        eighty_set = data.loc[data['RID'].isin(eighty_ids)].reset_index(drop=True)

        fifty_set = pd.DataFrame()
        fifty_set = data.loc[data['RID'].isin(fifty_ids)].reset_index(drop=True)

        thirty_set = pd.DataFrame()
        thirty_set = data.loc[data['RID'].isin(thirty_ids)].reset_index(drop=True)

 
        ##########policy_iteration#########

        trans_mat, reward_mean = transition_probability_reward(full_train_set)
        trans_mat_eighty, reward_mean_eighty = transition_probability_reward(eighty_set)
        trans_mat_fifty, reward_mean_fifty = transition_probability_reward(fifty_set)
        trans_mat_thirty, reward_mean_thirty = transition_probability_reward(thirty_set)

        _,p_iteration_policy=policy_iteration(trans_mat, reward_mean)
        _,p_iteration_policy_eighty=policy_iteration(trans_mat_eighty, reward_mean_eighty)
        _,p_iteration_policy_fifty=policy_iteration(trans_mat_fifty, reward_mean_fifty)
        _,p_iteration_policy_thirty=policy_iteration(trans_mat_thirty, reward_mean_thirty)

        ##########behavior_policy#########

        behavior_policy = np.ones((num_states, num_actions), dtype=np.float32)*epsilon
        groups12 = full_train_set.groupby('states')
        for state, rows in groups12:
            cnt = Counter(rows['action'])
            for action, count in cnt.items():
                behavior_policy[int(state), int(action)]=count
        # print(behavior_policy.sum(axis=1)[:, np.newaxis])
        behavior_policy=behavior_policy/behavior_policy.sum(axis=1)[:, np.newaxis]
        
        ##########q_learning########

        q1=np.zeros((num_states, num_actions), dtype=np.float32)
        q2=np.zeros((num_states, num_actions), dtype=np.float32)
        q3=np.zeros((num_states, num_actions), dtype=np.float32)
        q4=np.zeros((num_states, num_actions), dtype=np.float32)
        q_tables=QLearning(q1, full_train_set)
        q_tables_eighty=QLearning(q2, eighty_set)
        q_tables_fifty=QLearning(q3, fifty_set)
        q_tables_thirty=QLearning(q4, thirty_set)
        q_policy=derive_policy(q_tables, epsilon)
        q_policy_eighty=derive_policy(q_tables_eighty, epsilon)
        q_policy_fifty=derive_policy(q_tables_fifty, epsilon)
        q_policy_thirty=derive_policy(q_tables_thirty, epsilon)

        ##########zero_policy#########
        zero_policy=make_zero_policy()

        ##########policy_copy#########
        _,copy_policy=policy_copy(trans_mat, reward_mean)
        _,copy_policy_eighty=policy_copy(trans_mat_eighty, reward_mean_eighty)
        _,copy_policy_fifty=policy_copy(trans_mat_fifty, reward_mean_fifty)
        _,copy_policy_thirty=policy_copy(trans_mat_thirty, reward_mean_thirty)

        ##########random_policy#########
        random_policy=make_random_policy()

        #######weighted_list########## 

        arr.append(repetition)

        current_p_reward=weighted_importance_sampling(p_iteration_policy, val_set,behavior_policy)
        # print("before_comparing",current_p_reward)
        if current_p_reward>=old_p_reward:
            # print(current_p_reward)
            best_p_iteration_policy= p_iteration_policy
            old_p_reward=current_p_reward
            # print(best_p_iteration_policy)
        current_q_reward=weighted_importance_sampling(q_policy, val_set,behavior_policy)
        if current_q_reward>=old_q_reward:
            best_q_policy= q_policy
            old_q_reward=current_q_reward
        current_copy_reward=weighted_importance_sampling(copy_policy, val_set,behavior_policy)
        if current_copy_reward>=old_copy_reward:
            best_copy_policy= copy_policy
            old_copy_reward=current_copy_reward


        current_p_eighty_reward=weighted_importance_sampling(p_iteration_policy_eighty, val_set,behavior_policy)
        if current_p_eighty_reward>=old_p_eighty_reward:
            best_p_iteration_policy_eighty= p_iteration_policy_eighty
            old_p_eighty_reward=current_p_eighty_reward
        current_q_eighty_reward=weighted_importance_sampling(q_policy_eighty, val_set,behavior_policy)
        if current_q_eighty_reward>=old_q_eighty_reward:
            best_q_policy_eighty= q_policy_eighty
            old_q_eighty_reward=current_q_eighty_reward
        current_copy_eighty_reward=weighted_importance_sampling(copy_policy_eighty, val_set,behavior_policy)
        if current_copy_eighty_reward>=old_copy_eighty_reward:
            best_copy_policy_eighty= copy_policy_eighty
            old_copy_eighty_reward=current_copy_eighty_reward


        current_p_fifty_reward=weighted_importance_sampling(p_iteration_policy_fifty, val_set,behavior_policy)
        if current_p_fifty_reward>=old_p_fifty_reward:
            best_p_iteration_policy_fifty= p_iteration_policy_fifty
            old_p_fifty_reward=current_p_fifty_reward
        current_q_fifty_reward=weighted_importance_sampling(q_policy_fifty, val_set,behavior_policy)
        if current_q_fifty_reward>=old_q_fifty_reward:
            best_q_policy_fifty= q_policy_fifty
            old_q_fifty_reward=current_q_fifty_reward
        current_copy_fifty_reward=weighted_importance_sampling(copy_policy_fifty, val_set,behavior_policy)
        if current_copy_fifty_reward>=old_copy_fifty_reward:
            best_copy_policy_fifty= copy_policy_fifty
            old_copy_fifty_reward=current_copy_fifty_reward



        current_p_thirty_reward=weighted_importance_sampling(p_iteration_policy_thirty, val_set,behavior_policy)
        if current_p_thirty_reward>=old_p_thirty_reward:
            best_p_iteration_policy_thirty= p_iteration_policy_thirty
            old_p_thirty_reward=current_p_thirty_reward
        current_q_thirty_reward=weighted_importance_sampling(q_policy_thirty, val_set,behavior_policy)
        if current_q_thirty_reward>=old_q_thirty_reward:
            best_q_policy_thirty= q_policy_thirty
            old_q_thirty_reward=current_q_thirty_reward
        current_copy_thirty_reward=weighted_importance_sampling(copy_policy_thirty, val_set,behavior_policy)
        if current_copy_thirty_reward>=old_copy_thirty_reward:
            best_copy_policy_thirty= copy_policy_thirty
            old_copy_thirty_reward=current_copy_thirty_reward


    ##########behavior_policy#########

    behavior_policy = np.ones((num_states, num_actions), dtype=np.float32)*epsilon
    groups12 = data.groupby('states')
    for state, rows in groups12:
        cnt = Counter(rows['action'])
        # print(cnt)
        for action, count in cnt.items():
            # print(action)
            behavior_policy[int(state), int(action)]=count
    # print(behavior_policy.sum(axis=1)[:, np.newaxis])
    behavior_policy=behavior_policy/behavior_policy.sum(axis=1)[:, np.newaxis]
    

    ##########zero_policy#########
    zero_policy=make_zero_policy()


    ##########random_policy#########
    random_policy=make_random_policy()

    #######weighted_list########## 

    arr.append(repetition)

    weighted_group[repetition,:]=[weighted_importance_sampling(best_p_iteration_policy,test_set,behavior_policy),weighted_importance_sampling(best_q_policy,test_set,behavior_policy), weighted_importance_sampling(zero_policy,test_set,behavior_policy), weighted_importance_sampling(random_policy,test_set,behavior_policy),weighted_importance_sampling(best_copy_policy,test_set,behavior_policy)]


    weighted_group_eighty[repetition,:]=[weighted_importance_sampling(best_p_iteration_policy_eighty,test_set,behavior_policy),weighted_importance_sampling(best_q_policy_eighty,test_set,behavior_policy), weighted_importance_sampling(zero_policy,test_set,behavior_policy), weighted_importance_sampling(random_policy,test_set,behavior_policy),weighted_importance_sampling(best_copy_policy_eighty,test_set,behavior_policy)]


    weighted_group_fifty[repetition,:]=[weighted_importance_sampling(best_p_iteration_policy_fifty,test_set,behavior_policy),weighted_importance_sampling(best_q_policy_fifty,test_set,behavior_policy), weighted_importance_sampling(zero_policy,test_set,behavior_policy), weighted_importance_sampling(random_policy,test_set,behavior_policy),weighted_importance_sampling(best_copy_policy_fifty,test_set,behavior_policy)]


    weighted_group_thirty[repetition,:]=[weighted_importance_sampling(best_p_iteration_policy_thirty,test_set,behavior_policy),weighted_importance_sampling(best_q_policy_thirty,test_set,behavior_policy), weighted_importance_sampling(zero_policy,test_set,behavior_policy), weighted_importance_sampling(random_policy,test_set,behavior_policy),weighted_importance_sampling(best_copy_policy_thirty,test_set,behavior_policy)]


print("best policy iteration=",best_p_iteration_policy.tolist())
print("\n")
print("current_p_reward=", current_p_reward)

print("best q learning=",best_q_policy.tolist())
print("\n")
print("current_q_reward=", current_q_reward)

print("weighted_group=", weighted_group.tolist())
print("weighted_group_eighty=", weighted_group_eighty.tolist())
print("weighted_group_fifty=", weighted_group_fifty.tolist())
print("weighted_group_thirty=", weighted_group_thirty.tolist())



###################################################################################plotting################################################################
# print(weighted_group)
import matplotlib.pyplot as plt

 
# Creating dataset
np.random.seed(10)
 
fig, ((ax, ax1), (ax2, ax3))= plt.subplots(2,2,figsize=(15, 6))

main=ax.boxplot(weighted_group, patch_artist=True)
m = weighted_group.mean(axis=0)
st = weighted_group.std(axis=0)
ax.set_title('100% Train Set')
for i, line in enumerate(main['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(m[i], st[i])
    ax.annotate(text, xy=(x, y))
ax.set_xticks([1, 2,3,4, 5])
ax.set_xticklabels(["policy_iteration","q_learning", "zero", "random",'clinician'], rotation=10)
ax.set_ylim(-20,10)

colors=["forestgreen", "blue", "dimgray",  "slategray", "tomato"]
 
for patch, color in zip(main['boxes'],colors):
    patch.set_facecolor(color)

eighty=ax1.boxplot(weighted_group_eighty, patch_artist=True)
m1 = weighted_group_eighty.mean(axis=0)
st1 = weighted_group_eighty.std(axis=0)
for i, line in enumerate(eighty['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(m1[i], st1[i])
    ax1.annotate(text, xy=(x, y))
ax1.set_title('80% Train Set')
ax1.set_xticks([1, 2,3,4, 5])
ax1.set_xticklabels(["policy_iteration","q_learning", "zero", "random",'clinician'], rotation=10)
ax1.set_ylim(-20,10)

colors=["forestgreen", "blue", "dimgray",  "slategray", "tomato"]
 
for patch, color in zip(eighty['boxes'],colors):
    patch.set_facecolor(color)
fifty=ax2.boxplot(weighted_group_fifty, patch_artist=True)
m2 = weighted_group_fifty.mean(axis=0)
st2 = weighted_group_fifty.std(axis=0)
for i,line in enumerate(fifty['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(m2[i], st2[i])
    ax2.annotate(text, xy=(x, y))
ax2.set_title('50% Train Set')
ax2.set_xticks([1, 2,3,4, 5])
ax2.set_xticklabels(["policy_iteration","q_learning", "zero", "random",'clinician'], rotation=10)
ax2.set_ylim(-20,10)

colors=["forestgreen", "blue", "dimgray",  "slategray", "tomato"]
 
for patch, color in zip(fifty['boxes'],colors):
    patch.set_facecolor(color)
thirty=ax3.boxplot(weighted_group_thirty, patch_artist=True)
m3 = weighted_group_thirty.mean(axis=0)
st3 = weighted_group_thirty.std(axis=0)
for i, line in enumerate(thirty['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(m3[i], st3[i])
    ax3.annotate(text, xy=(x, y))
ax3.set_title('30% Train Set')
ax3.set_xticks([1, 2,3,4, 5])
ax3.set_xticklabels(["policy_iteration","q_learning", "zero", "random",'clinician'], rotation=10)
ax3.set_ylim(-20,10)

colors=["forestgreen", "blue", "dimgray",  "slategray", "tomato"]
 
for patch, color in zip(thirty['boxes'],colors):
    patch.set_facecolor(color)
 

plt.savefig('5000_full_colored_hi.png')
# show plot
plt.show()

