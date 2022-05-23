from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import csv, operator


class AlzheimersEnv(Env):
    # global total_visitor
    # global episode_counter
    episode_counter=0
    # global visitor_counter
    visitor_counter=0
    total_visitor=0
    total_episode=0
    # global total_episode
    def __init__(self):
        #extact data
        self.data=self.data()
        AlzheimersEnv.total_visitor=len(self.data)-1

        self.normal_MMSE, self.normal_ADAS13=self.get_records(AlzheimersEnv.visitor_counter,0)
        #discrete actions: 2 main medication, no medication
        self.action_space=Discrete(4)
        # self.normal_ADAS13=5
        # self.normal_MMSE=30
        self.AD_ADAS13=85
        self.AD_MMSE=8
        #MMSE,ADAS13
        high = np.array(
                    [
                        self.normal_MMSE,
                        self.normal_ADAS13
                    ],
                    dtype=np.float32,
                )
        low = np.array(
                    [
                        self.AD_MMSE,
                        self.AD_ADAS13
                    ],
                    dtype=np.float32,
                )
        #observation of MMSE score of patients 
        self.observation_space=Box(low, high,dtype=np.float32,)

    def data(self):
        file = open('/Users/kritibbhattarai/Desktop/Combined.csv')
        csvreader = csv.reader(file, delimiter=',')
        reader=next(csvreader)
        # print(reader.index('CMMED'))
        lis=[]
        dic={}
        for row in csvreader:
            #RID
            id_1=float(row[0])
            #MMSE
            mmse=float(row[21])
            #ADAS13
            ADAS13=float(row[20])
            #action
            action=float(row[98])

            if id_1 in dic:
                dic[id_1].append((mmse,ADAS13,action))
            else:
                dic={}
                dic[id_1]=([(mmse,ADAS13,action)])
                lis.append(dic)
        return(lis)
        # print(lis)
        # print(len(lis))

    def step(self, action):
        AlzheimersEnv.episode_counter+=1
        self.original_MMSE=self.normal_MMSE
        # print(AlzheimersEnv.visitor_counter,AlzheimersEnv.episode_counter)

        self.normal_MMSE, self.normal_ADAS13=self.get_records(AlzheimersEnv.visitor_counter,AlzheimersEnv.episode_counter)
        self.state=(self.normal_MMSE, self.normal_ADAS13)
        
        reward=self.original_MMSE-self.normal_MMSE
        if AlzheimersEnv.episode_counter<AlzheimersEnv.total_episode-1:
            done=False
        else:
            done=True

        info={}
        return np.array(self.state, dtype=np.float32), reward, done, {}


    def render(self):
        pass
    def reset(self):
        if AlzheimersEnv.visitor_counter<AlzheimersEnv.total_visitor:
            AlzheimersEnv.visitor_counter+=1
            AlzheimersEnv.episode_counter=0
        state=self.get_records(AlzheimersEnv.visitor_counter,AlzheimersEnv.episode_counter)
        self.state=state
        return self.state

    def get_records(self,visitor_counter,episode_counter):
        
        #a particular patient record of all the visits and patient's id
        n_patient_record=self.data[visitor_counter]
        #patient_id
        n_patient_id=list(n_patient_record.keys())[0]
        #nth visit record of that particular patient
        n_visit_record=n_patient_record[n_patient_id][episode_counter]
        #total visits for one patient
        AlzheimersEnv.total_episode=len(n_patient_record[n_patient_id])
        # print("total episode", AlzheimersEnv.total_episode)
        #creating states
        self.normal_MMSE, self.normal_ADAS13, _=n_visit_record
        return (self.normal_MMSE, self.normal_ADAS13)
