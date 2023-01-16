import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter

pd.options.display.float_format = '{:.2f}'.format

ad= pd.read_csv('ad_only.csv', low_memory=False)
ad_hyp_dep= pd.read_csv('ad_hyp_dep.csv', low_memory=False)
ad_hyp= pd.read_csv('ad-hyp.csv', low_memory=False)
whole= pd.read_csv('whole_only.csv', low_memory=False)

lis={0:ad, 1:ad_hyp_dep, 2:ad_hyp, 3:whole}
lis1=["AD_only", "AD_Hypertension_Depression", "AD_Hypertension", "Whole"]

import matplotlib.pyplot as plt
# ax1 = result.plot(kind='scatter', x='reward_q', y='reward')    
# ax1.set_xlim(-25,5)
# # ax2 = result.plot(kind='scatter', x='reward', y='reward_q', color='g', ax=ax1)    
# plt.savefig('ad_only_w_reward.png')
# plt.show()
# fig=df.plot(x="index", kind="bar", stacked=True, title="States Actions Count", figsize=(15, 10))
for counter in range(4):
    # a=lis[counter]
    ax=plt.subplot(2,2,counter+1)
    ax.set_xlim(-25,5)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    fig=lis[counter].plot( kind="scatter",  x='reward_q', y='reward',facecolors='none',title=f"{lis1[counter]} data cohort", figsize=(15, 10),s=50,ax=ax)
    fig.set_facecolor('#D3D3D3')
    fig.set_xlabel("Q-learning predicted reward (MMSE)")
    fig.set_ylabel("Behavior Policy Predicted reward (MMSE)")
    plt.tight_layout()

    # fig=row.T.plot( kind="bar",  title="States Actions Count",color={'No':["b"], 'In':["b"],'Na':["b"],'Hy':["b"],'Ni':["b"],'So':["b"],'No_ai':["r"],'In_ai':["r"],'Na_ai':["r"],'Hy_ai':["r"],'Ni_ai':["r"],'So_ai':["r"]}, figsize=(15, 10))
    # plt.ylim(0,900)
    # fig.set_figheight(10)
    # fig.set_figwidth(15)
    # plt.xlabel("actions")
    # plt.ylabel("counts")

    # plt.show()
plt.savefig('final_test.png')
    # plt.show()
    # fig.clf()
    # fig.show()
