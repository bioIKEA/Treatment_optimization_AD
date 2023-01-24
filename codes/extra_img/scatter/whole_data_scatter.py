import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter

pd.options.display.float_format = '{:.2f}'.format

whole= pd.read_csv('whole_only.csv', low_memory=False)

lis={0:whole}
lis1=["Whole data cohort"]


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

df=lis[0]
x=df.reward_q
y1=df.reward

fig = make_subplots(rows=1, cols=1, subplot_titles=lis1,horizontal_spacing = 0.2, vertical_spacing=0.2)

fig.add_trace(go.Scatter(x=lis[0].reward_q,
                          y=lis[0].reward,
                        mode='markers',
    marker=dict(
            color='white',
            size=12,
            line=dict(
                color='MediumPurple',
                width=2
            )
        ),
                        ), row=1,col=1)
fig.add_shape(type="line",
              x0=-25, 
              y0=-25, 
              x1=5, 
              y1=5,
              row=1,
              col=1)
              
fig.update_yaxes(tick0=-25, dtick=5)
fig.update_xaxes(tick0=-25, dtick=5)
# Update xaxis properties
fig.update_xaxes(title_text="Q-learning predicted reward (MMSE)", row=1, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Clinician Policy predicted reward (MMSE)", row=1, col=1)

fig.update_layout(height=500, width=500,plot_bgcolor='aliceblue', title_text="Predicted rewards accross different data cohort",showlegend=False)

fig.write_image("whole_scatter.png")
fig.show()

'''

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
    fig=lis[counter].plot( kind="scatter",  x='reward_q', y='reward',facecolors='none',title=f"{lis1[counter]} data cohort", figsize=(15, 10),s=25,ax=ax)
    curr=lis[counter]
    fig.fill_between(curr.reward_q, curr.reward_q.mean() - 2*curr.reward_q.std(), curr.reward_q.mean() + 2*curr.reward_q.std(), color='red', alpha=0.2)
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
plt.savefig('final_test1.png')
    # plt.show()
    # fig.clf()
    # fig.show()

'''
