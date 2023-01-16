import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter

pd.options.display.float_format = '{:.2f}'.format

ad= pd.read_csv('ad_only.csv', low_memory=False)
ad["reward_q"]+=25
ad["reward"]+=25
print(ad.head())
ad_hyp_dep= pd.read_csv('ad_hyp_dep.csv', low_memory=False)
ad_hyp_dep["reward_q"]+=25
ad_hyp_dep["reward"]+=25
ad_hyp= pd.read_csv('ad-hyp.csv', low_memory=False)
ad_hyp["reward_q"]+=25
ad_hyp["reward"]+=25
whole= pd.read_csv('whole_only.csv', low_memory=False)
whole["reward_q"]+=25
whole["reward"]+=25

lis={0:ad, 1:ad_hyp_dep, 2:ad_hyp, 3:whole}
lis1=["AD data cohort", "AD,Hypertension,Depression data", "AD Hypertension data", "Whole data cohort"]


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

df=lis[0]
x=df.reward_q
y1=df.reward

fig = make_subplots(rows=2, cols=2, subplot_titles=lis1,horizontal_spacing = 0.2, vertical_spacing=0.2)

#plot 1
fig.add_trace(go.Scatter(x=df.index,y=df.reward_q,
                         mode="lines+markers+text",
                         name="Q-learning reward",
                        ), row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=df.reward,
                         mode="lines+markers",
                         name="Clinician's Policy reward",
                        opacity=0.7,
                        ), row=1,col=1)

fig.add_annotation(x=38,y=12,
            text="Clinician's Policy reward",
                   ax=45,
                   ay=15,
            showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
            ),
            arrowhead=1, row=1,col=1)
fig.add_annotation(x=31,y=24.99,
            text="Q-learning reward",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
            ),
            showarrow=True,
            arrowhead=1, row=1,col=1)
#plot 2
fig.add_trace(go.Scatter(x=lis[1].index,
                          y=lis[1].reward_q,
                         name="Q-learning reward",
                         mode="lines+markers",
                            ),row=1,col=2)
fig.add_trace(go.Scatter(x=lis[1].index,
                          y=lis[1].reward,
                        opacity=0.7,
                         name="Clinician's Policy reward",
                         mode="lines+markers",
                            ),row=1,col=2)

fig.add_annotation(x=25,y=5,
            text="Clinician's Policy reward",
                   ax=30,
                   ay=20,
            showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
            ),
            arrowhead=1, row=1,col=2)
fig.add_annotation(x=82,y=25.0,
            text="Q-learning reward",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
            ),
            showarrow=True,
            arrowhead=1, row=1,col=2)


#plot 3
fig.add_trace(go.Scatter(x=lis[2].index,
                          y=lis[2].reward_q,
                         name="Q-learning reward",
                         mode="lines+markers",
                            ),row=2,col=1)
fig.add_trace(go.Scatter(x=lis[2].index,
                          y=lis[2].reward,
                        opacity=0.7,
                         name="Clinician's Policy reward",
                         mode="lines+markers",
                            ),row=2,col=1)

fig.add_annotation(x=32,y=1,
            text="Clinician's Policy reward",
                   ax=30,
                   ay=20,
            showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
            ),
            arrowhead=1, row=2,col=1)
fig.add_annotation(x=53,y=25.0,
            text="Q-learning reward",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
            ),
            showarrow=True,
            arrowhead=1, row=2,col=1)


#plot 4
fig.add_trace(go.Scatter(x=lis[3].index,
                          y=lis[3].reward_q,
                         mode="lines+markers",
                            marker=dict(color="red"),
                         name="Q-learning reward",
                            ),row=2,col=2)
fig.add_trace(go.Scatter(x=lis[3].index,
                          y=lis[3].reward,
                         mode="lines+markers",
                            marker=dict(color="orange"),
                        opacity=0.7,
                         name="Clinician's Policy reward",
                            ),row=2,col=2)

fig.add_annotation(x=127,y=3,
            text="Clinician's Policy reward",
                   ax=30,
                   ay=20,
            showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
            ),
            arrowhead=1, row=2,col=2)
fig.add_annotation(x=116,y=25.0,
            text="Q-learning reward",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
            ),
            showarrow=True,
            arrowhead=1, row=2,col=2)


              
# fig.add_shape(type="line",
#               x0=-25, 
#               y0=-25, 
#               x1=5, 
#               y1=5,
#               row=1,
#               col=2)

# fig.add_shape(type="line",
#               x0=-25, 
#               y0=-25, 
#               x1=5, 
#               y1=5,
#               row=2,
#               col=1)
# fig.add_shape(type="line",
#               x0=-25, 
#               y0=-25, 
#               x1=5, 
#               y1=5,
#               row=2,
#               col=2)
# fig.update_traces(marker=dict(
#             color='white',
#             size=12,
#             line=dict(
#                 color='blue',
#                 width=2
#             )
#         ),
#                   selector=dict(mode='markers'))
                            
# fig.update_yaxes(tick0=-25, dtick=5)
# fig.update_xaxes(tick0=-25, dtick=5)
# Update xaxis properties
fig.update_xaxes(title_text="Patients", row=1, col=1)
fig.update_xaxes(title_text="Patients", row=1, col=2)
fig.update_xaxes(title_text="Patients", row=2, col=1)
fig.update_xaxes(title_text="Patients", row=2, col=2)
# fig.update_xaxes(title_text="Q-learning predicted reward (MMSE)", row=1, col=2)
# fig.update_xaxes(title_text="Q-learning predicted reward (MMSE)", row=2, col=1)
# fig.update_xaxes(title_text="Q-learning predicted reward (MMSE)", row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Rewards (MMSE score)", row=1, col=1)
fig.update_yaxes(title_text="Rewards (MMSE score)", row=1, col=2)
fig.update_yaxes(title_text="Rewards (MMSE score)", row=2, col=1)
fig.update_yaxes(title_text="Rewards (MMSE score)", row=2, col=2)
# fig.update_yaxes(title_text="Behavior Policy predicted reward (MMSE)", row=1, col=2)
# fig.update_yaxes(title_text="Behavior Policy predicted reward (MMSE)", row=2, col=1)
# fig.update_yaxes(title_text="Behavior Policy predicted reward (MMSE)", row=2, col=2)


fig.update_layout(height=800, width=800, plot_bgcolor='aliceblue',title_text="Predicted rewards accross different data cohort",showlegend=False)

fig.write_image("test_dec12.png")
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
plt.savefig('fin.png')
    # plt.show()
    # fig.clf()
    # fig.show()
'''
