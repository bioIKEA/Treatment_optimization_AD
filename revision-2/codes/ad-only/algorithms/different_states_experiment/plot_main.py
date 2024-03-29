import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter

pd.options.display.float_format = '{:.2f}'.format

ad_only_5=  [[-1.9021345 ],
 [-3.173421  ],
 [-6.7482476 ],
 [-0.12369895],
 [-0.45708546],
 [-3.725393  ],
 [ 1.5731839 ],
 [-1.5571496 ],
 [-0.708563  ],
 [-0.12772883],
 [-2.756554  ],
 [-1.1524941 ],
 [ 0.25742146],
 [-1.8885543 ],
 [-1.3168622 ],
 [-2.7354186 ],
 [-2.37238   ],
 [-2.4176946 ],
 [ 1.7194309 ],
 [ 0.2918277 ],
 [ 0.66431093],
 [ 1.0666486 ],
 [-1.6189151 ],
 [-1.0467775 ],
 [-1.8329357 ],
 [-0.05422183],
 [-2.083256  ],
 [-1.5932606 ],
 [-1.2829306 ],
 [-2.5562565 ],
 [-3.292758  ],
 [-1.4752963 ],
 [-1.0148673 ],
 [-0.47361684],
 [-1.0918822 ],
 [ 0.25212476],
 [-7.3400545 ],
 [-1.310754  ],
 [-4.0708604 ],
 [-2.186042  ],
 [-1.7922729 ],
 [-3.8231556 ],
 [ 1.7612073 ],
 [ 0.43214715],
 [-2.3665287 ],
 [ 0.3934537 ],
 [-2.2834473 ],
 [-0.9725746 ],
 [-0.08514278],
 [-1.3910125 ],
 [-3.0328817 ],
 [-0.02586456],
 [-3.4894104 ],
 [ 2.0618892 ],
 [-3.538855  ],
 [-2.4684727 ],
 [-3.157712  ],
 [-1.5150412 ],
 [-1.0380169 ],
 [-1.2055016 ],
 [ 0.02574223],
 [ 0.43627357],
 [ 1.7385707 ],
 [-0.5171519 ],
 [-1.8142291 ],
 [-0.39224726],
 [ 1.5145484 ],
 [-1.7193233 ],
 [ 0.0531235 ],
 [-2.2754436 ],
 [-2.501755  ],
 [-0.2207276 ],
 [-2.9393165 ],
 [-1.1703452 ],
 [-1.2950923 ],
 [-3.369303  ],
 [-2.240104  ],
 [ 0.42530409],
 [ 0.956659  ],
 [ 0.06314766],
 [-0.65244746],
 [-1.9530401 ],
 [-2.119668  ],
 [-2.5311322 ],
 [-3.0958064 ],
 [-2.554345  ],
 [-2.5553622 ],
 [-2.5056765 ],
 [-2.1877034 ],
 [-0.9196644 ],
 [-2.9475    ],
 [-6.113001  ],
 [-1.8679869 ],
 [-1.4240216 ],
 [-0.94401276],
 [-3.8272507 ],
 [-3.9718044 ],
 [-2.1354527 ],
 [-2.5885403 ],
 [-2.2268355 ]]


ad_only_9=  [[-3.183246  ],
 [-1.0902733 ],
 [-1.7677819 ],
 [-2.435556  ],
 [-0.29003936],
 [-4.877305  ],
 [ 1.7286237 ],
 [ 0.46872783],
 [ 0.7566318 ],
 [-1.3451642 ],
 [-5.2715964 ],
 [-1.2522776 ],
 [-0.9875041 ],
 [-1.780512  ],
 [-0.12316321],
 [-0.45853975],
 [-0.07295078],
 [-0.22083363],
 [-1.039217  ],
 [-0.690693  ],
 [-1.6687922 ],
 [-1.5709738 ],
 [ 0.04712154],
 [-3.108138  ],
 [ 1.2450541 ],
 [ 0.86401635],
 [-3.0265331 ],
 [-1.0974696 ],
 [-0.5216691 ],
 [-1.0926207 ],
 [-1.5975372 ],
 [-0.3606199 ],
 [-3.1639624 ],
 [-4.9708614 ],
 [-2.5999599 ],
 [-1.0925214 ],
 [-0.767202  ],
 [ 0.5846809 ],
 [ 0.0178124 ],
 [-3.5966475 ],
 [-0.9034553 ],
 [-4.3576875 ],
 [-1.8900397 ],
 [-4.9433155 ],
 [-1.2303863 ],
 [-0.71993876],
 [-1.6299413 ],
 [-1.2185556 ],
 [-2.16164   ],
 [-0.5168781 ],
 [-2.1871037 ],
 [-1.4297551 ],
 [-0.3943698 ],
 [ 0.8954218 ],
 [-3.673605  ],
 [-1.6764128 ],
 [-0.5468468 ],
 [ 2.7139587 ],
 [-0.9980041 ],
 [-2.970326  ],
 [-1.0987418 ],
 [-2.5606215 ],
 [-2.1484787 ],
 [-4.657935  ],
 [-4.3845935 ],
 [-1.6416686 ],
 [-1.2267884 ],
 [-1.8660264 ],
 [-0.9423356 ],
 [-0.54875225],
 [-0.4384858 ],
 [ 0.16943373],
 [-0.07895764],
 [-1.4549618 ],
 [-1.9764084 ],
 [-0.18620749],
 [-0.6695359 ],
 [-3.0621638 ],
 [-5.0523224 ],
 [-2.0640364 ],
 [-2.4299922 ],
 [-1.8568124 ],
 [-0.9815708 ],
 [ 1.0434036 ],
 [ 0.5575426 ],
 [-1.9027418 ],
 [-1.3558687 ],
 [-3.5299075 ],
 [-4.278906  ],
 [ 0.34962827],
 [ 1.8929441 ],
 [-1.4560866 ],
 [-3.6792595 ],
 [-2.747783  ],
 [-0.41876698],
 [-2.0209794 ],
 [-0.54012716],
 [-4.3192005 ],
 [-0.39175972],
 [-0.7171733 ]]

 
ad_only_10=  [[-1.5474288 ],
 [-2.2563655 ],
 [-3.429945  ],
 [-2.0920093 ],
 [ 0.51494753],
 [-1.4230618 ],
 [-1.2388407 ],
 [-1.3918202 ],
 [-0.99106336],
 [-2.3031342 ],
 [ 1.1925585 ],
 [-0.15917054],
 [-0.18997735],
 [-0.35962915],
 [-1.6753075 ],
 [-2.676046  ],
 [-0.1435569 ],
 [-1.8726265 ],
 [-2.7563148 ],
 [-3.2386758 ],
 [-2.1007063 ],
 [-0.63115126],
 [-1.9114302 ],
 [-0.27067617],
 [ 0.4792434 ],
 [-0.07436908],
 [ 0.779337  ],
 [-0.68149596],
 [-2.264811  ],
 [-2.1472633 ],
 [-2.4603572 ],
 [ 0.53538144],
 [-3.896694  ],
 [ 0.12220097],
 [ 0.18725707],
 [-1.2995638 ],
 [-0.14137267],
 [-2.6674716 ],
 [-1.8449634 ],
 [ 0.4285261 ],
 [-1.2048876 ],
 [-1.2139394 ],
 [-0.08229684],
 [ 0.03395736],
 [-2.5953462 ],
 [-1.3410391 ],
 [-4.258478  ],
 [-0.66956544],
 [-1.4347227 ],
 [-4.5248146 ],
 [-3.407435  ],
 [-2.7087226 ],
 [-0.22123721],
 [-3.322312  ],
 [-0.43320823],
 [ 0.39861733],
 [ 0.01203712],
 [-1.6764882 ],
 [-3.7207417 ],
 [-0.69469047],
 [-5.667946  ],
 [-0.1814644 ],
 [-2.2495968 ],
 [-1.9528275 ],
 [-0.17218362],
 [-3.0015104 ],
 [-2.5355227 ],
 [-3.938351  ],
 [-1.570136  ],
 [ 0.2863142 ],
 [-1.3521358 ],
 [-0.12216028],
 [-1.637263  ],
 [-2.1223366 ],
 [ 0.21838532],
 [-1.091679  ],
 [-1.6338909 ],
 [-0.6506033 ],
 [ 0.43270308],
 [-5.0216746 ],
 [-1.1147275 ],
 [-1.8321482 ],
 [ 0.0718589 ],
 [ 0.8484154 ],
 [ 1.6351781 ],
 [-0.35598704],
 [-0.5997027 ],
 [-2.0076578 ],
 [ 0.22220306],
 [ 0.70895135],
 [-2.4262989 ],
 [-4.08324   ],
 [ 0.05814546],
 [-0.39182064],
 [-1.4689345 ],
 [-1.8701057 ],
 [ 0.17136052],
 [-3.1715467 ],
 [-1.9107997 ],
 [-1.4195664 ]]

weighted_group = np.concatenate((ad_only_10, ad_only_9, ad_only_5), axis=1)
print(weighted_group)
import matplotlib.pyplot as plt

# Creating dataset
np.random.seed(10)

fig, ax= plt.subplots()

main=ax.boxplot(weighted_group, patch_artist=True)
m = weighted_group.mean(axis=0)
st = weighted_group.std(axis=0)
ax.set_title('AD only data: Q-learning rewards for different States')
for i, line in enumerate(main['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(m[i], st[i])
    ax.annotate(text, xy=(x, y))
ax.set_xticks([1, 2,3])
ax.set_xticklabels(["10 states for samples>50","9 states for samples>100", "5 states for samples>200"], rotation=10)
# ax.set_ylim(-20,10)

colors=["forestgreen", "blue", "dimgray"]
 
for patch, color in zip(main['boxes'],colors):
    patch.set_facecolor(color)

 

plt.savefig('ad-only.png')
# show plot
plt.show()
