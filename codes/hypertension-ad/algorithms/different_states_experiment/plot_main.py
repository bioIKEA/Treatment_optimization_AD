import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter

pd.options.display.float_format = '{:.2f}'.format

ad_hyp_6=  [[-1.5728543 ],
 [-0.84561956],
 [-0.6648155 ],
 [-3.0816667 ],
 [-2.9641118 ],
 [-3.1724694 ],
 [-4.262516  ],
 [-3.4579065 ],
 [ 1.7575779 ],
 [ 0.7299205 ],
 [-1.4279991 ],
 [-2.9829497 ],
 [-3.137629  ],
 [-3.2539575 ],
 [-2.2373836 ],
 [-2.0334365 ],
 [-4.0573835 ],
 [-3.0994923 ],
 [-0.18330875],
 [-1.3317341 ],
 [-1.2282678 ],
 [-2.210219  ],
 [ 1.2188982 ],
 [-0.03160502],
 [-0.43406984],
 [-0.89335895],
 [ 1.7066904 ],
 [-0.16295443],
 [-2.0382736 ],
 [ 0.2335013 ],
 [-0.86992437],
 [ 0.3675267 ],
 [-0.68235195],
 [-1.9499376 ],
 [-2.6297214 ],
 [-0.28697187],
 [ 0.5632262 ],
 [-2.2082443 ],
 [-1.0564715 ],
 [-0.37749594],
 [ 0.46412554],
 [-2.457613  ],
 [-1.7212151 ],
 [-2.2133172 ],
 [-4.5592747 ],
 [-2.2070165 ],
 [-0.78871894],
 [-0.83717185],
 [-0.7313314 ],
 [-3.9090304 ],
 [-3.6976392 ],
 [-0.05587395],
 [-0.37974665],
 [-3.4723797 ],
 [ 0.7489788 ],
 [-2.8714254 ],
 [-0.38284042],
 [-1.8538768 ],
 [ 0.8311222 ],
 [ 1.0275583 ],
 [-1.1702574 ],
 [-2.59392   ],
 [ 0.23566802],
 [-0.16923705],
 [-0.25876737],
 [-0.8733028 ],
 [ 0.88883007],
 [-4.5533853 ],
 [-4.175216  ],
 [-4.0286355 ],
 [-1.4119459 ],
 [-0.55264705],
 [ 0.545662  ],
 [-0.25347355],
 [-1.0497581 ],
 [-1.7043324 ],
 [-1.0798378 ],
 [-0.5292912 ],
 [-2.2791524 ],
 [-0.69170064],
 [ 0.4832067 ],
 [-0.10791736],
 [-3.945628  ],
 [ 0.02171293],
 [-1.6299657 ],
 [-0.3235565 ],
 [-3.516077  ],
 [-0.34621894],
 [-5.545423  ],
 [-1.914763  ],
 [-2.13694   ],
 [-1.7747813 ],
 [-2.7025104 ],
 [-1.1678264 ],
 [-0.8200039 ],
 [-2.778369  ],
 [-0.33453682],
 [-2.5779305 ],
 [-1.9707806 ],
 [-1.0388399 ]]

ad_hyp_10=  [[-3.4268279e+00],
 [-1.2600949e+00],
 [-2.5942750e+00],
 [-1.6960231e+00],
 [-4.0507903e+00],
 [-7.9445690e-01],
 [-2.5985806e+00],
 [-3.5282439e-01],
 [-2.9367471e+00],
 [-4.8536345e-01],
 [-1.5525542e+00],
 [-3.4904642e+00],
 [-1.0104324e+00],
 [-1.9970585e+00],
 [ 6.5310544e-01],
 [-2.2012587e+00],
 [-9.0127653e-01],
 [-9.4969302e-01],
 [-4.3884274e-01],
 [-1.3043146e+00],
 [-1.9323761e+00],
 [-4.5802513e-01],
 [-1.9258430e+00],
 [-1.8066856e+00],
 [-6.8900180e+00],
 [-9.8035794e-01],
 [-1.6249827e+00],
 [-1.6563160e+00],
 [-1.6482797e+00],
 [-1.2363470e+00],
 [-2.0315838e-03],
 [-1.9422069e+00],
 [-2.9226443e-01],
 [-5.5851173e-01],
 [-3.9472487e+00],
 [-1.0091499e+00],
 [ 3.7378243e-01],
 [ 8.7934965e-01],
 [-5.3635502e+00],
 [-4.9633656e+00],
 [-3.9403886e-01],
 [-3.7795417e+00],
 [ 1.5033933e+00],
 [ 1.8126348e-01],
 [-8.9681655e-02],
 [-1.9230036e+00],
 [ 8.9358515e-01],
 [-2.5234389e+00],
 [-3.9919585e-01],
 [-2.1252322e+00],
 [-2.4421198e+00],
 [-1.0494653e+00],
 [-9.7213799e-01],
 [-3.0284493e+00],
 [-7.5128675e-01],
 [-2.3054130e+00],
 [-2.0695022e-01],
 [-2.1350131e+00],
 [ 6.9964737e-02],
 [-3.1641264e+00],
 [-2.3738728e-01],
 [-1.7083744e+00],
 [-2.0757570e+00],
 [-8.4653866e-01],
 [ 5.1743853e-01],
 [-1.9254491e+00],
 [-4.1705699e+00],
 [-2.1701241e+00],
 [-1.4122401e+00],
 [-1.4672375e-01],
 [-1.9719052e+00],
 [-1.0448784e+00],
 [ 3.3145651e-01],
 [-2.4059455e+00],
 [-3.0775216e+00],
 [-8.5966718e-01],
 [-2.5636239e+00],
 [-2.9377396e+00],
 [-2.4827702e+00],
 [-2.5808749e+00],
 [-1.5348449e+00],
 [-9.9464709e-01],
 [-1.6988416e+00],
 [-1.1266503e+00],
 [-1.7791514e+00],
 [-2.8316734e+00],
 [-1.1739389e+00],
 [-3.0945113e+00],
 [ 1.8650607e+00],
 [-3.5628884e+00],
 [ 1.2756983e+00],
 [ 8.8657153e-01],
 [-1.6149921e+00],
 [ 1.3881197e+00],
 [-4.4854042e-01],
 [-3.7581933e+00],
 [-1.6980362e+00],
 [-3.7610159e+00],
 [-1.0685568e+00],
 [-2.4210792e+00]]


ad_hyp_11=  [[-1.3303622 ],
 [-0.3876099 ],
 [-1.5452303 ],
 [-1.8834049 ],
 [-1.0107143 ],
 [-0.8582707 ],
 [-0.5336087 ],
 [-2.162726  ],
 [-3.0128133 ],
 [ 0.48541313],
 [-1.4369293 ],
 [-1.9102507 ],
 [-3.8461018 ],
 [-4.1805277 ],
 [ 0.9109535 ],
 [-1.0100708 ],
 [-1.2659024 ],
 [-1.2024851 ],
 [-3.8017905 ],
 [-1.6649604 ],
 [-2.875483  ],
 [-1.196343  ],
 [-0.8587115 ],
 [-2.6369534 ],
 [-6.0941024 ],
 [-2.802258  ],
 [-1.7502787 ],
 [ 0.30756935],
 [-1.1068105 ],
 [-1.5007962 ],
 [-0.97352535],
 [ 1.2064241 ],
 [-1.0190781 ],
 [-0.77551687],
 [-1.3119668 ],
 [-0.26694986],
 [-3.991613  ],
 [-2.8397126 ],
 [ 0.37797213],
 [-2.7104714 ],
 [-0.6426808 ],
 [-0.77534133],
 [-2.6733325 ],
 [-1.1386068 ],
 [-1.4981968 ],
 [ 0.11226091],
 [-0.22433805],
 [-0.32171944],
 [-2.9728158 ],
 [-3.933002  ],
 [-3.3606813 ],
 [-1.9981154 ],
 [-1.9145646 ],
 [-0.12825717],
 [-1.559267  ],
 [ 0.4020363 ],
 [-0.9336706 ],
 [-1.8355458 ],
 [ 0.5098114 ],
 [-1.3116497 ],
 [-0.831756  ],
 [-4.2081695 ],
 [-2.0357695 ],
 [-0.73771393],
 [-0.48052993],
 [ 0.17506313],
 [-4.08775   ],
 [ 1.3462428 ],
 [-2.1299143 ],
 [-1.7573222 ],
 [-1.3725255 ],
 [ 0.5557692 ],
 [-1.9486814 ],
 [-2.7797623 ],
 [-3.4656155 ],
 [-3.8815875 ],
 [-3.3012123 ],
 [-2.2433374 ],
 [-3.984924  ],
 [-2.403539  ],
 [-1.6620276 ],
 [-1.2476282 ],
 [ 0.70056343],
 [ 0.02559426],
 [-2.010968  ],
 [-1.9807734 ],
 [-0.54442114],
 [-3.3212075 ],
 [-1.9930742 ],
 [-2.8872292 ],
 [-1.9889274 ],
 [-0.6063874 ],
 [-3.645447  ],
 [-0.90102154],
 [-2.2009826 ],
 [-1.2063124 ],
 [-2.9370475 ],
 [-1.061515  ],
 [-0.8632531 ],
 [-4.2804165 ]]


weighted_group = np.concatenate((ad_hyp_11, ad_hyp_10, ad_hyp_6), axis=1)
print(weighted_group)
import matplotlib.pyplot as plt

# Creating dataset
np.random.seed(10)

fig, ax= plt.subplots()

main=ax.boxplot(weighted_group, patch_artist=True)
m = weighted_group.mean(axis=0)
st = weighted_group.std(axis=0)
ax.set_title('Ad-Hypertension: Q-learning rewards for different States')
for i, line in enumerate(main['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(m[i], st[i])
    ax.annotate(text, xy=(x, y))
ax.set_xticks([1, 2,3])
ax.set_xticklabels(["11 states for samples>50","10 states for samples>100", "6 states for samples>200"], rotation=10)
# ax.set_ylim(-20,10)

colors=["forestgreen", "blue", "dimgray"]
 
for patch, color in zip(main['boxes'],colors):
    patch.set_facecolor(color)

 

plt.savefig('hyp-ad_latest.png')
# show plot
plt.show()
