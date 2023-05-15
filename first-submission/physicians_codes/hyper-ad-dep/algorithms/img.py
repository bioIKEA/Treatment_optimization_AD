import numpy as np

jr_list=  [[-1.3582433462142944, -0.8955047130584717], [0.03467046469449997, -1.7545448541641235], [-0.9926657676696777, -0.9251022338867188], [-1.5259599685668945, -1.5910300016403198], [0.7103135585784912, 1.3568414449691772], [-0.1398329883813858, -6.980891227722168], [-2.093017578125, -1.635175347328186], [-0.13378044962882996, -1.3568778038024902], [-2.6637473106384277, -2.2725918292999268], [0.649219274520874, 0.21748921275138855], [-1.2798744440078735, -1.0736565589904785], [-0.10141909122467041, -2.854623556137085], [-0.3601854145526886, -5.246603965759277], [-0.22252681851387024, -0.7374071478843689], [-2.827423334121704, -3.278843879699707], [-1.7085832357406616, -1.819715976715088], [0.13876581192016602, 0.3577097952365875], [-2.546306610107422, -0.9666625261306763], [-2.8791956901550293, -2.4919631481170654], [-1.101385235786438, -1.6826447248458862], [-0.8794393539428711, -0.7557589411735535], [-1.2650816440582275, -2.27895188331604], [-1.1439619064331055, -1.4024320840835571], [-2.654836416244507, -4.437993049621582], [-0.5379539132118225, -2.1929938793182373], [-4.2523980140686035, -4.071230411529541], [-1.6483622789382935, -1.219759464263916], [-0.23086334764957428, -5.922744274139404], [-2.254155158996582, -1.0307036638259888], [-5.806319236755371, -3.9439549446105957], [-0.9437433481216431, -1.121416449546814], [-1.039363980293274, -0.6506485342979431], [-0.6090960502624512, -5.809161186218262], [-3.313253402709961, -3.737748622894287], [-3.5883193016052246, -2.486091375350952], [-0.8828745484352112, -2.837496042251587], [-0.8214953541755676, -2.4386539459228516], [-2.0055856704711914, -4.392728805541992], [-2.1827123165130615, -0.853058397769928], [-0.05783804878592491, -0.5365883111953735], [-1.3547037839889526, -0.5013425946235657], [-1.9586317539215088, -1.5773776769638062], [0.14841075241565704, -1.1372467279434204], [-4.195932865142822, -3.3423306941986084], [-0.06168817728757858, -0.6304159760475159], [-0.2111714631319046, -2.0915451049804688], [0.20625293254852295, -1.1869313716888428], [-1.2528460025787354, -2.381727695465088], [-0.029814433306455612, -0.8552641868591309], [-2.870807647705078, -0.4112371504306793], [1.0001741647720337, -0.7265523076057434], [-0.27588951587677, -0.3278810679912567], [0.12247941642999649, -3.4002561569213867], [-2.4645864963531494, -0.9732716083526611], [-3.5036377906799316, -2.004112958908081], [-0.0883667841553688, -2.397963762283325], [-2.3878068923950195, -2.461724042892456], [-0.5975150465965271, 0.06223701313138008], [-1.4066064357757568, -3.6851770877838135], [-1.4321790933609009, -4.892117977142334], [-0.39773812890052795, 0.636326253414154], [-0.41579023003578186, -2.9091384410858154], [-2.4178307056427, -3.423483371734619], [-2.842881441116333, -6.741327285766602], [-0.7598391771316528, -3.7954652309417725], [-3.6938300132751465, -3.8109564781188965], [-0.14866364002227783, -2.9459962844848633], [-0.9986057877540588, 0.6418024301528931], [0.8712416887283325, 1.2820987701416016], [-3.8771936893463135, -2.846703052520752], [-2.04154896736145, -2.230835437774658], [-0.5559362173080444, -1.42623770236969], [-2.4663922786712646, -2.935788869857788], [-2.69219970703125, -4.269822597503662], [-3.0405211448669434, -2.2715859413146973], [-0.5230892300605774, -0.7234209179878235], [-1.2939324378967285, -3.7532267570495605], [-0.11511237174272537, -2.5624330043792725], [-3.3829991817474365, 0.7390915751457214], [-2.1899781227111816, -2.3621439933776855], [-2.4384818077087402, -2.59751033782959], [-0.9348821043968201, 0.17356464266777039], [-2.3189334869384766, -1.3949358463287354], [-0.9418359398841858, -4.632557392120361], [-0.27838534116744995, -5.810412883758545], [0.9590367078781128, 1.5471357107162476], [-3.4163215160369873, -3.7976527214050293], [-1.0596942901611328, -1.262991189956665], [-0.8059723973274231, -3.3706130981445312], [-1.0189279317855835, -3.2320618629455566], [1.0018017292022705, -0.4646776616573334], [-3.748033285140991, -3.6775741577148438], [0.17923690378665924, -0.6164196133613586], [-2.2075388431549072, -4.553851127624512], [-3.0662968158721924, -4.978537082672119], [-1.0049433708190918, -1.7569913864135742], [-3.7806572914123535, -6.715665340423584], [-2.710033655166626, -4.512689590454102], [-4.748570919036865, -7.328916549682617], [1.6716479063034058, 2.5990805625915527]]


sr_list=  [[-1.4256423711776733, -0.9492397904396057], [-1.0515061616897583, -1.0795382261276245], [0.9881845712661743, -0.8211866021156311], [1.9092715978622437, 1.5274155139923096], [0.42946216464042664, -0.7352274656295776], [0.3107520341873169, -0.7779585719108582], [-0.5542882084846497, 1.0737745761871338], [0.4227360785007477, -1.8595774173736572], [0.19274841248989105, -0.4056939482688904], [-1.0561305284500122, -0.07264716178178787], [0.11607255041599274, 0.10857865959405899], [-0.5344319343566895, -0.5344316363334656], [0.27689164876937866, 0.27613186836242676], [-0.1792614609003067, -0.7729642987251282], [-0.27649298310279846, -0.07428587228059769], [-1.7654504776000977, -0.9432055354118347], [-1.2972098588943481, -0.8205084800720215], [-0.12636034190654755, -1.1540850400924683], [-1.255663514137268, -1.129576563835144], [-1.927160620689392, -1.927160620689392], [2.5720930099487305, -0.3889084458351135], [-2.334730386734009, -1.7845882177352905], [0.34498873353004456, -0.9129366874694824], [-0.07171662151813507, -1.3710616827011108], [0.07484374195337296, -0.4489462375640869], [-0.09497450292110443, -1.1508901119232178], [-0.13015885651111603, 0.47465091943740845], [1.7142564058303833, 0.14123167097568512], [0.2532906234264374, 0.2532906234264374], [-1.7107888460159302, -0.8872307538986206], [0.7602569460868835, -0.6097163558006287], [-2.218748092651367, -0.9248225092887878], [-1.5433260202407837, -1.5433260202407837], [-0.07395878434181213, -0.7045522332191467], [-0.8784264326095581, -0.9075392484664917], [-0.981501042842865, -0.9815017580986023], [-0.5213693380355835, -0.35830003023147583], [-1.015624761581421, -1.063537359237671], [-0.09968294203281403, -0.23336850106716156], [-1.2237553596496582, -1.208387851715088], [-1.870395541191101, 0.7860145568847656], [-1.0960936546325684, -1.3860846757888794], [-0.9359263777732849, -2.024657726287842], [-0.12032680213451385, -0.06574435532093048], [-2.315586566925049, -2.1975369453430176], [-1.5528591871261597, -1.558764934539795], [-0.3730884790420532, -0.3730884790420532], [-1.255096435546875, -0.7775254845619202], [-2.2047512531280518, -2.746946096420288], [0.2869529724121094, 0.01701369695365429], [0.31335732340812683, 0.0441778339445591], [-2.3941028118133545, -1.4276500940322876], [-0.7005013227462769, -0.6928436756134033], [-2.1728944778442383, -2.1728944778442383], [-1.6252890825271606, -0.8316666483879089], [0.16999493539333344, -1.4682790040969849], [-0.23710878193378448, -0.6730931401252747], [-1.3935706615447998, -2.2677557468414307], [0.9160034656524658, 0.02846859022974968], [0.11698157340288162, 0.12393978238105774], [-1.2809141874313354, -1.3814740180969238], [0.2675687074661255, -1.0174777507781982], [-1.4544191360473633, -0.9253054857254028], [-1.8185927867889404, -1.0243961811065674], [-0.6417921781539917, -0.6429846286773682], [0.3501293957233429, -1.5669358968734741], [-2.6365251541137695, -0.40395236015319824], [0.49626070261001587, -0.5983548164367676], [-0.2842634916305542, -0.2842642366886139], [0.01857680268585682, -0.2175655961036682], [-1.2294888496398926, -0.40249255299568176], [-0.22647491097450256, 0.2995971441268921], [-1.05351984500885, -1.0594196319580078], [0.23817363381385803, -1.1813572645187378], [-1.045777678489685, -1.0312998294830322], [-0.08461582660675049, -0.0850653126835823], [0.568845808506012, 1.1705594062805176], [-0.020009268075227737, -0.28918877243995667], [0.5457158088684082, -1.5355607271194458], [-0.6174409985542297, -0.6266493797302246], [-0.8796018362045288, -0.8811375498771667], [-1.2871958017349243, 0.49916768074035645], [-1.6996307373046875, -1.173541784286499], [-0.9831066131591797, -0.9831066131591797], [0.7589607238769531, -0.005875219125300646], [-0.8393558263778687, -0.8393558263778687], [-1.4514544010162354, -1.4510756731033325], [-0.927747368812561, -0.8733493089675903], [-0.4362187087535858, -0.4362187087535858], [-0.510474443435669, -0.6522813439369202], [1.2809652090072632, 0.2205078899860382], [0.5291581153869629, -1.2211424112319946], [-1.2224280834197998, -1.2208834886550903], [-1.2342437505722046, -0.013391450047492981], [-0.01876871846616268, -0.33268705010414124], [-1.3680238723754883, -0.3120890259742737], [-0.9837539792060852, -0.4552466869354248], [-1.3206616640090942, -0.8509990572929382], [-2.6126301288604736, -2.606092929840088], [-0.7390676140785217, -0.5639714002609253]]


main_list=  [[-2.530221700668335, -1.0884002447128296], [-2.3969902992248535, -1.7067310810089111], [-3.806079387664795, 0.010087656788527966], [-2.7128186225891113, -4.707451343536377], [-3.6911966800689697, -1.495408535003662], [0.15369239449501038, 0.8738635778427124], [-0.6958496570587158, -0.6958496570587158], [-1.7614716291427612, 0.31485846638679504], [0.15588659048080444, -1.5478421449661255], [-0.2533102035522461, -0.2541378140449524], [-1.4020683765411377, -2.4615750312805176], [-1.8843061923980713, -2.6199028491973877], [-4.201834678649902, -5.783059120178223], [-0.08926410228013992, -0.10063477605581284], [-1.6483827829360962, -1.6486247777938843], [-0.2940083146095276, -0.30748113989830017], [-1.6295897960662842, 0.4297259449958801], [0.3007291257381439, -0.7061454057693481], [-1.9386451244354248, -3.996051788330078], [-3.0401268005371094, -1.2985352277755737], [-0.9083371162414551, -1.3333343267440796], [-0.705764651298523, -1.6229283809661865], [-1.393923282623291, -5.571409702301025], [-0.2940995395183563, -0.2691124379634857], [-4.9905290603637695, -4.656851291656494], [-4.801359176635742, -3.00224232673645], [-2.1860039234161377, -0.9373666644096375], [-0.26039835810661316, -0.2649366855621338], [-0.5050519108772278, -0.41353175044059753], [-1.5804601907730103, -0.015364372171461582], [-3.4355645179748535, -3.627875328063965], [-1.0497373342514038, -2.2878031730651855], [-1.6338812112808228, -2.502152442932129], [-1.758690595626831, -2.2896573543548584], [0.7427219152450562, 1.178411602973938], [-1.3847798109054565, -1.3731296062469482], [0.21639879047870636, -2.6274917125701904], [-1.1350404024124146, -5.410776138305664], [-1.2551900148391724, -1.2717102766036987], [-1.6463409662246704, -0.839552104473114], [-0.5253642201423645, -2.6536083221435547], [-3.503493547439575, -3.735089063644409], [-3.4667184352874756, -2.1929476261138916], [0.602573037147522, 0.8879709243774414], [-2.583209991455078, -0.8663126230239868], [-1.4281367063522339, -2.0625417232513428], [-1.41396963596344, -2.010684013366699], [-1.1666589975357056, 0.5791561603546143], [0.32321253418922424, -1.3959972858428955], [-2.252495765686035, -1.5003355741500854], [-0.9895250797271729, -1.3028287887573242], [-2.6277685165405273, -4.865662574768066], [-0.7599954009056091, 0.9590189456939697], [0.3833705186843872, 0.3833705186843872], [-1.9938278198242188, -0.716301441192627], [-3.6421568393707275, -1.5309011936187744], [-0.8170180320739746, -0.4209747016429901], [1.8659147024154663, -0.03136000037193298], [-1.2678245306015015, -0.45165276527404785], [-2.3625071048736572, -2.3625071048736572], [-3.4182944297790527, -5.181323528289795], [-1.009042501449585, -0.7598571181297302], [-1.7402015924453735, -1.4847512245178223], [-0.9587865471839905, -1.0408861637115479], [0.4790814518928528, 0.6529686450958252], [-1.676706314086914, -1.2402743101119995], [0.31487253308296204, -0.01823827438056469], [-1.4402748346328735, 0.3071199655532837], [0.17192943394184113, -0.9423949718475342], [-1.6600227355957031, -1.358822226524353], [-1.3997036218643188, -0.7417927384376526], [-0.11338420957326889, -2.0385422706604004], [-1.9934505224227905, -3.4390852451324463], [-2.002408742904663, -3.243131637573242], [-0.2963710427284241, 0.21551480889320374], [-0.08550738543272018, -2.874868154525757], [-1.6900925636291504, -1.7238578796386719], [-3.949207067489624, -4.5350494384765625], [-2.25616455078125, -1.284942626953125], [-4.987493515014648, -3.7995970249176025], [-2.934460163116455, -2.935255289077759], [-3.515387773513794, -1.8209394216537476], [-0.8087267875671387, -0.9085788726806641], [-0.1003989428281784, -0.1174139603972435], [-2.013655662536621, -2.9357001781463623], [-4.306276798248291, -3.7811081409454346], [-1.3093726634979248, -5.656035423278809], [-2.968139171600342, -2.859992027282715], [0.2764478623867035, 0.7598217725753784], [-1.5945918560028076, -1.8775500059127808], [0.33128613233566284, 0.3192885220050812], [0.792768657207489, -0.2800976037979126], [-2.5489556789398193, -2.5492589473724365], [-2.231559991836548, -2.227301836013794], [-2.306715965270996, -0.6240829825401306], [-3.6582887172698975, -1.5317405462265015], [-0.23353520035743713, 0.013678300194442272], [-0.33071964979171753, -3.2162094116210938], [-0.45157569646835327, -4.682487487792969], [-1.4645665884017944, -1.4645665884017944]]

# cli_main=np.array(main)[:,1]

weighted_group=np.concatenate((sr_list,jr_list, main_list), axis=1)
print("ad-hyp-dep=",weighted_group.tolist())


import matplotlib.pyplot as plt

 
# Creating dataset
np.random.seed(10)
 
fig, ax= plt.subplots()

main=ax.boxplot(weighted_group, patch_artist=True)
m = weighted_group.mean(axis=0)
st = weighted_group.std(axis=0)
ax.set_title('Hypertension Depression AD Data')
for i, line in enumerate(main['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(m[i], st[i])
    ax.annotate(text, xy=(x, y))
ax.set_xticks([1, 2,3,4, 5,6])
ax.set_xticklabels(["Sr Q-learning","Sr Clinician", "Jr Q-learning", "Jr Clinician",'Full Q-learning', 'Full Clinician'], rotation=10)
ax.set_ylim(-20,10)

colors=["forestgreen", "blue", "dimgray",  "slategray", "tomato", "yellow"]
 
for patch, color in zip(main['boxes'],colors):
    patch.set_facecolor(color)


plt.savefig('HyperADDep_SR_JR.png')
# show plot
plt.show()
