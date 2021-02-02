
# This creates a list of words for an online word cloud generator...
# Numbers are the #replicates, which are inversely proportional to word size

temp = {'Irrigation': 1,

        'Hydropedology': 2,
        'Agriculture': 2,
        'Crop modeling': 2,

        'Yield prediction': 3,
        'Water management': 3,
        'Ecosystem services': 3,

        # 'Groundwater recharge': 4,
        # 'Ressource management': 4,
        'Plant productivity': 4,
        'Food security': 4,

        'Sustainability': 5,
        'Drought stress': 5,
        'Early warning': 5,

        'Climate change': 6,
        'Carbon budget': 6,
        # 'Greenhouse gas emission': 6,
        'Weather prediction': 6,
        'Long-term trends': 6,
        'Growth limiting factors': 6,
        'Crop water productivity': 6,

        'Water cycle': 7,
        'Energy-limited': 7,
        'Artificial Intelligence': 7,

        'Smartphone Apps': 8,
        'Remote Sensing': 8,
        'Sensor Technology': 8,
        'Solar-inducded flourescence': 8,
        'Biomass': 8,
        'Soil Moisture': 8,
        'Machine Learning': 8,
        'Modeling': 8,

        'Root Zone': 10,
        'Policy-support': 10,
        'Citizen science': 10,
        'Crowd sourcing': 10,

        }

lim = max([x[1] for x in temp.items()]) + 0

cnt = 0
for key, val in temp.items():
    for i in range(lim-val):
        cnt += 1
        print(key)
print(cnt)
print(len(temp))