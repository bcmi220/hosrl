import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font',family='Times New Roman')

plt.figure(figsize=(20, 8), dpi=100)

x_name = ['w/o BERT CPU', 'w/ BERT CPU','w/o BERT GPU','w/ BERT GPU']
first_day = [282.2,	45.0, 436.2, 179.0]
first_weekend = [206.8,	42.5, 369.0, 165.4]

x = range(len(x_name))

plt.bar(x, first_day, width=0.2, label='baseline')
plt.bar([i + 0.2 for i in x], first_weekend, width=0.2, label='+HO')

plt.xticks([i + 0.1 for i in x], x_name)

plt.legend(loc='upper right')


plt.ylabel('Sent./Sec.')

plt.show()