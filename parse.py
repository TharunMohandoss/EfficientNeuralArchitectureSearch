import pickle
output_file = open('new.out','r')
lines = output_file.readlines()
output_file.close()
lines = [x.strip() for x in lines]
acc_lines = [x for x in lines if ('Accuracy:  ' in x and not('epoch' in x))]
flop_lines = [x for x in lines if 'flops :' in x]
param_lines = [x for x in lines if 'params :' in x]
rew_lines = [x for x in lines if 'rew :' in x]

lenval = len('Accuracy:  ')
acc_list = [float(x[lenval:]) for x in acc_lines]
lenval = len('flops :  ')
flop_list = [float(x[lenval:]) for x in flop_lines]
lenval = len('params :  ')
params_list = [float(x[lenval:]) for x in param_lines]
lenval = len('rew :  ')
rew_list = [float(x[lenval:]) for x in rew_lines]

fp = open('completed_action_list.pickle','rb')
cal = pickle.load(fp)
fp.close()

data = []
for i in range(len(cal)):
	data.append( (cal[i],acc_list[i],flop_list[i],params_list[i],rew_list[i]))

fp = open('data.pickle','wb')
pickle.dump(data,fp)
fp.close()

acc_list.sort()
print(acc_list)
# print(flop_list)
# print(params_list)
# print(rew_list)