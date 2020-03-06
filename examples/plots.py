import numpy as np
import pdb
import matplotlib.pyplot as plt

# read_data
# file_name = 

dat1 = np.load('/home/anowak/ssvmdata/ssvm_0.5.npy', allow_pickle=True)
dat2 = np.load('/home/anowak/ssvmdata/ours_0.1.npy', allow_pickle=True)

train_error1, test_error1, train_error_batch1, test_error_batch1 = dat1

train_error2, test_error2, train_error_batch2, test_error_batch2, oracles = dat2

train_error1 = [1 - train_error_batch1[1]] + train_error1
test_error1 = [1 - test_error_batch1[1]] + test_error1
train_error2 = [train_error_batch2[1]] + train_error2
test_error2 = [test_error_batch2[1]] + test_error2

maxlen = min(len(train_error1), len(train_error2))
maxlenbatch = min(len(train_error_batch1), len(train_error_batch2))

print(len(train_error1), len(train_error2))

plt.figure()
pl1, = plt.plot(np.arange(maxlen), train_error1[:maxlen])
pl2, = plt.plot(np.arange(maxlen), train_error2[:maxlen])
handles = [pl1, pl2]
plt.legend(handles, ["SSVM", "Max-Min"], fontsize=15)
plt.xlabel('Passes on data', fontsize=15)
plt.ylabel('training error', fontsize=15)
plt.savefig('/home/anowak/INRIA/pystruct/figures/train.pdf')
plt.show()

plt.figure()
pl1, = plt.plot(np.arange(maxlen), test_error1[:maxlen])
pl2, = plt.plot(np.arange(maxlen), test_error2[:maxlen])
handles = [pl1, pl2]
plt.legend(handles, ["SSVM", "Max-Min"], fontsize=15)
plt.xlabel('Passes on data', fontsize=15)
plt.ylabel('testing error', fontsize=15)
plt.savefig('/home/anowak/INRIA/pystruct/figures/test.pdf')
plt.show()


plt.figure()
pl1, = plt.plot(np.arange(maxlenbatch), 1 - np.array(train_error_batch1[:maxlenbatch]))
pl2, = plt.plot(np.arange(maxlenbatch), train_error_batch2[:maxlenbatch])
handles = [pl1, pl2]
plt.legend(handles, ["SSVM", "Max-Min"], fontsize=15)
plt.xlabel('data points observed / 100', fontsize=15)
plt.ylabel('training error', fontsize=15)
plt.savefig('/home/anowak/INRIA/pystruct/figures/trainbatch.pdf')
plt.show()

plt.figure()
plt.plot(np.arange(maxlenbatch), 1 - np.array(test_error_batch1[:maxlenbatch]))
plt.plot(np.arange(maxlenbatch), test_error_batch2[:maxlenbatch])
handles = [pl1, pl2]
plt.legend(handles, ["SSVM", "Max-Min"], fontsize=15)
plt.xlabel('data points observed / 100', fontsize=15)
plt.ylabel('testing error', fontsize=15)
plt.savefig('/home/anowak/INRIA/pystruct/figures/testbatch.pdf')
plt.show()

plt.figure()
plt.plot(np.arange(len(oracles)), oracles)
plt.xlabel('Passes on data', fontsize=15)
plt.ylabel('max-min oracle dual gap', fontsize=15)
plt.savefig('/home/anowak/INRIA/pystruct/figures/oracle.pdf')
plt.show()
# pdb.set_trace()