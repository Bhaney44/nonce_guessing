import requests
import numpy as np
import matplotlib.pyplot as plt

class block():

    def __init__(self, data):
        self.idx = data["block_index"]
        self.time = data["time"]
        self.nonce = data["nonce"]
        self.prev = data["prev_block"]
        if len(data["next_block"]) > 0:
            self.next = data["next_block"]

class chain():

    def __init__(self):
        self.blocks = []

    def add_block(self, block_data):
        self.blocks.append(
            block(block_data)
        )

    def build_chain(self, length=100):
        prev = requests.get(f'https://blockchain.info/latestblock').json()["hash"]
        for ii in np.arange(length):
            prev_block = requests.get(f'https://blockchain.info/rawblock/{prev}').json()
            self.add_block(prev_block)
            prev = prev_block["prev_block"]
            print(f"Built: {ii}")

    def plot_nonces(self):
        bb = self.blocks
        idxs = [b.idx for b in bb]
        nonces = [b.nonce for b in bb]
        avg3 = [np.mean([bb[ii-1].nonce, bb[ii].nonce, bb[ii+1].nonce]) for ii in np.arange(1, len(bb)-1)]
        plt.plot(idxs, nonces)
        plt.plot(idxs[1:len(idxs)-1], avg3)
        plt.show()
        
c = chain()
c.build_chain(100)
c.plot_nonces()

xx = 13
sided = int((xx-1)/2)
bb = c.blocks
n = [b.nonce for b in c.blocks]
idx = [b.idx for b in c.blocks]
avgxx = [np.mean(
            [bb[jj].nonce for jj in np.arange(ii-sided, ii+sided+1)]
        ) for ii in np.arange(sided, len(bb)-sided)]
            

plt.plot(idx, n, 'b-')
plt.plot(idx[sided:len(idx)-sided], avgxx, 'r--')
plt.show()
