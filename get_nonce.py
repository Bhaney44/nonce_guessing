import requests, json
import numpy as np
import matplotlib.pyplot as plt

class block():

    def __init__(self, data):
        self.block_index = data['block_index']
        self.hash = data['hash']
        self.time = data['time']
        self.nonce = data['nonce']
        self.prev_block = data['prev_block']
        if data['next_block'] and len(data['next_block']) > 0:
            self.next_block = data['next_block']
        else: self.next_block = None

    def dump(self):
        return {
            'block_index': self.block_index,
            'hash': self.hash,
            'time': self.time,
            'nonce': self.nonce,
            'prev_block': self.prev_block,
            'next_block': self.next_block
        }

class chain():

    def __init__(self):
        self.blocks = []

    def add_block(self, block_data):
        self.blocks = [block(block_data)] + self.blocks

    def build_chain(self, length=100):
        prev = requests.get(f'https://blockchain.info/latestblock').json()['hash']
        for ii in np.arange(length):
            prev_block = requests.get(f'https://blockchain.info/rawblock/{prev}').json()
            self.add_block(prev_block)
            prev = prev_block['prev_block']
            print(f'Built: {ii}')

    def update_chain(self):
        current = self.top_block.hash
        curr_block = requests.get(f'https://blockchain.info/rawblock/{current}').json()
        while True:
            if len(curr_block['next_block']) > 0:
                next_hash = curr_block['next_block'][0]
                next_block = requests.get(f'https://blockchain.info/rawblock/{next_hash}').json()
                self.add_block(next_block)
                curr_block = next_block
            else:
                break

    def get_averages(self):
        bb = self.blocks
        self.avg3 = [np.mean(
                [bb[jj].nonce for jj in np.arange(ii-3, ii)]
            ) for ii in np.arange(3, len(bb))]
        self.avg21 = [np.mean(
                [bb[jj].nonce for jj in np.arange(ii-21, ii)]
            ) for ii in np.arange(21, len(bb))]
        self.avg101 = [np.mean(
                [bb[jj].nonce for jj in np.arange(ii-21, ii)]
            ) for ii in np.arange(101, len(bb))]

    def plot_nonces(self):
        bb = self.blocks
        self.get_averages()
        idxs = [b.block_index for b in bb]
        nonces = [b.nonce for b in bb]
        plt.plot(idxs, nonces)
        plt.plot(idxs[3:len(idxs)], self.avg3)
        plt.plot(idxs[21:len(idxs)], self.avg21)
        plt.plot(idxs[101:len(idxs)], self.avg101)
        plt.show()

    @property
    def top_block(self):
        return self.blocks[len(self.blocks)-1]

    def save(self):
        savechain = [b.dump() for b in self.blocks]
        jobj = json.dumps({'chain': savechain})
        with open("data.json", "w") as outfile:
            outfile.write(jobj)

    def load(self):
        with open('data.json', 'r') as openfile:
            jobj = json.load(openfile)
        for block in jobj['chain']:
            self.add_block(block)


# c = chain()
# c.build_chain(10000)
# c.plot_nonces()
# c.save()

cc = chain()
cc.load()
cc.plot_nonces()