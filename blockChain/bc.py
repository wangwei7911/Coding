# coding: utf-8
import datetime
import hashlib

# 区块实体
class Block:
    def __init__(self, previous_block_hass, transaction, timestamp):
        # 前驱块
        self.previous_block_hass = previous_block_hass
        # 交易信息
        self.transaction = transaction
        # 时间戳
        self.timestamp = timestamp
        # 哈希值
        self.hash = self.get_hass()

    # 生成创世区块，创世区块没有前驱块，没有交易信息，时间戳就是创建时的信息
    @staticmethod
    def creat_gensis_block():
        return Block('0', '0', datetime.datetime.now())

    # 获取当前区块的哈希值
    def get_hass(self):
        header_bin = str(self.previous_block_hass) + str(self.transaction) + str(self.timestamp)
        out_hass = hashlib.sha256(header_bin.encode()).hexdigest()

        return out_hass


block_number_need_to_generate = 10
blockchains = [Block.creat_gensis_block()]

for i in range(1, block_number_need_to_generate):
    transaction = "张三给李四了{}个比特币".format(i)
    blockchains.append(Block(blockchains[i - 1].hash, transaction, datetime.datetime.now()))

    print('区块{} 已经创建，交易是{}'.format(i, transaction))
    print('区块的哈希是：', blockchains[i].hash)