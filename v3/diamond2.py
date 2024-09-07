import hashlib
import secrets
import binascii
from binascii import unhexlify, hexlify
import csv
import random

################################
print("##########################################################")
################################
# Version + hashPrevBlock + hashMerkleRoot + Time + Bits + Nonce
# print("##########################################################")
def hex_to_int(hex_value):
    return int(hex_value, 16)

################################
with open('list.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        for i in row:
            def ensure_10_digits(i):
                if len(i) >= 10:
                    i = i
                    print(i)
                else:
                    additional_digits = ''.join(str(random.randint(0, 9)) for _ in range(10 - len(i)))
                    i = i + additional_digits
                    print(i)
            #ensure_10_digits(i)
            Nonce = str(i)
            Version = "20000000"
            hashPrevBlock = "0000000000000000000571509eac4819dae8ff2c320c487cfd612d36db7ffe1a"
            hashMerkleRoot = "e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122b"
            Time = "1672956895"
            Bits = "1708417e"
            # Variables
            #Version = "01000000"
            #hashPrevBlock = "0000000000000000000571509eac4819dae8ff2c320c487cfd612d36db7ffe1a"
            #hashMerkleRoot = "e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122b"
            #Time = "1672956895"
            #Bits = "1708417e"
            #Nonce = "42014695"
            # Check
            header_hex = (Version + hashPrevBlock + hashMerkleRoot + Time + Bits + Nonce)
            def process_header(header_hex):
                if len(header_hex) % 2 == 0:
                    header_bin = unhexlify(header_hex)
                    hash = hashlib.sha256(hashlib.sha256(header_bin).digest()).digest()
                    hexyh = hexlify(hash).decode("utf-8")
                    newBlockHash = hexlify(hash[::-1]).decode("utf-8")
                    print("preBlockHash", hashPrevBlock)
                    print("newBlockHash", newBlockHash)
                    ###################################
                    # Convert the hex hashes to integers
                    prev_hash_int = hex_to_int(hashPrevBlock)
                    print(prev_hash_int)
                    new_hash_int = hex_to_int(newBlockHash)
                    print(new_hash_int)
                    # Compare the hashes and stop if the new block hash is less than or equal to the previous block hash
                    if new_hash_int <= prev_hash_int:
                        print("New block hash is less than or equal to the previous hash. Stopping.")
                        return 
                else:
                    # add a leading zero to make the string an even length
                    header_hex = "0" + header_hex
                    header_bin = unhexlify(header_hex)
                    hash = hashlib.sha256(hashlib.sha256(header_bin).digest()).digest()
                    hexyh = hexlify(hash).decode("utf-8")
                    newBlockHash = hexlify(hash[::-1]).decode("utf-8")
                    prev_hash_int = hex_to_int(hashPrevBlock)
                    print(prev_hash_int)
                    new_hash_int = hex_to_int(newBlockHash)
                    print(new_hash_int)
                    ###################################
                    # Convert the hex hashes to integers
                    prev_hash_int = hex_to_int(hashPrevBlock)
                    new_hash_int = hex_to_int(newBlockHash)
                    # Compare the hashes and stop if the new block hash is less than or equal to the previous block hash
                    if new_hash_int <= prev_hash_int:
                        print("New block hash is less than or equal to the previous hash. Stopping.")
                        return 
            process_header(header_hex)
