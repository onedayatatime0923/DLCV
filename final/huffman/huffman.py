import heapq
import pickle

"""
Code for Huffman Coding, compression and decompression. 
Explanation at http://bhrigu.me/blog/2017/01/17/huffman-coding-python-implementation/
"""

class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.freq < other.freq
    def __eq__(self, other):
        if(other == None):
                return False
        if(not isinstance(other, HeapNode)):
                return False
        return self.freq == other.freq


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.index2code = {}
        self.code2index= {}
    # functions for compression:
    def make_frequency_dict(self, layer_list):
        frequency = {}
        text = []
        for l in layer_list:
            text.extend(l)
            text.append('end')

        for character in text:
            if not character in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency
    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)
    def merge_nodes(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)
    def make_codes_helper(self, root, current_code):
        if(root == None):
            return

        if(root.char != None):
            self.index2code[root.char] = current_code
            self.code2index[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")
    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)
    def get_encoded_text(self, layer_list):
        encoded_text = ""
        text = []
        for l in layer_list:
            text.extend(l)
            text.append('end')
        for character in text:
            encoded_text += self.index2code[character] 
        return encoded_text
    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
                encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text
    def get_byte_array(self, padded_encoded_text):
        if(len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return b
    def compress(self, layer_list, weight_path, code_path):
        with open(weight_path, 'wb') as output:
            frequency = self.make_frequency_dict( layer_list)
            self.make_heap(frequency)
            self.merge_nodes()
            self.make_codes()

            encoded_text = self.get_encoded_text(layer_list)
            padded_encoded_text = self.pad_encoded_text(encoded_text)

            b = self.get_byte_array(padded_encoded_text)
            output.write(bytes(b))
        self.save(code_path)
        print("Compressed")
    """ functions for decompression: """
    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:] 
        encoded_text = padded_encoded_text[:-1*extra_padding]

        return encoded_text
    def decode_text(self, encoded_text):
        current_code = ""
        decoded_list = [[]]

        for bit in encoded_text:
            current_code += bit
            if(current_code in self.code2index):
                character = self.code2index[current_code]
                if (character) == 'end':
                    decoded_list.append([])
                else: 
                    decoded_list[-1].append(character)
                current_code = ""
        return decoded_list[:-1]
    def decompress(self, input_path, code_path):
        self.load(code_path)
        with open(input_path, 'rb') as file:
            bit_string = ""

            byte = file.read(1)
            while(len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)

            decompressed_list = self.decode_text(encoded_text)

        print("Decompressed")
        return decompressed_list
    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.code2index,f)
    def load(self,path):
        with open(path, 'rb') as f:
            self.code2index= pickle.load(f)

if __name__ == '__main__':
    layer_list = [[123,5,2,4,5,5,],[3,43,12,211,3,2],
                [123,5,2,4,5,5,],[3,43,12,211,3,2]]
    huffman = HuffmanCoding()
    huffman.compress(layer_list, 'test.pt', 'code.pt')
    print(huffman.decompress('test.pt', 'code.pt'))


