from ast import literal_eval
s = open('/home/patryk/test/net1/parameters.txt', "r").read()
a = (literal_eval(s)[-1][2])
print(a)