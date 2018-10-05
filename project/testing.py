def test(tekst):
	dict_bokstaver_til_tall = {"a":2 ,"b":2 ,"c":2 ,"d":3 ,"e":3 ,"f":3 ,"g":4 ,"h":4 ,\
								"i":4 ,"j":5 ,"k":5 ,"l":5 ,"m":6 ,"n":6 ,"o":6 ,\
								"p":7 ,"q":7 ,"r":7 ,"s":7 ,"t":8 ,"u":8 ,"v":8 ,\
								"w":9 ,"x":9 ,"y":9 ,"z":9," ":" "}
	ny_tekst = ""
	for i in tekst:
		ny_tekst += str(dict_bokstaver_til_tall[i])
	return ny_tekst

print(test("mikael kiste"))


def test_injektiv(tekst):
	dict_bokstaver_til_tall = {"a":2 ,"b":22 ,"c":222 ,"d":3 ,"e":33 ,"f":333 ,"g":4 ,"h":44 ,\
								"i":444 ,"j":5 ,"k":55 ,"l":555 ,"m":6 ,"n":66 ,"o":666 ,\
								"p":7 ,"q":77 ,"r":777 ,"s":7777 ,"t":8 ,"u":88 ,"v":888 ,\
								"w":9 ,"x":99 ,"y":999 ,"z":9999," ":" "}
	ny_tekst = ""
	for i in tekst:
		ny_tekst += str(dict_bokstaver_til_tall[i])
	return ny_tekst

print(test_injektiv("mikael kiste"))

outfile_dict = {"scikit": {2: [],3: [],4:[],5:[]},"manually": {2: [],3: [],4:[],5:[]},"ridge": {2: [],3: [],4:[],5:[]},"lasso": {2: [],3: [],4:[],5:[]}}