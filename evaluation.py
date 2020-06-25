from Levenshtein import distance 

name_to_character = {'א': "Alef",'ב': "Bet",'ג': "Gimel",'ד': "Dalet", 'ה':"He", "ו":"Waw","ז":"Zayin","ח":"Het","ט":"Tet","י":"Yod","כ":"Kaf","ך":"Kaf-final","ל":"Lamed","מ":"Mem-medial","ם":"Mem","נ":"Nun-medial","ן":"Nun-final","ס":"Samekh","ע":"Ayin","פ":"Pe","ף":"Pe-final","צ":"Tsadi-medial","ץ":"Tsadi-final","ק":"Qof","ר":"Resh","ש":"Shin","ת":"Taw", " ":" "}

with open("25-Fg001_characters.txt", "r") as file:
	x = [l.rstrip("\n") for l in file]

f=open("25-Fg001_characters.txt",'r').read()

def create_dummy_trial(x):
	# this creates a trial output
	texts=[]
	for sent in x:
		text=[]
		for ch in sent:
			if ch in name_to_character.keys():
				text.append(name_to_character[ch])
		texts.append(text)
	return texts

def evaluate_output(output, golden):
	texts=''
	for sent in output:
		text=''
		for ch in sent:
			for character, name in name_to_character.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
				if name == ch:
					text += str(character)
		texts+=(text+ '\n')
	print(distance(texts, golden))

dummy_output = create_dummy_trial(x)

evaluate_output(dummy_output, f)

