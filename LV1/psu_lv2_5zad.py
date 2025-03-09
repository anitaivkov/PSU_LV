"""5. zadatak"""

from collections import Counter
import string

with open('song.txt', 'r') as datoteka:
    sadrzaj = datoteka.read()

#pretvaranje teksta u mala slova i uklanjanje interpunkcije
#str.maketrans: stvara tablicu za zamjenu koja uklanja interpunkcijske znakove
rijeci = sadrzaj.lower().translate(str.maketrans('', '', string.punctuation)).split()

#stvaranje python rječnika s brojem pojavljivanja svake riječi
brojac_rijeci = Counter(rijeci)

# Pronalaženje riječi koje se pojavljuju samo jednom
jednokratne_rijeci = [rijec for rijec, count in brojac_rijeci.items() if count == 1]

# Ispis rezultata
print("Broj rijeci koje se pojavljuju samo jednom: ", len(jednokratne_rijeci))
print("Rijeci koje se pojavljuju samo jednom:")
print(', '.join(jednokratne_rijeci))
