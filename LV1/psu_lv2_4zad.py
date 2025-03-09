"""4. zadatak"""

def pouzdanost_filtra (datoteka):
    ukupno = 0
    brojac = 0

    if datoteka:
        #with open sama otvara i zatvara datoteku, ja ne trebam samostalno zatvarati datoteku
        with open(datoteka, 'r') as file:
            for line in file:
                if line.startswith('X-DSPAM-Confidence:'):
                    try:
                        confidence = float(line.split(':')[1])
                        ukupno += confidence
                        brojac += 1
                    except ValueError:
                        print("Greška pri konverziji vrijednosti u liniji: ", line.strip)
        if brojac > 0:
            prosjek = ukupno / brojac
            return prosjek
        else:
            return None
    else:
        print ("Varijabla s imenom datoteke je prazna.")

datoteka = input("Ime datoteke: ")
prosjecna_pouzdanost = pouzdanost_filtra(datoteka)

if prosjecna_pouzdanost is not None:
    print("Average X-DSPAM-Confidence: ", prosjecna_pouzdanost)
