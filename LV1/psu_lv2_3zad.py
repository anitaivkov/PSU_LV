"""3. zadatak"""

brojevi = []

while True:
    unos = input("Unesite broj ili Done za zavrsetak: ")

    if unos.lower() == "done":
        break

    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print ("Error: unos nije broj, pokusajte ponovno")

    #provjera je li lista prazna
if brojevi:
    brojac = len(brojevi)
    srednja = sum(brojevi) / brojac
    mini = min(brojevi)
    maksi = max(brojevi)

    brojevi.sort()

    print("Sortirana lista: ", brojevi)
    print("Broj brojeva: ", brojac)
    print("Srednja vrijednost: ", srednja)
    print("Minimalna vrijednost: ", mini)
    print("Maksimalna vrijednost: ", maksi)
else:
    print ("Lista je prazna, unesite broj")
