"""2. zadatak"""

try:
    ocjena = float(input("\nUnos ocjene: "))

    if 0.0 <= ocjena <= 1.0:

        #python rjecnik
        kategorije = {
        "A": (0.9, 1.0),
        "B": (0.8, 0.9),
        "C": (0.7, 0.8),
        "D": (0.6, 0.7),
        "F": (0.0, 0.6),
        }

        for kategorija, (dg, gg) in kategorije.items():
            if dg <= ocjena <= gg:
                print(kategorija)
                break
    else:
        print("Greska: ocjena mora biti izmedju 0.0 i 1.0")

except ValueError:
    print("Greska: unesena vrijednost nije broj")
