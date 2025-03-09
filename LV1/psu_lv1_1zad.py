"""1. zadatak"""

def total_euro(sati, satnica):
    return sati * satnica

radni_sati = float(input("\nRadni sati: "))
satnica = float(input("eura/h: "))

zarada = total_euro(radni_sati, satnica)
print("Ukupno:", zarada, " eura")
