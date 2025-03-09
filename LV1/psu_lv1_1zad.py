"""1. zadatak"""

def total_euro(sati, satnica):
    return sati * satnica

print(" ")
radni_sati = float(input("Radni sati: "))
satnica = float(input("eura/h: "))

zarada = total_euro(radni_sati, satnica)
print("Ukupno:", zarada, " eura")
