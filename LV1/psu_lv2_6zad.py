"""6. zadatak"""

ham_rijeci = 0
spam_rijeci = 0
ham_brojac = 0
spam_brojac = 0
spam_usklicnici = 0

#encoding='utf-8' je tu za svaki slučaj
with open('SMSSpamCollection.txt', 'r', encoding='utf-8') as datoteka:
    for redak in datoteka:
        # Razdvajanje oznake (ham/spam) i teksta poruke
        oznaka, poruka = redak.strip().split('\t', 1)

        # Brojanje riječi
        rijeci = poruka.split()
        rijeci_brojac = len(rijeci)

        if oznaka == 'ham':
            ham_rijeci += rijeci_brojac
            ham_brojac += 1
        elif oznaka == 'spam':
            spam_rijeci += rijeci_brojac
            spam_brojac += 1

            # Provjera završava li spam poruka uskličnikom
            if poruka.strip().endswith('!'):
                spam_usklicnici += 1

avg_ham_rijeci = ham_rijeci / ham_brojac if ham_brojac > 0 else 0
avg_spam_rijeci = spam_rijeci / spam_brojac if spam_brojac > 0 else 0

print("Prosjecan broj rijeci u ham porukama: {:.2f}".format(avg_ham_rijeci))
print("Prosjecan broj rijeci u spam porukama: {:.2f}".format(avg_spam_rijeci))
print("Broj spam poruka koje završavaju uskličnikom: ", spam_usklicnici)
