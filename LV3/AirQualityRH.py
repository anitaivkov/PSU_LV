import urllib.request as ur
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

# postaja=160 (Osijek), polutant=5 (PM10), tipPodatka=0 (dnevni), vrijemeOd i vrijemeDo za 2017. godinu
url = 'http://iszz.azo.hr/iskzl/rs/podatak/export/xml?postaja=160&polutant=5&tipPodatka=0&vrijemeOd=01.01.2017&vrijemeDo=31.12.2017'

airQualityHR = ur.urlopen(url).read() 
root = ET.fromstring(airQualityHR) 

df = pd.DataFrame(columns=('mjerenje', 'vrijeme')) 

i = 0
while True:
    try:
        obj = root[i] 
    except IndexError:
        break

    if len(obj) > 2:
        row = dict(zip(['mjerenje', 'vrijeme'], [obj[0].text, obj[2].text])) 
        row_s = pd.Series(row) 
        row_s.name = i 
        df = pd.concat([df, row_s.to_frame().T], ignore_index=True)
        df.loc[i, 'mjerenje'] = float(df.loc[i, 'mjerenje']) 
    else:
        print(f"Upozorenje: Element na indeksu {i} nema očekivani broj pod-elemenata i bit će preskočen.")

    i = i + 1

df.vrijeme = pd.to_datetime(df.vrijeme, utc=True) 

df.plot(y='mjerenje', x='vrijeme') 
plt.title('Dnevna koncentracija PM10 u Osijeku (2017)')
plt.xlabel('Datum')
plt.ylabel('Koncentracija PM10')
plt.grid(True)
plt.show() 

najvece_koncentracije_pm10 = df.sort_values(by='mjerenje', ascending=False).head(3) 
print("\nTri datuma s najvećom koncentracijom PM10 u 2017. godini za Osijek:")
print(najvece_koncentracije_pm10[['vrijeme', 'mjerenje']])