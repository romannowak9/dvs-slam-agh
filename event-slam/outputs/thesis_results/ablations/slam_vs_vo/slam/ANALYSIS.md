# Częściowa weryfikacja seq007

Przebieg został świadomie przerwany po zapisaniu 2301 poprawnie przetworzonych
ramek (0–2300). Obejmuje około 27,5 s sekwencji. Nie wystąpił żaden nieudany
krok lokalizacji: 2299 póz pochodziło z mapy, jedna z inicjalizacji i jedna z
fallbacku VO. Relokalizacja nie była potrzebna, a IMU nie odrzuciło żadnego
kroku.

## Mapa i graf

- 271 ukończonych keyframe'ów oraz jeden keyframe rozpoczęty podczas przerwania,
- 27 231 landmarków, z czego 23 880 leży nie dalej niż 5 m,
- 271 krawędzi sekwencyjnych i 180 krawędzi pętli,
- 179 pętli zostało w pełni ukończonych przed przerwaniem,
- mediana inlierów pętli: 31,
- mediana błędu reprojekcji pętli: 1,39 px,
- brak widocznych skoków trajektorii w ramkach zamknięcia pętli.

Ostatni, 272. keyframe (frame 2301) i jego krawędź pętli zostały dodane przed
przerwaniem optymalizacji, ale odpowiadająca mu ramka nie trafiła już do
`trajectory.csv`. Jest to oczekiwana pozostałość po bezpiecznym przerwaniu,
nie wynik błędu normalnego przebiegu.

## Dokładność dla wspólnych 275 timestampów

| Wariant | ATE SE(3) [m] | ATE Sim(3) [m] | skala Sim(3) | mediana RVE SE(3) | AUC SE(3) |
|---|---:|---:|---:|---:|---:|
| obecny SLAM | 0,1493 | 0,0406 | 1,0966 | 0,1913 | 0,7560 |
| SLAM etap 2 | 0,1824 | 0,1399 | 1,0724 | 0,1462 | 0,7881 |
| stare VO | 0,4743 | 0,4762 | 0,9894 | 0,1645 | 0,7718 |

Zamknięcie pętli obniżyło ATE SE(3) o około 18% względem etapu 2, a względem
starego VO o około 69%. Po dopuszczeniu skali kształt trajektorii jest znacznie
lepszy niż w obu punktach odniesienia. Jednocześnie graf skurczył trajektorię:
do GT potrzebna jest skala 1,0966. Jest to niepożądane w stereo SLAM-ie i
wyjaśnia część pogorszenia prędkości; pozostałą część powoduje nierównomierna
korekta pozy między keyframe'ami. Parametrów trackera, PnP i IMU nie należy na
tej podstawie zmieniać — zachowują się stabilnie. Przed dalszym strojeniem
warto poprawić sposób dodawania i optymalizacji powtarzających się pętli.

## Czas wykonania

Przetworzenie 2301 ramek trwało około 10 h 38 min. Pierwsze około 1075 ramek
zajęło około 1 h 53 min (średnio 6,3 s/ramkę), a następne 1226 około 8 h 45 min
(średnio 25,7 s/ramkę). Zatem średni koszt kolejnej części przebiegu wyraźnie
rośnie, choć zwykłe ramki nie stają się stopniowo cztery razy wolniejsze.
Najdłuższe postoje przypadają na keyframe'y z zaakceptowaną pętlą.

Przyczyny:

1. Każdy nowy keyframe jest porównywany deskryptorami ze wszystkimi dostatecznie
   starymi keyframe'ami, więc wyszukiwanie kandydatów rośnie wraz z mapą.
2. Po każdej zaakceptowanej pętli optymalizowany jest od nowa cały graf. Dla 272
   keyframe'ów oznacza to 1626 zmiennych, a `least_squares` oblicza gęsty
   numeryczny Jacobian bez informacji o rzadkości grafu.
3. Po optymalizacji aktualizowane są wszystkie keyframe'y, landmarki i wszystkie
   wcześniejsze wyniki ramek oraz przebudowywana jest trajektoria.
4. Pętle są obecnie przyjmowane bardzo często: 179 ukończonych pętli, zwykle w
   kolejnych keyframe'ach tego samego ponownie odwiedzanego obszaru. Większość
   późnych optymalizacji zmniejsza koszt tylko nieznacznie.

Najważniejsze przyszłe poprawki to rzadka struktura Jacobianu lub dedykowany
optymalizator grafu, rzadsza/batchowa optymalizacja, ograniczenie powtarzających
się pętli i odroczenie pełnej przebudowy trajektorii. Nie zostały one wdrożone
w ramach tej weryfikacji.
