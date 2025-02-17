# Tiatnic statistic

## Opis projektu

Projekt służy do analizy danych dotyczących pasażerów Titanica oraz przewidywania ich szans na przeżycie. 
Wykorzystuje techniki analizy danych, wstępnego przetwarzania, budowania modeli uczenia maszynowego oraz oceny wyników.


## Struktura projektu

````
├── data/               # Dane wejściowe
│   ├── test.csv        # Dane testowe
│   ├── train.csv       # Dane treningowe
├── results/            # Wyniki uczeania
│   ├── predictions.csv # Wyniki predukcji uczenia
│   ├── report.txt      # Raport z uczenia
│   ├── trained_model.pkl  # Model treningowy
├── src/                # Główne moduły projektu
│   ├── __init__.py      # Init
│   ├── data_loader.py  # Ładowanie danych
│   ├── data_analysis.py # Analiza danych
│   ├── data_preprocessing.py # Przetwarzanie danych
│   ├── model_building.py # Tworzenie i trenowanie modelu
│   ├── logger.py       # Konfiguracja loggera
│   ├── config.py       # Konfiguracja projektu
├── tests/              # Testy jednostkowe
│   ├── test_data_loader.py  # Testowanie ładowanie danych
│   ├── test_data_analysis.py # Testowanie analizy danych
│   ├── test_data_preprocessing.py # Testowanie przetwarzania danych
│   ├── test_model_building.py # Testowanie tworzenia i trenowania modelu
│   ├── test_logger.py       # Testowanie loggera
│   ├── test_config.py  # Testowanie configa
├── config.json         # Plik konfiguracyjny
├── main.py             # Główny plik uruchamiający cały proces
├── README.md           # Dokumentacja projektu
├── requirements.txt    # Zależności 
````
## Wykorzytsane biblioteki 

* pandas – przetwarzanie danych
* numpy – operacje numeryczne
* seaborn – wizualizacja danych
* matplotlib – wykresy
* scikit-learn – budowa i ewaluacja modelu ML
* pytest – testowanie
* logging – logowanie zdarzeń
* pickle – serializacja modelu

## Opis funkcji

* **load_data(filepath)** – wczytuje dane z pliku CSV
* **data_analysis(data)** – przeprowadza podstawową analizę danych
* **handle_missing_data(data)** – uzupełnia brakujące wartości w danych
* **extract_features(data)** – upraszcza cechy danych
* **encode_categorical_features(data)** – koduje zmienne kategoryczne
* **scale_numeric_features(data)** – standaryzuje wartości numeryczne
* **split_data(data, target_column)** – dzieli dane na zbiór treningowy i testowy
* **create_pipeline()** – tworzy i zwraca model ML w postaci pipeline’u
* **setup_logger(log_file)** – konfiguruje logger do zapisywania logów

## Instalacja 

Klonowanie repozytorium
````
git clone https://github.com/twoje-repo/titanic-analysis.git
cd titanic-analysis
````

Instalacja zależności

````
pip install -r requirements.txt
````

## Testowanie 

Aby uruchomić testy jednostkowe, wykonaj:
````
pytest tests/
````

## Wyniki i raporty 
* Analiza danych i raporty zapisywane są w pliku report.txt
* Wytrenowany model jest zapisywany w models/trained_model.pkl
* Predykcje dla danych testowych zapisywane są w predictions.csv



