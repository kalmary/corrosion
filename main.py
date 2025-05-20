from AnalizingTools import CorrosionData, Analyzer

def loadRawData(path):
        column_names = ['Unix Time (s)', 'Test Time (h)', 'Air Temp (C)', 'RH (%)',
                              'Surface Temp (C)', 'Cond Lo Freq (S)', 'Cond Hi Freq (S)',
                              'Galv Corr 1 (A)', 'Galv Corr 2 (A)', 'Tot Cond Lo Freq (C/V)',
                              'Tot Cond Hi Freq (C/V)', 'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)']


        data_obj = CorrosionData(path=path, column_names=column_names)
        return data_obj



def main():
        path = 'Acuity_LS_00833_20250226_102627.csv'
        """
        Wczytanie danych jako DataFrame i stworzenie z nich obiektu data_obj. 
        data_obj jest jak data_frame (zbiór danych), ale z kilkoma dopisanymi metodami (funckjami). 
        Metody, których warto używać podczas analiz mają postać data_obj.funkcja(argumenty).
        Metody o postaci data_obj._funkcja(argumenty) odpowiadają za wewnętrzne działanie kodu. Nie trzeba ich ruszać.
        
        Da się "dostać" do danych zawartych w data_obj:
        data = data_obj.data <- jako data zadeklarujemy DataFrame z funkcjonalnością taką jak w bibliotece pandas
        
        alternatywnie te same efekty można uzyskać poprzez:
        data_obj.data.wybrana_metoda_z_pandas
        """
        data_obj = loadRawData(path) #
        """
        data_obj.add_NewColumn() dodaje nową kolumnę do datasetu. Dodanie kolumn czasowych będzie wymagało modyfikacji kodu!!!!
        argumenty:
        column2apply: Kolumna będąca argumentem do wzoru w func
        new_column_name: Nazwa nowej kolumny
        func: funkcja na podstawie której powstaje nowa kolumna
                Funkcja ma konstrukcję lambda (oznaczenie funkcji) x (argument na której działa funkcja): wzór funkcji(x)
        """
        data_obj.add_NewColumn(column2apply='Galv Corr 1 (A)',
                               new_column_name='Galv Corr Mass Loss Rate (g/m-a)',
                               func=lambda x: x * data_obj.coeffs['galvanic_corrosion'])

        data_obj.add_NewColumn(column2apply='Tot Galv Corr 1 (C)',
                               new_column_name='Tot Galv Corr Mass Loss Rate (g/m-a)',
                               func=lambda x: x * data_obj.coeffs['galvanic_corrosion'])


        print('--- COLUMN NAMES ---\n') # wypisanie nazw kolumn
        for i, name in enumerate(data_obj.column_names):
                print(f'index: {i}, name: {name}')

        """
        metoda data_obj.Compute_1dAverages() zwraca nowy data frame, 
        który zawiera dane uśrednione: powstaje jeden dzień średni dla dni zawartych we wszystkich rekordach
        """
        data_avg = data_obj.Compute_DailyAverages()
        """
        metoda data_obj.Compute_1dAverages() zwraca nowy data frame, 
        który zawiera dane uśrednione: każdy dzień jest sprowadzony do jednego punktu pomiarowego - średniej z tego dnia
        """
        data_avg = data_obj.Compute_DailyAverages() # each day averaged to a single data point


        """
        powyższe metody zwracają nowy DataFrame (nową "tabelkę z danymi")
        Żeby dalej je analizować z wykorzystaniem CorrosionData trzeba stworzyć nowy obiekt
        """
        data_obj_avg = CorrosionData(data=data_avg)
        """
        Da się stworzyć DataFrame tylko z wybranymi kolumnami żeby pracowało się z nimi wygodniej:
        data_obj_avg.select_byColumnNames zwróci DataFrame z wybranymi nazwami kolumn.
        Metoda przyjmuje listę nazw kolumn: list[str]
        """
        # data_selected = data_obj_avg.select_byColumnNames(['Air Temp (C)', 'RH (%)', 'Surface Temp (C)',
        #                                                    'Cond Lo Freq (S)', 'Cond Hi Freq (S)',
        #                                                    'Galv Corr 1 (A)', 'Galv Corr 2 (A)',
        #                                                    'Tot Cond Lo Freq (C/V)', 'Tot Cond Hi Freq (C/V)',
        #                                                    'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)',
        #                                                    'Galv Corr Mass Loss Rate (g/m-a)', 'Free Corr Mass Loss Rate (g/m-a)',
        #                                                    'Tot Galv Corr Mass Loss Rate (g/m-a)', 'Tot Free Corr Mass Loss Rate (g/m-a)2'])


        """
        data_obj_avg.plot_parameters() służy do tworzenia wykresów. Wybieramy argument osi x i listę kolumn na osi y 
        (jeszcze nie da się ustawić kilku osi z oddzielnym skalowaniem :c)
        trend_line = True: dodatkowo wyrysuje linię trendu (aproksymację wielomianową)
        sort_x = False: czasami rekordy na osi x mogą zmieniać się w sposób niemonotoniczny i wykres może nie wyjść. 
        sort_x sortuje wartości kolumn i umożliwia sprawdzenie trendu w zależności argumentów.
        """
        data_obj_avg.plot_parameters(x_param_name = 'Air Temp (C)', y_params=['Surface Temp (C)'],
                                     trend_line=True, sort_x=True)

        """
        Analyzer obiekt do alternatywnych analiz danych. Do inicjalizacji jest potrzebny DataFrame
        Póki co jedyną metodą jest .find_correlations
        """
        analyzer_obj = Analyzer(data_obj_avg.select_byColumnNames(['Air Temp (C)', 'RH (%)', 'Surface Temp (C)',
                                                                   'Tot Cond Lo Freq (C/V)', 'Tot Cond Hi Freq (C/V)',
                                                                   'Tot Galv Corr 1 (C)', 'Tot Galv Corr 2 (C)',
                                                                   'Galv Corr Mass Loss Rate (g/m-a)',
                                                                   'Tot Galv Corr Mass Loss Rate (g/m-a)']))

        """
        find_correlations umożliwia wyrysowanie macierzy korelacji dla wybranych (zawartych w dataframe kolumn)
        dodatkowo metoda zwraca tekstową formę macierzy
        """
        cor = analyzer_obj.find_correlations()



if __name__ == '__main__':
        main() # główna część kodu patrz tutaj