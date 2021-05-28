clear;clc;

iris_names = ["Setosa", "Versicolor", "Viriginica"];

% Nazwy w zbiorze iris zostaly zastepione przez liczbowe etykiety (1, 2, 3)
% aby latwiej bylo wczytac zbior danych
iris_data = load('Data/iris.data');
wine_data = load('Data/wine.data');
seeders_data = load('Data/seeds_dataset.txt');

% Ustawienie liczby funkcji przynależności
numberOfMembershipFunctions = 5;
% Ustawienie rozmiaru populacji
populationSize = 15;

% Wyobrażam sobie ten skrypt jako mini aplikacje konsolową. Użytkownik po
% wprowadzeniu odpowiedniej cyfry spowoduje wczytanie pożądanych danych
% (oszczędność pamięci).
% W celu uogólnienia mamy zmienną data_store do przechowywania pożądanego
% zbioru danych
data_store = iris_data;
% Pobranie rozmiaru "bazy danych"
data_size = size(data_store);
% Wyznaczenie liczby atrybutów
numberOfAttributes = data_size(2) - 1;
% Wyłuskanie atrybutów wejściowych
dataSet = data_store(:, 1:data_size(2)-1);
% Wyszukanie wartości maksymalnej wśród danych wejściowych
maxValueFromDataset = max(max(dataSet));
% Wyszukanie wartości minimalnej wśród danych wejściowych
minValueFromDataset = min(min(dataSet));
% Wyłuskanie oryginalnych wyjść (klas wyjściowych)
outputs = unique(data_store(:, data_size(2)));
% Pobranie rozmiaru tablicy outputs
numberOfOutputs = length(outputs);
% Wyznaczenie długości wyktora parametrów
numberOfParameters = numberOfAttributes*numberOfOutputs*numberOfMembershipFunctions;
% Zainicjowanie populacji zerami
pop = zeros(numberOfParameters, populationSize);
% Przypisanie wylosowanych liczb do populacji
for i=1:numberOfParameters
   for j=1:populationSize
      pop(i, j) = randInRange(minValueFromDataset, maxValueFromDataset); 
   end
end
pop

% Funkcja do losowania liczby z przekazanego zakresu
% Losowany zakres to [begin - 10% * range, end + 10%*range]
function number = randInRange(beginRange, endRange)
    range = endRange - beginRange;
    number = (range + range * 0.2) * rand() + (beginRange - range * 0.1);
end