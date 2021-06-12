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

% Funkcja znajdujaca przy pomocy algorytmu rojowego strukture fisu.
% Znajduje niekoniecznie optymalna strukture jak to w algorytmach rojowych.
% x - populacja roju, data_matrix - przemieszany wektor danych
% nalezy rozwazyc czy nie powinnismy go przeslac
% przemieszany, bo w przypadku niektorych zbiorow nie mamy pewnosci
% czy pierwsze np. 80% danych zapewni odpowiednia generalizacje, jesli
% nie wymieszamy tych danych
function fis = prepareFisWithPSO(x, data_matrix)
    EXPERIMENT_ATTEMPTS = 10; % z tylu prob wyciagac bedziemy srednia dla danego elementu,
    % moze pozniej dodamy na razie sprobujmy doprowadzic zeby cokolwiek sie
    % liczylo
    
    pop_size = length(x); % rozmiar populacji
    vector_size = length(pop(0)); % rozmiar wektora reprezentujacego jednego agenta / osobnika
    c1 = 2; % stala akceleracji
    c2 = 2; % stala akceleracji

    it = 1; % licznik iteracji
    it_max = 100; % maksymalny nr iteracji

    % wektor predkosci
    v = zeros(1, vector_size); % wektor o dlugosci wiersza danych (60 dla irysow)
    
    % losowo generujemy predkosci
    for i = 1 : pop_size
        v(i, :) = randn(1, pop_size); % tutaj trzeba bedzie pokombinowac zakres losowania
    end

    pbest = ones(pop_size, vector_size); % lokalnie najlepsze czasteczki
    gbest = ones(vector_size); % globalnie najlepsza czasteczka
    for i = 1 : vector_size
        gbest(i) = Inf; % ustawmy pola na nieskonczonosc na poczatku, w sumie nie jestem pewien
        % czy akurat w tym przypadku bedzie dzialac, ale pomysle
    end

    w = calc_constriction_factor(c1, c2);

    while it <= it_max
        it = it + 1;
        % wyznaczamy przystosowanie kazdej czastki - wartosc funkcji w punkcie
        % i czy jest ona mniejsza od poprzedniej najlepszej dla danej czastki
        for i = 1 : pop_size
                % jak i-ta czasteczka jest lepsza w itej iteracji to
                % zamieniamy
                if i == 1 || get_func_val(x(i, :), data_matrix) < get_func_val(pbest(i, :), data_matrix)
                    pbest(i, :) = x(i, :);
                end
        end

        %szukamy rozwiazania globalnie najlepszego
        gbest = get_global_best(pbest, gbest, data_matrix);

        % uaktualniamy predkosc i polozenie dla kazdej czastki
        for i = 1 : pop_size
                U1 = c1 * rand(1);
                U2 = c2 * rand(1);
                v(i, :) = w * (v(i, :) + U1 * (pbest(i) - x(i, :)) + U2 * (gbest - x(i, :)));
                x(i, :) = x(i, :) + v(i, :);            
        end

        fis = get_func_val(gbest, data_matrix);
        % experiment_result_values_sum = experiment_result_values_sum + estimated_best_value;
        % experiment_vectors_sum = experiment_vectors_sum + gbest;
    end

    % average_experiment_value = experiment_result_values_sum / EXPERIMENT_ATTEMPTS;
    % average_experiment_solution = experiment_vectors_sum / EXPERIMENT_ATTEMPTS;
end

function factor = calc_constriction_factor(c1, c2)
    fi = c1 + c2;
    factor = 1. / (2 + (fi * (fi - 4)) ^ 0.5); % tutaj jeszcze sprawdze w labach
    % jak dokladnie sie liczy ten wspolczynnik z reguly
end

function gbest = get_global_best(pbest, previous_best, data_matrix)
    pop_size = length(pbest); % wezmy rozmiar populacji
    gbest = ones(1, pop_size); % tu bedziemy wektor najlepszego rozwiazania przechowywac
    current_best = Inf; % ustawmy wstepnie ze nieskonczonosc to najlepszy blad
    for i = 1 : pop_size
        % jesli ktorys z lokalnych rozwiazan jest lepszy to zmieniamy
        % indeks i current)best
        if get_func_val(pbest(i, :), data_matrix) < current_best
            gbest = pbest(i, :);
            current_best = get_func_val(pbest(i, :), data_matrix);
        end
    end
    
    % czy nowe potencjalne rozwiazanie globalne jest lepsze od starego?
    if current_best > get_func_val(previous_best, data_matrix)
        gbest = previous_best;
    end
end

% wybieramy najlepszy pod katem najmniejszego calkowitego bledu
% bezwzglednego
function value = get_func_val(vect, data_matrix)
    fis = parseFis(vect, data_matrix); % czy parseFis uczy na zbiorze uczacym czy ustawia
    % parametry a ja mam uczyc?
    % 20% np. na zbior testowy
    testing = data_matrix(0.8 * length(data_matrix) + 1:length(data_matrix), :);
    testing_vector_length = length(testing(1,:));
    testing_input = testing(:, 1:testing_vector_length-1);
    testing_output = testing(:, testing_vector_length);
    y = evalfis(fis, testing_input);% wynik uczenia przetestowany na zbiorze testowym
    value = mse(y, testing_output); % jakos trzeba wyznaczyc jakos fisu dla danych parametrow poprzez jakis blad
end
