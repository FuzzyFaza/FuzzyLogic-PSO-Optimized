clear;clc;

iris_names = ["Setosa", "Versicolor", "Viriginica"];

% Nazwy w zbiorze iris zostaly zastepione przez liczbowe etykiety (1, 2, 3)
% aby latwiej bylo wczytac zbior danych
iris_data = load('Data/iris.data');
wine_data = load('Data/wine.data');
% Przesunięcie numerów klas w zbiorze win na ostatnią pozycję
wineSize = size(wine_data);
wine_data = circshift(wine_data, wineSize(2)-1, 2);
seeders_data = load('Data/seeds_dataset.txt');

% Ustawienie liczby funkcji przynależności
 numberOfMembershipFunctions = 5;
% Ustawienie rozmiaru populacji
populationSize = 15;
% Ilość części na jakie dzielimy zbiór
numberOfFolds = 10;

% Wyobrażam sobie ten skrypt jako mini aplikacje konsolową. Użytkownik po
% wprowadzeniu odpowiedniej cyfry spowoduje wczytanie pożądanych danych
% (oszczędność pamięci).
% W celu uogólnienia mamy zmienną data_store do przechowywania pożądanego
% zbioru danych
data_store = iris_data;
[data_matrix, results, dataSet, numberOfAttributes, maxValueFromDataset, minValueFromDataset, numberOfOutputs] = prepare_folds(data_store, numberOfFolds);
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
customFis = mamfis('Name', 'test', 'NumInputs', numberOfAttributes, 'NumOutputs', 1, 'NumInputMFs', numberOfMembershipFunctions, 'NumOutputMFs', numberOfOutputs);
cross_validation(data_matrix, customFis, results);
%perfect = prepareFisWithPSO(pop, data_matrix, dataSet, customFis);
%firstOut = evalfis(customFis,...);
%perfectFis = genfis();
%secOut = evalfis(perfectFis, ...);


% customFis = mamfis('Name', 'test', 'NumInputs', numberOfAttributes, 'NumOutputs', 1, 'NumInputMFs', numberOfMembershipFunctions, 'NumOutputMFs', numberOfOutputs);
% [in,out] = getTunableSettings(customFis);
% for i=1:length(in)
%     for j=1:numberOfMembershipFunctions 
%        in(i).MembershipFunctions(j).Parameters.Minimum = minValueFromDataset - 0.1 * (maxValueFromDataset-minValueFromDataset);
%        in(i).MembershipFunctions(j).Parameters.Maximum = maxValueFromDataset + 0.1 * (maxValueFromDataset-minValueFromDataset);
%     end
% end
% test = pop(:, 1);
% numOfParams = length(customFis.Input(1).MembershipFunctions(1).Parameters);
% singlePop = zeros(numberOfParameters/numOfParams, numOfParams);
% counter = 1;
% for i=1:numberOfParameters/numOfParams
%     for j=1:numOfParams
%         singlePop(i, j) = test(counter, 1);
%         counter = counter + 1;
%     end
% end
% singlePop = sort(singlePop, 2);
% counter = 1;
% for i=1:length(in)
%     for j=1:numberOfMembershipFunctions 
%        customFis.Input(i).MembershipFunctions(j).Parameters = singlePop(counter, :);
%        counter = counter + 1;
%     end
% end


% Funkcja do losowania liczby z przekazanego zakresu
% Losowany zakres to [begin - 10% * range, end + 10%*range]
function number = randInRange(beginRange, endRange)
    range = endRange - beginRange;
    number = (range + range * 0.2) * rand() + (beginRange - range * 0.1);
end

% Funkcja znajdujaca przy pomocy algorytmu rojowego strukture fisu.
% data_matrix - dane przygotowane do cross-10-validation
% dataSet - atrybuty irysów
function fis = prepareFisWithPSO(pop, data_matrix, dataSet, basicFis)
    EXPERIMENT_ATTEMPTS = 10; % z tylu prob wyciagac bedziemy srednia dla danego elementu,
    % moze pozniej dodamy na razie sprobujmy doprowadzic zeby cokolwiek sie
    % liczylo
    
    [vector_size, pop_size] = size(pop) % vector_size to rozmiar pojedyńczego wektora populacji
    % pop_size to rozmiar całej populacji
    c1 = 2; % stala akceleracji
    c2 = 2; % stala akceleracji

    it = 0; % licznik iteracji
    it_max = 5; % maksymalny nr iteracji

    % wektor predkosci
    v = zeros(vector_size, pop_size); % wektor o dlugosci wiersza danych (60 dla irysow)
    
    % losowo generujemy predkosci
    for i = 1 : vector_size
        for j = 1: pop_size
            v(i, j) = rand(); % zakres losowania [0-1]
        end
    end

    pbest = ones(1, vector_size); % lokalnie najlepsze czasteczki
    gbest = ones(1, vector_size); % globalnie najlepsza czasteczka
%     for i = 1 : vector_size
%         gbest(1, i) = Inf; % ustawmy pola na nieskonczonosc na poczatku, w sumie nie jestem pewien
%         % czy akurat w tym przypadku bedzie dzialac, ale pomysle
%     end

     w = calc_constriction_factor(c1, c2);

    while it <= it_max
        it = it + 1;
        % wyznaczamy przystosowanie kazdej czastki - wartosc funkcji w punkcie
        % i czy jest ona mniejsza od poprzedniej najlepszej dla danej czastki
        for i = 1 : pop_size
                % jak i-ta czasteczka jest lepsza w itej iteracji to
                % zamieniamy
                if i == 1 || get_func_val(pop(:, i), data_matrix, dataSet, basicFis) < get_func_val(pbest, data_matrix, dataSet, basicFis)
                    pbest = pop(:, i);
                end
        end

        %szukamy rozwiazania globalnie najlepszego
        if it == 1
            gbest = pbest;
        elseif get_func_val(pbest, data_matrix, dataSet, basicFis) < get_func_val(gbest, data_matrix, dataSet, basicFis)
            gbest = pbest;
        end
        %gbest = get_global_best(pbest, gbest, data_matrix);
        pbest
%         % uaktualniamy predkosc i polozenie dla kazdej czastki
%         for i = 1 : pop_size
%                 U1 = c1 * rand(1);
%                 U2 = c2 * rand(1);
%                 v(:, i) = w * (v(:, i) + U1 * (pbest - pop(:, i)) + U2 * (gbest - pop(:, i)));
%                 pop(:, i) = pop(:, i) + v(:, i);            
%         end
%         pbest
%         fis = get_func_val(gbest, data_matrix, basicFis);
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
    pop_size = size(pbest); % wezmy rozmiar populacji
    gbest = ones(1, pop_size(2)); % tu bedziemy wektor najlepszego rozwiazania przechowywac
    current_best = Inf; % ustawmy wstepnie ze nieskonczonosc to najlepszy blad
    for i = 1 : pop_size(2)
        % jesli ktorys z lokalnych rozwiazan jest lepszy to zmieniamy
        % indeks i current_best
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
function value = get_func_val(vect, data_matrix, dataSet, basicFis)
    fis = parseFis(vect, dataSet, basicFis); % czy parseFis uczy na zbiorze uczacym czy ustawia
    % parametry a ja mam uczyc?
    % 20% np. na zbior testowy
    %testing = data_matrix(0.8 * length(data_matrix) + 1:length(data_matrix), :);
    %testing_vector_length = length(testing(1,:));
    %testing_input = testing(:, 1:testing_vector_length);
    %testing_output = testing(:, testing_vector_length);
    %out = cross_validation();
    value = cross_validation(data_matrix, fis);
    %y = evalfis(fis, testing_input);% wynik uczenia przetestowany na zbiorze testowym
    %value = mse(y, testing_output); % jakos trzeba wyznaczyc jakos fisu dla danych parametrow poprzez jakis blad
end

function customFis = parseFis(vect, data_matrix, customFis)
    [in,out] = getTunableSettings(customFis);
    numberOfMembershipFunctions = length(in(1).MembershipFunctions);
    maxValueFromDataset = max(max(data_matrix));
    minValueFromDataset = min(min(data_matrix));
    minValue = minValueFromDataset - 0.1 * (maxValueFromDataset-minValueFromDataset);
    maxValue = maxValueFromDataset + 0.1 * (maxValueFromDataset-minValueFromDataset);
    for i=1:length(in)
            customFis.Inputs(i).Range = [minValue maxValue];
    end
    tmpSize = size(data_matrix);
    numberOfParameters = numberOfMembershipFunctions * tmpSize(2) * length(out(1).MembershipFunctions);
    numOfParams = length(customFis.Input(1).MembershipFunctions(1).Parameters);
    singlePop = zeros(numberOfParameters/numOfParams, numOfParams);
    counter = 1;
    for i=1:numberOfParameters/numOfParams
        for j=1:numOfParams
            singlePop(i, j) = vect(counter, 1);
            counter = counter + 1;
        end
    end
    singlePop = sort(singlePop, 2);
    counter = 1;
    for i=1:length(in)
        for j=1:numberOfMembershipFunctions 
            customFis.Input(i).MembershipFunctions(j).Parameters = singlePop(counter, :);
            counter = counter + 1;
        end
    end
    %showrule(customFis, 'Format', 'indexed')
end

function [data_matrix, results, dataSet, numberOfAttributes, maxValueFromDataset, minValueFromDataset, numberOfOutputs] = prepare_folds(data, numOfFolds)
    % Pobranie rozmiaru "bazy danych"
    data_size = size(data);
    % Wyznaczenie długości jednej części
    fold_length = floor(data_size(1) / numOfFolds);
    % Wyznaczenie liczby atrybutów
    numberOfAttributes = data_size(2) - 1;
    % WAŻNE !!!!!!!! (na przykładzie irysów)
    % data_matrix trzyma 10 części (bo 10-cross-validation, ale można u nas ustawić dowolny numOfFolds)
    % po 15 elementów (bo długość zbioru 150/10), gdzie każdy element to
    % wektor zawierający atrybuty (w przypadku irysów 4)
    % data_matrix to zbiór przygotowany do 10-cross-validation (ale to nie
    % jest samo w sobie 10-cross-validation!!!)
    data_matrix = zeros(numOfFolds, fold_length, numberOfAttributes);
    results = zeros(numOfFolds, fold_length);
    % Wyłuskanie atrybutów wejściowych
    dataSet = data(:, 1:data_size(2)-1);
    % Wyszukanie wartości maksymalnej wśród danych wejściowych
    maxValueFromDataset = max(max(dataSet));
    % Wyszukanie wartości minimalnej wśród danych wejściowych
    minValueFromDataset = min(min(dataSet));
    % Wyłuskanie oryginalnych wyjść (klas wyjściowych)
    outputs = unique(data(:, data_size(2)));
    counter = 1;
    for i=1:fold_length
       for j=1:numOfFolds
          data_matrix(j, i, :) = dataSet(counter, :);
          results(j, i) = data(counter, data_size(2));
          counter = counter+1;
       end
    end
    % Pobranie rozmiaru tablicy outputs
    numberOfOutputs = length(outputs);
end

function value = cross_validation(data_matrix, customFis, results)
    data_size = size(data_matrix);
    averages = zeros(data_size(1));
    for i=1:data_size(1)
       testData = data_matrix(i, :, :);
       test_size = size(testData);
       testData = reshape(testData, test_size(2), test_size(3));
       % testowanie przygotowanego zbioru
       out = evalfis(customFis, testData);
       %badanie dokładności i zapisanie w tablicy do policzenia wartości średniej
       % badanie odbywa się przez porównanie wartości wyjściowej z
       % wartością oczekiwaną (nie rozróżniam na tym etapie klas)
       correct = 0;
       for j=1:data_size(2)
            if out(i, j) == results(i, j)
                correct = correct + 1;
            end
       end
       average = correct / data_size(2);
       averages(i) = average;
    end
    value = sum(averages) / length(averages);
end