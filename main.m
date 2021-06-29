clear;clc;warning('off');
iris_names = ["Setosa", "Versicolor", "Viriginica"];

% Nazwy w zbiorze iris zostaly zastepione przez liczbowe etykiety (1, 2, 3)
% aby latwiej bylo wczytac zbior danych

choice = input("Proszę wybrać zbiór danych, na którym będziemy operować:\n 1. Irysy 2. Wina 3. Zbiór ziaren - seeds 4. Haberman 5. Teaching Assistants\n");
 
data_store = [];

switch choice
    case 1
        data_store = load('Data/iris.data');
    case 2
        % Przesunięcie numerów klas w zbiorze win na ostatnią pozycję
        wine_data = load('Data/wine.data');
        wineSize = size(wine_data);
        wine_data = circshift(wine_data, wineSize(2)-1, 2);
        data_store = wine_data;
    case 3
        data_store = load('Data/seeds_dataset.txt');
    case 4
        data_store = load('Data/haberman.data');
    case 5
        data_store = load('Data/tae.data');
    otherwise
        printf("Nie wybrano poprawnie zbioru, wiec domyslnie zostanie wybrany zbior irysow. \n")
end

% Ustawienie rozmiaru populacji
populationSize = 15;
% Ilość części na jakie dzielimy zbiór
numberOfFolds = 10;

[data_matrix, results, dataSet, numberOfAttributes, maxValueFromDataset, minValueFromDataset, numberOfClasses] = prepare_folds(data_store, numberOfFolds);
customFis = mamfis('Name', 'test', 'NumInputs', numberOfAttributes, 'NumOutputs', 1, 'NumInputMFs', numberOfClasses, 'NumOutputMFs', numberOfClasses);
customFis = prepareFisRules(dataSet, customFis);
pop = generatePopulation(customFis, numberOfAttributes, numberOfClasses, populationSize, minValueFromDataset, maxValueFromDataset);
[perfectParams, bestDuringIterations] = prepareFisWithPSO(pop, data_matrix, dataSet, results, customFis);
drawTrainingProcess(bestDuringIterations);
customFis = parseFis(perfectParams, dataSet, customFis);
customFisResult = get_func_val(perfectParams, data_matrix, dataSet, results, customFis);
%plotfis(customFis)
genFisResult = genfisTest(data_matrix, results);
customFisResult 
genFisResult
customFis = parseFis(perfectParams, dataSet, customFis);
[acc_matrix, sensitivity] = get_acc_matrix(customFis, data_matrix, results, numberOfClasses);
acc_matrix
sensitivity

% Funkcja do losowania liczby z przekazanego zakresu
% Losowany zakres to [begin - 10% * range, end + 10%*range]
function number = randInRange(beginRange, endRange)
    range = endRange - beginRange;
    number = (range + range * 0.2) * rand() + (beginRange - range * 0.1);
end

%Funkcja do rysowania wykresu wartości globalnie najlepszego rozwiązania od
%numeru iteracji
function drawTrainingProcess(bests)
    bestSize = size(bests);
    plot([1:bestSize(2)], bests);
end

%Funkcja tworząca populacje
%customFis - dostrajany przez PSO układ logiki rozmytej
%numberOfAttributes - ilość cech w zbiorze danych
%numberOfClasses - ilość różnych wyjść
%populationSize - rozmiar populacji
%minValueFromDataset - wartość minimalna w zbiorze danych
%maxValueFromDataset - wartość maksymalna w zbiorze danych
function pop = generatePopulation(customFis, numberOfAttributes, numberOfClasses, populationSize, minValueFromDataset, maxValueFromDataset)
    [in, out] = getTunableSettings(customFis);
    % Wyznaczenie ilości osobników w populacji
    numberOfParameters = numberOfAttributes*numberOfClasses*length(in(1).MembershipFunctions(1).Parameters.Free);
    % Zainicjowanie populacji zerami
    pop = zeros(numberOfParameters, populationSize);
    % Przypisanie wylosowanych liczb do populacji
    for i=1:numberOfParameters
        for j=1:populationSize
            pop(i, j) = randInRange(minValueFromDataset, maxValueFromDataset); 
        end
    end
end

% Funkcja znajdujaca przy pomocy algorytmu rojowego strukture fisu.
% data_matrix - dane przygotowane do cross-10-validation
% dataSet - atrybuty irysów
function [fis, gbestList] = prepareFisWithPSO(pop, data_matrix, dataSet, results, basicFis)    
    [vector_size, pop_size] = size(pop); % vector_size to rozmiar pojedyńczego wektora populacji
    % pop_size to rozmiar całej populacji
    c1 = 2; % stala akceleracji
    c2 = 2; % stala akceleracji

    it = 0; % licznik iteracji
    it_max = 10; % maksymalny nr iteracji
    gbestList = zeros(1, it_max);
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

    w = calc_constriction_factor(c1, c2);
    
    while it <= it_max
        it = it + 1;
        % wyznaczamy przystosowanie kazdej czastki - wartosc funkcji w punkcie
        % i czy jest ona mniejsza od poprzedniej najlepszej dla danej czastki
        for i = 1 : pop_size
                % jak czasteczka w i-tej iteracji jest lepsza to
                % zamieniamy
                if i == 1 || get_func_val(pop(:, i), data_matrix, dataSet, results, basicFis) > get_func_val(pbest, data_matrix, dataSet, results, basicFis)
                    pbest = pop(:, i);
                end
        end
        
        %szukamy rozwiazania globalnie najlepszego
        if it == 1
            gbest = pbest;
        elseif get_func_val(pbest, data_matrix, dataSet, results, basicFis) > get_func_val(gbest, data_matrix, dataSet, results, basicFis)
            gbest = pbest;
        end
        
        %Przypisanie globalnie najlepszego rozwiązania w i-tej iteracji do
        %wektora (wykorzystane przy rysowaniu wykresu)
        gbestList(1, it) = get_func_val(gbest, data_matrix, dataSet, results, basicFis);
        
        % uaktualniamy predkosc i polozenie dla kazdej czastki
        for i = 1 : pop_size
                U1 = c1 * rand();
                U2 = c2 * rand();
                v(:, i) = w * (v(:, i) + U1 * (pbest - pop(:, i)) + U2 * (gbest - pop(:, i)));
                pop(:, i) = pop(:, i) + v(:, i);            
        end
        %Funkcja zwraca najlepszego osobnika z populacji
        fis = gbest;
    end
end

%Wyznaczanie współczynniku ścisku
function factor = calc_constriction_factor(c1, c2)
    fi = c1 + c2;
    factor = 2. / (2 + (fi * (fi - 4)) ^ 0.5);
end

% wybieramy najlepszy pod katem najmniejszego calkowitego bledu
% bezwzglednego
function value = get_func_val(vect, data_matrix, dataSet, results, basicFis)
    fis = parseFis(vect, dataSet, basicFis);
    value = cross_validation(data_matrix, fis, results);
end

%Ustawianie parametrów układu logiki rozmytej
function customFis = parseFis(vect, data_matrix, customFis)
    [in,out] = getTunableSettings(customFis);
    numberOfMembershipFunctions = length(in(1).MembershipFunctions);
    tmpSize = size(data_matrix);
    numberOfParameters = length(in(1).MembershipFunction(1).Parameters.Free) * numberOfMembershipFunctions * tmpSize(2);
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

%Przygotowanie reguł układu logiki rozmytej
%Funkcja działa prawidłowo dla układów które zwracają co najwyżej 3 klasy 
function customFis = prepareFisRules(data_matrix, customFis)
    [in,out] = getTunableSettings(customFis);
    numberOfMembershipFunctions = length(in(1).MembershipFunctions);
    %Przygotowanie zakresu dla każdej funkcji przynależności
    maxValueFromDataset = max(max(data_matrix));
    minValueFromDataset = min(min(data_matrix));
    minValue = minValueFromDataset - 0.1 * (maxValueFromDataset-minValueFromDataset);
    maxValue = maxValueFromDataset + 0.1 * (maxValueFromDataset-minValueFromDataset);
    for i=1:length(in)
        customFis.Inputs(i).Range = [minValue maxValue];
    end
    for i=1:length(out)
        customFis.Outputs(i).Range = [minValue maxValue];
    end
    % Jest jedno wyjście i trzy klasy więc można to zrobić ręcznie
    customFis.Outputs(1).MembershipFunctions(1).Parameters = [minValue (minValue+1/3*(maxValue-minValue) + minValue)/2 minValue+1/3*(maxValue-minValue)];
    customFis.Outputs(1).MembershipFunctions(2).Parameters = [minValue+1/3*(maxValue-minValue) (maxValue-1/3*(maxValue-minValue) + minValue+1/3*(maxValue-minValue))/2 maxValue-1/3*(maxValue-minValue)];
    customFis.Outputs(1).MembershipFunctions(3).Parameters = [maxValue-1/3*(maxValue-minValue) (maxValue-1/3*(maxValue-minValue) + maxValue) / 2 maxValue];
    % Przygotowanie reguł
    numberOfRules = numberOfMembershipFunctions ^ length(in);
    tmp = zeros(length(out(1).MembershipFunctions));
    tmpHolder = 0;
    for i=1:numberOfRules
        for j=1:length(customFis.Rule(i).Antecedent)
            if customFis.Rule(i).Antecedent(j) == 1
                tmp(1) = tmp(1) + 1;
            elseif customFis.Rule(i).Antecedent(j) == 2
                tmp(2) = tmp(2) + 1;
            else
                tmp(3) = tmp(3) + 1;
            end
            if tmp(1) > tmp(2) && tmp(1) > tmp(3)
                tmpHolder = 1;
            elseif tmp(2) > tmp(1) && tmp(2) > tmp(3)
                tmpHolder = 2;
            else
                tmpHolder = 3;
            end
        end
        customFis.Rule(i).Consequent = tmpHolder;
        tmp = zeros(length(out(1).MembershipFunctions));
    end
end

% Funkcja dzieląca zbiór danych na odpowiednią liczbę podzbiorów
% data - dane wejściowe
% numberOfFolds - ilość części na które dzielony jest zbiór
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

% Walidacja krzyżowa dla układu logiki rozmytej tworzonej ręcznie
% data_matrix - dane podzielone na części
% customFis - ręcznie utworzony układ logiki rozmytej
% results - prawidłowe wyniki
function value = cross_validation(data_matrix, customFis, results)
    data_size = size(data_matrix);
    averages = zeros(data_size(1));
    maxValue = max(max(max(data_matrix)));
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
            if out(j) < maxValue/3 && results(i, j) == 1
                correct = correct + 1;
            elseif out(j) >= maxValue/3 && out(j) < maxValue*2/3 && results(i, j) == 2
                correct = correct + 1;
            elseif out(j) >= maxValue*2/3 && results(i, j) == 3
                correct = correct + 1;
            end
       end
       average = correct / data_size(2);
       averages(i) = average;
    end
    tmpSum = 0;
    for i=1:length(averages)
        tmpSum = tmpSum + averages(i);
    end
    value = tmpSum / length(averages);
end

% Funkcja tworząca genfis do porównania z układem logiki rozmytej
% utworzonym ręcznie
% Utworzony genfis ma zaledwie 16 reguł, więc praktycznie zawsze będzie
% gorszy od fis'u utworzonego ręcznie
% data_matrix - dane podzielone na części
% results - prawidłowe wyniki
function value = genfisTest(data_matrix, results)
    flag = 0;
    data_size = size(results);
    averages = zeros(1, data_size(1));
    maxValue = max(max(max(data_matrix)));

    % genfis przy CV-10
    for i=1:10
        for j=1:10
            if j ~= i
                if flag == 0
                    perfectFis = genfis(squeeze(data_matrix(j, :, :)), results(j, :)', genfisOptions('GridPartition'));
                    [in, out] = getTunableSettings(perfectFis);
                    flag = 1;
                else
                    perfectFis = tunefis(perfectFis, [in;out], squeeze(data_matrix(j, :, :)), results(j, :)', tunefisOptions("Method", "anfis", "Display", "none"));
                end
            end
        end
        fisResults = evalfis(perfectFis, squeeze(data_matrix(i, :, :)));
        correct = 0;
        for j=1:data_size(2)
            if fisResults(j) < maxValue/3 && results(i, j) == 1
                correct = correct + 1;
            elseif fisResults(j) >= maxValue/3 && fisResults(j) < maxValue*2/3 && results(i, j) == 2
                correct = correct + 1;
            elseif fisResults(j) >= maxValue*2/3 && results(i, j) == 3
                correct = correct + 1;
            end
        end
        average = correct / data_size(2);
        averages(i) = average;
        flag = 0;
    end
    tmpSum = 0;
    for i=1:length(averages)
        tmpSum = tmpSum + averages(i);
    end
    value = tmpSum / length(averages);
end

% Funkcja tworząca macierz pomyłek i wyznaczająca czułość klasyfikatora
% customFis - ręcznie dostrajany układ logiki rozmytej
% data_matrix - macierz z danymi podzielona na części
% result - macierz z prawidłowymi wynikami
% NUM_OF_CLASSES - liczba klas wynikowych
function [acc_matrix, sensitivity] = get_acc_matrix(customFis, data_matrix, result, NUM_OF_CLASSES)
    acc_matrix = zeros(NUM_OF_CLASSES, NUM_OF_CLASSES);
    data_size = size(data_matrix);
    maxValue = max(max(max(data_matrix)));
    for i=1:data_size(1)
       testData = data_matrix(i, :, :);
       test_size = size(testData);
       testData = reshape(testData, test_size(2), test_size(3));
       % testowanie przygotowanego zbioru
       out = evalfis(customFis, testData);
       for j=1:data_size(2)
            if out(j) < maxValue/3
                out(j) = 1;
            elseif out(j) >= maxValue/3 && out(j) < maxValue*2/3
                out(j) = 2;
            elseif out(j) >= maxValue*2/3
                out(j) = 3;
            end
            acc_matrix(result(i, j), out(j)) = acc_matrix(result(i, j), out(j)) + 1;
       end
    end
    
    sensitivity = zeros(1, NUM_OF_CLASSES);
    tmpSum = 0;
    for i=1:NUM_OF_CLASSES
        currCorrect = acc_matrix(i, i);
        for j=1:NUM_OF_CLASSES
            tmpSum = tmpSum + acc_matrix(i, j);
        end
        sensitivity(1, i) = currCorrect / tmpSum;
        tmpSum = 0;
    end
end