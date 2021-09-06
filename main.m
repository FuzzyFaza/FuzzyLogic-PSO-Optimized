clear;clc;warning('off');
iris_names = ["Setosa", "Versicolor", "Viriginica"];

% Nazwy w zbiorze iris zostaly zastepione przez liczbowe etykiety (1, 2, 3)
% aby latwiej bylo wczytac zbior danych

choice = input("Proszę wybrać zbiór danych, na którym będziemy operować:\n 1. Irysy 2. Zbiór ziaren - seeds 3. Haberman\n");
 
data_store = [];
switch choice
    case 1
        data_store = load('Data/iris.data');
    case 2
        data_store = load('Data/seeds_dataset.txt');
    case 3
        data_store = load('Data/haberman.data');
        habermanSize = size(data_store);
        data_store = sortrows(data_store, habermanSize(2));
    otherwise
        fprintf("Nie wybrano poprawnie zbioru\n")
        return
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
drawTrainingProcess(bestDuringIterations, 'mse(it)_v=[0,1]');
customFis = parseFis(perfectParams, dataSet, customFis);
customFisResult = get_func_val(perfectParams, data_matrix, dataSet, results, customFis);
genFisResult = genfisTest(data_matrix, results);
customFisResult 
genFisResult
customFis = parseFis(perfectParams, dataSet, customFis);
[acc_matrix, sensitivity, percent] = get_acc_matrix_versatile(customFis, data_matrix, results, numberOfClasses);
acc_matrix
sensitivity
percent

% Funkcja do losowania liczby z przekazanego zakresu
% Losowany zakres to [begin - 10% * range, end + 10%*range]
function number = randInRange(beginRange, endRange)
    range = endRange - beginRange;
    number = (range + range * 0.2) * rand() + (beginRange - range * 0.1);
end

%Funkcja do rysowania wykresu wartości globalnie najlepszego rozwiązania od
%numeru iteracji
function drawTrainingProcess(bests, file_name)
    gcf = figure;
    bestSize = size(bests);
    plot([1:bestSize(2)], bests)
    title("Calkowity globalny blad sredniokwadratowy w funkcji nr iteracji")
    xlabel("Nr iteracji")
    ylabel("MSE")
    saveas(gcf, file_name)
    close(gcf)
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
    numberOfParameters = numberOfAttributes*3*length(in(1).MembershipFunctions(1).Parameters.Free);
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
    c1 = 2.05; % stala akceleracji
    c2 = 2.05; % stala akceleracji

    it = 0; % licznik iteracji
    it_max = 100; % maksymalny nr iteracji
    gbestList = zeros(1, it_max);
    % wektor predkosci
    v = zeros(vector_size, pop_size); % wektor o dlugosci wiersza danych (60 dla irysow)
    
    % losowo generujemy predkosci
    for i = 1 : vector_size
        for j = 1: pop_size
            v(i, j) = randInRange(0, 1); % zakres losowania [0-1]
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
                if i == 1 || get_func_val(pop(:, i), data_matrix, dataSet, results, basicFis) < get_func_val(pbest, data_matrix, dataSet, results, basicFis)
                    pbest = pop(:, i);
                end
        end
        
        %szukamy rozwiazania globalnie najlepszego
        if it == 1
            gbest = pbest;
        elseif get_func_val(pbest, data_matrix, dataSet, results, basicFis) < get_func_val(gbest, data_matrix, dataSet, results, basicFis)
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
end

%Przygotowanie reguł układu logiki rozmytej
%Funkcja działa prawidłowo dla układów które zwracają co najwyżej 3 klasy 
function customFis = prepareFisRules(data_matrix, customFis)
    [in,out] = getTunableSettings(customFis);
    numberOfMembershipFunctions = length(in(1).MembershipFunctions);
    %Przygotowanie zakresu dla każdej funkcji przynależności
    maxValueFromDataset = max(data_matrix);
    minValueFromDataset = min(data_matrix);
    
    minValue = minValueFromDataset - 0.1 * (maxValueFromDataset-minValueFromDataset);
    maxValue = maxValueFromDataset + 0.1 * (maxValueFromDataset-minValueFromDataset);
    
    for i = 1 : length(in)
        customFis.Inputs(i).Range = [minValue(i) maxValue(i)];
    end
    
    for i = 1 : length(out)
        customFis.Outputs(i).Range = [min(minValue) max(maxValue)];
    end
    customFis.Outputs(1).MembershipFunctions(1).Parameters = [-1 1 1.5];
    customFis.Outputs(1).MembershipFunctions(2).Parameters = [1.51 2 2.5];
    customFis.Outputs(1).MembershipFunctions(3).Parameters = [2.51 3 4];
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
    value = 0;
    for i=1:data_size(1)
       testData = data_matrix(i, :, :);
       test_size = size(testData);
       testData = reshape(testData, test_size(2), test_size(3));
       % testowanie przygotowanego zbioru
       out = evalfis(customFis, testData);
       % badanie dokładności i zapisanie w tablicy do policzenia wartości średniej
       % wyliczenie błędu średniokwadratowego
       for j=1:length(out)
          value = value + (out(j) - results(i, j))^2; 
       end
    end
end

% Funkcja tworząca genfis do porównania z układem logiki rozmytej
% utworzonym ręcznie
% data_matrix - dane podzielone na części
% results - prawidłowe wyniki
function value = genfisTest(data_matrix, results)
    flag = 0;
    data_size = size(results);

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
        value = 0;
        for j=1:data_size(2)
            value = value + (fisResults(j) - results(i, j))^2;
        end
        flag = 0;
    end
end

% Funkcja tworząca macierz pomyłek i wyznaczająca czułość klasyfikatora
% customFis - ręcznie dostrajany układ logiki rozmytej
% data_matrix - macierz z danymi podzielona na części
% result - macierz z prawidłowymi wynikami
% NUM_OF_CLASSES - liczba klas wynikowych
function [acc_matrix, sensitivity, percent] = get_acc_matrix_versatile(customFis, data_matrix, result, NUM_OF_CLASSES)
    acc_matrix = zeros(NUM_OF_CLASSES, NUM_OF_CLASSES);
    data_size = size(data_matrix);
    [globalMin, globalMax, globalMedian, globalAverage, delta, averages, medians] = getOutputParams(customFis, data_matrix, result, NUM_OF_CLASSES);
    averages
    medians
    valueSet = sort(medians);
    newIndexes = zeros(1, length(medians));
    for i=1:length(valueSet)
            for j=1:length(valueSet) 
                if medians(i) == valueSet(j)
                    newIndexes(1, i) = j;
                end
            end
    end
    medians = sort(medians);
    averages = sort(averages);
    interval_endings = zeros(1, NUM_OF_CLASSES + 1);
    factor = 1. / (2*NUM_OF_CLASSES);
    interval_endings(1) = globalMin;
    interval_endings(NUM_OF_CLASSES + 1) = globalMax; 
    for i=1:data_size(1)
       testData = data_matrix(i, :, :);
       test_size = size(testData);
       testData = reshape(testData, test_size(2), test_size(3));
       % testowanie przygotowanego zbioru
       out = evalfis(customFis, testData);
       
       if NUM_OF_CLASSES > 2
            for k = 2 : floor((NUM_OF_CLASSES + 1) / 2)
                interval_endings(k) = globalMedian - (medians(k)-medians(k-1))/2;
            end
      
            if mod(NUM_OF_CLASSES + 1, 2) ~= 0
                interval_endings(floor((NUM_OF_CLASSES + 1) / 2)) = globalMedian;
            end
       
            for k = floor((NUM_OF_CLASSES + 1) / 2) + 1 : NUM_OF_CLASSES
                interval_endings(k) = globalMedian + (medians(k)-medians(k-1))/2; 
            end
       
       else
           interval_endings(1) = globalMin;
           interval_endings(2) = globalMedian + (medians(2) - medians(1)) / 2;
           interval_endings(3) = globalMax;
       end
       
       for j=1:data_size(2)
            current_result = out(j);
            
            for k = 2 : NUM_OF_CLASSES + 1
                if current_result >= interval_endings(k - 1) && current_result <= interval_endings(k)
                    out(j) = newIndexes(k-1);
                end    
            end
            
            acc_matrix(result(i, j), out(j)) = acc_matrix(result(i, j), out(j)) + 1;
       end
    end
    
    sensitivity = zeros(1, NUM_OF_CLASSES);
    tmpSum = 0;
    totalCorrect = 0;
    for i=1:NUM_OF_CLASSES
        currCorrect = acc_matrix(i, i);
        for j=1:NUM_OF_CLASSES
            tmpSum = tmpSum + acc_matrix(i, j);
        end
        sensitivity(1, i) = currCorrect / tmpSum;
        totalCorrect = totalCorrect + currCorrect;
        tmpSum = 0;
    end
    percent = totalCorrect / sum(sum(acc_matrix));
    [interval_endings]
    newIndexes
end

function value = findMedian(vec)
    vec = sort(vec);
    len = length(vec);
    if mod(len, 2) == 0
       value = (vec(len/2) + vec((len)/2 + 1)) / 2; 
    else
        value = vec((len + 1)/2);
    end
end

%Funkcja zwracająca informacje na temat wyników otrzymanych z customFis.
function [globalMin, globalMax, globalMedian, globalAverage, delta, averages, medians] = getOutputParams(customFis, data_matrix, results, numOfClasses)
    data_size = size(data_matrix);
    allOuts = zeros(1, data_size(2)*data_size(3));
    averages = zeros(1, numOfClasses);
    medians = zeros(1, numOfClasses);
    classSizes = zeros(1, numOfClasses);
    counter = 1;
    secCounter = 1;
    tmp = 1;
    globalMin = 10000;
    globalMax = -10000;
    for i=1:data_size(1)
       testData = data_matrix(i, :, :);
       test_size = size(testData);
       testData = reshape(testData, test_size(2), test_size(3));
       out = evalfis(customFis, testData);
       localMin = min(out);
       localMax = max(out);
       if localMin < globalMin
           globalMin = localMin;
       end
       if localMax > globalMax
           globalMax = localMax;
       end
       for j=1:length(out)
           allOuts(1, counter) = out(j);
           counter = counter + 1;
           tmpX = tmp;
           tmpY = mod(counter-1, data_size(2));
           if tmpY == 0 
               tmpY = data_size(2); 
           end
           for k=1:numOfClasses
              if results(tmpX,tmpY) == k
                 averages(k) = averages(k) + out(j);
                 classSizes(k) = classSizes(k) + 1;
              end
           end
       end
       tmp = tmp + 1;
    end
    for k=1:numOfClasses
       averages(k) = averages(k) / classSizes(k); 
    end
    sizeHolder = counter - 1;
    counter = 1;
    tmp = 1;
    for k=1:numOfClasses
        test = zeros(1, classSizes(k));
        for i=1:data_size(1)
            testData = data_matrix(i, :, :);
            test_size = size(testData);
            testData = reshape(testData, test_size(2), test_size(3));
            out = evalfis(customFis, testData);
            for j=1:length(out)
                counter = counter + 1;
                tmpX = tmp;
                tmpY = mod(counter-1, data_size(2));
                if tmpY == 0 
                    tmpY = data_size(2); 
                end
                if results(tmpX,tmpY) == k
                    test(secCounter) = out(j);
                    secCounter = secCounter + 1;
                end
            end
            tmp = tmp + 1;
        end
        test = reshape(test, [classSizes(k), 1]);
        medians(k) = findMedian(test);
        tmp = 1;
        counter = 1;
        secCounter = 1;
    end
    allOuts = reshape(allOuts, [sizeHolder,1]);
    globalMedian = findMedian(allOuts);
    delta = globalMax - globalMin;
    globalAverage = (globalMin + globalMax) / 2; 
end