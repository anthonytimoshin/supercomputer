program lab1_apple_silicon
    use iso_fortran_env, only: real64
    use omp_lib
    implicit none

    real(real64) :: start_time, end_time, total_time
    integer, parameter :: elements_edu = 10000, elements_test = 100, fields = 5
    integer, parameter :: max_floors = 50, max_rooms = 5, max_square = 300, distincts = 3

    integer :: i, j, k
    real :: f, r, s, d
    real :: diff1, diff2, diff3, diff4
    real :: sq1, sq2, sq3, sq4

    ! ОБЪЯВЛЕНИЕ МАССИВОВ
    integer :: apartments(elements_edu, fields)
    integer :: test_apartments(elements_test, fields)
    integer :: euclidean_distance(elements_edu)
    integer :: manhattan_distance(elements_edu)
    integer :: id_euclidean(elements_edu)
    integer :: id_manhattan(elements_edu)

    real, parameter :: base_price_per_sqm = 100.00
    real, parameter :: floorK = 0.2
    real, parameter :: roomK = 0.8
    real, parameter :: distinctK(3) = [0.6, 1.0, 1.8]

    integer :: k1 = max_square / max_floors, k2 = max_square / max_rooms
    integer :: k3 = 1, k4 = max_square / distincts 
    integer :: KNN = 35

    integer :: euclidean_predict_price, manhattan_predict_price
    real :: euclidean_accuracy(elements_test), manhattan_accuracy(elements_test)
    integer :: dist_temp, id_temp

    ! ДИАГНОСТИКА APPLE SILICON
    print *, "=== APPLE SILICON M1 DIAGNOSTICS ==="
    print *, "Number of GPU devices: ", omp_get_num_devices()
    print *, "Max CPU threads: ", omp_get_max_threads()
    
    ! Настройка для Apple Silicon
    if (omp_get_num_devices() > 0) then
        call omp_set_default_device(0)  ! Основное устройство (GPU)
        print *, "GPU devices available - using device 0"
    else
        print *, "No GPU devices - using CPU only"
    end if

    start_time = omp_get_wtime()

    ! Генерация данных с использованием всех ядер CPU
    call random_seed()
    
    !$omp parallel do private(i, f, r, s, d) schedule(static, 32)
    do i = 1, elements_edu
        call random_number(f)
        apartments(i, 1) = int(f * max_floors + 1)
        call random_number(r)
        apartments(i, 2) = int(r * max_rooms + 1)
        call random_number(s)
        apartments(i, 3) = int(s * max_square + apartments(i, 2) * 10)
        call random_number(d)
        apartments(i, 4) = int(d * distincts + 1)
        apartments(i, 5) = int(apartments(i, 1) * floorK * apartments(i, 2) * roomK * &
                           apartments(i, 3) * base_price_per_sqm * distinctK(apartments(i, 4)))
    end do
    !$omp end parallel do

    !$omp parallel do private(i, f, r, s, d) schedule(static, 32)
    do i = 1, elements_test
        call random_number(f)
        test_apartments(i, 1) = int(f * max_floors + 1)
        call random_number(r)
        test_apartments(i, 2) = int(r * max_rooms + 1)
        call random_number(s)
        test_apartments(i, 3) = int(s * max_square + test_apartments(i, 2) * 10)
        call random_number(d)
        test_apartments(i, 4) = int(d * distincts + 1)
        test_apartments(i, 5) = int(test_apartments(i, 1) * floorK * test_apartments(i, 2) * roomK * &
                                test_apartments(i, 3) * base_price_per_sqm * distinctK(test_apartments(i, 4)))
    end do
    !$omp end parallel do

    print *, 'Обучающая выборка:'
    do i = 1, 5
        print *, apartments(i, :)
    end do

    ! === ВЫЧИСЛЕНИЯ НА GPU ===
    
    ! Используем Unified Memory Architecture Apple Silicon
    !$omp target enter data map(to: apartments, test_apartments) &
    !$omp& map(alloc: euclidean_distance, manhattan_distance, id_euclidean, id_manhattan)

    ! Основной цикл с гибридными вычислениями
    !$omp parallel do private(i, j, k, diff1, diff2, diff3, diff4, &
    !$omp& sq1, sq2, sq3, sq4, dist_temp, id_temp, &
    !$omp& euclidean_predict_price, manhattan_predict_price) &
    !$omp& schedule(dynamic, 5)
    do i = 1, elements_test
        
        ! ВЫЧИСЛЕНИЕ РАССТОЯНИЙ НА GPU (графические ядра)
        !$omp target teams distribute parallel do &
        !$omp& private(j, diff1, diff2, diff3, diff4, sq1, sq2, sq3, sq4) &
        !$omp& shared(apartments, test_apartments, euclidean_distance, &
        !$omp& manhattan_distance, id_euclidean, id_manhattan, i, &
        !$omp& k1, k2, k3, k4)
        do j = 1, elements_edu
            ! Векторизованные вычисления
            diff1 = real(test_apartments(i, 1) - apartments(j, 1))
            diff2 = real(test_apartments(i, 2) - apartments(j, 2))
            diff3 = real(test_apartments(i, 3) - apartments(j, 3))
            diff4 = real(test_apartments(i, 4) - apartments(j, 4))
            
            ! Векторизованные операции
            sq1 = k1 * diff1 * diff1
            sq2 = k2 * diff2 * diff2  
            sq3 = k3 * diff3 * diff3
            sq4 = k4 * diff4 * diff4
            
            euclidean_distance(j) = int(sqrt(sq1 + sq2 + sq3 + sq4))
            
            ! Векторизованные абсолютные значения
            manhattan_distance(j) = int( &
                k1 * abs(diff1) + k2 * abs(diff2) + k3 * abs(diff3) + k4 * abs(diff4))
            
            id_euclidean(j) = j
            id_manhattan(j) = j
        end do
        !$omp end target teams distribute parallel do

        ! Быстрое копирование через Unified Memory
        !$omp target update from(euclidean_distance, manhattan_distance, &
        !$omp&                   id_euclidean, id_manhattan)

        ! ПУЗЫРЬКОВАЯ СОРТИРОВКА НА CPU
        do j = 1, elements_edu - 1
            do k = j + 1, elements_edu
                if (euclidean_distance(j) > euclidean_distance(k)) then
                    dist_temp = euclidean_distance(j)
                    euclidean_distance(j) = euclidean_distance(k)
                    euclidean_distance(k) = dist_temp
                    id_temp = id_euclidean(j)
                    id_euclidean(j) = id_euclidean(k)
                    id_euclidean(k) = id_temp
                end if
                if (manhattan_distance(j) > manhattan_distance(k)) then
                    dist_temp = manhattan_distance(j)
                    manhattan_distance(j) = manhattan_distance(k)
                    manhattan_distance(k) = dist_temp
                    id_temp = id_manhattan(j)
                    id_manhattan(j) = id_manhattan(k)
                    id_manhattan(k) = id_temp
                end if
            end do
        end do

        ! ВЫЧИСЛЕНИЕ ПРЕДСКАЗАНИЙ
        euclidean_predict_price = 0
        manhattan_predict_price = 0
        
        do j = 1, KNN
            euclidean_predict_price = euclidean_predict_price + apartments(id_euclidean(j), 5)
            manhattan_predict_price = manhattan_predict_price + apartments(id_manhattan(j), 5)
        end do
        
        euclidean_predict_price = euclidean_predict_price / KNN
        manhattan_predict_price = manhattan_predict_price / KNN

        euclidean_accuracy(i) = real(euclidean_predict_price)/test_apartments(i, 5)
        manhattan_accuracy(i) = real(manhattan_predict_price)/test_apartments(i, 5)

        !$omp critical
        if (i <= 5) then
            print *, 'Тестовая выборка', i, ':'
            print *, 'Реальная цена:', test_apartments(i, 5), &
                     'Евклид:', euclidean_predict_price, 'Точность:', euclidean_accuracy(i), &
                     'Манхэттен:', manhattan_predict_price, 'Точность:', manhattan_accuracy(i)
        end if
        !$omp end critical
    end do
    !$omp end parallel do

    !$omp target exit data map(delete: apartments, test_apartments, &
    !$omp& euclidean_distance, manhattan_distance, id_euclidean, id_manhattan)

    end_time = omp_get_wtime()
    total_time = end_time - start_time
    
    print *, '========================================='
    print *, 'APPLE SILICON M1 РЕЗУЛЬТАТЫ:'
    print *, 'Общее время выполнения: ', total_time, ' секунд'
    print *, 'Обработано тестовых образцов: ', elements_test
    print *, 'Размер обучающей выборки: ', elements_edu
    print *, 'Использована архитектура Apple Silicon'
    print *, '========================================='

end program lab1_apple_silicon
