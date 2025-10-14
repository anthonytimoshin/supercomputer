program lab1
    use cudafor
    use iso_fortran_env, only: real64
    implicit none

    real(real64) :: start_time, end_time, total_time
    ! инициализация переменных, объявление констант
    integer, parameter :: elements_edu = 10000, elements_test = 100, fields = 5
    integer, parameter :: max_floors = 50, max_rooms = 5, max_square = 300, distincts = 3

    integer :: i, j, k ! итераторы для циклов
    real :: f, r, s, d

    integer, device :: apartments_d(elements_edu, fields)
    integer, device :: test_apartments_d(elements_test, fields)
    integer :: apartments_h(elements_edu, fields)
    integer :: test_apartments_h(elements_test, fields)

    real, parameter :: base_price_per_sqm = 100.00
    real, parameter :: floorK = 0.2
    real, parameter :: roomK = 0.8
    real, parameter :: distinctK(3) = [0.6, 1.0, 1.8]

    integer :: k1 = 50, k2 = 60, k3 = 1, k4 = 100
    integer :: KNN = 35
    integer :: dist_temp, id_temp
    
    ! Device arrays for distances and indices
    integer, device :: euclidean_distance_d(elements_edu)
    integer, device :: manhattan_distance_d(elements_edu)
    integer, device :: id_euclidean_d(elements_edu)
    integer, device :: id_manhattan_d(elements_edu)
    
    ! Host arrays for results
    integer :: euclidean_distance_h(elements_edu)
    integer :: manhattan_distance_h(elements_edu)
    integer :: id_euclidean_h(elements_edu)
    integer :: id_manhattan_h(elements_edu)

    integer :: euclidean_predict_price, manhattan_predict_price
    real :: euclidean_accuracy(elements_test), manhattan_accuracy(elements_test)
    
    ! CUDA variables
    type(dim3) :: gridDim, blockDim
    integer :: istat

    call cpu_time(start_time)

    ! генерация обучающей выборки на хосте
    call random_seed()
    
    do i = 1, elements_edu
        call random_number(f)
        apartments_h(i, 1) = int(f * max_floors + 1)

        call random_number(r)
        apartments_h(i, 2) = int(r * max_rooms + 1)

        call random_number(s)
        apartments_h(i, 3) = int(s * max_square + apartments_h(i, 2) * 10)

        call random_number(d)
        apartments_h(i, 4) = int(d * distincts + 1)

        apartments_h(i, 5) = int(apartments_h(i, 1) * floorK * apartments_h(i, 2) * roomK * &
                           apartments_h(i, 3) * base_price_per_sqm * distinctK(apartments_h(i, 4)))
    end do

    ! Копирование данных на устройство
    apartments_d = apartments_h

    print *, 'Обучающая выборка:'
    print *, '      Этаж', '    Кол-во комнат', '   Площадь', '   Район ID', '  Стоимость'

    do i = 1, 10
        print *, apartments_h(i, :)
    end do

    ! генерация тестовой выборки на хосте
    do i = 1, elements_test
        call random_number(f)
        test_apartments_h(i, 1) = int(f * max_floors + 1)

        call random_number(r)
        test_apartments_h(i, 2) = int(r * max_rooms + 1)

        call random_number(s)
        test_apartments_h(i, 3) = int(s * max_square + test_apartments_h(i, 2) * 10)

        call random_number(d)
        test_apartments_h(i, 4) = int(d * distincts + 1)

        test_apartments_h(i, 5) = int(test_apartments_h(i, 1) * floorK * test_apartments_h(i, 2) * roomK * &
                                test_apartments_h(i, 3) * base_price_per_sqm * distinctK(test_apartments_h(i, 4)))
    end do

    ! Копирование тестовых данных на устройство
    test_apartments_d = test_apartments_h

    ! === CUDA РЕАЛИЗАЦИЯ АЛГОРИТМА KNN ===
    
    ! Настройка размерности grid и block
    blockDim = dim3(256, 1, 1)
    gridDim = dim3(ceiling(real(elements_edu) / real(blockDim%x)), 1, 1)
    
    do i = 1, elements_test
        ! Запуск kernel для вычисления расстояний на GPU
        call compute_distances_kernel<<<gridDim, blockDim>>>( &
            apartments_d, test_apartments_d, &
            euclidean_distance_d, manhattan_distance_d, &
            id_euclidean_d, id_manhattan_d, &
            elements_edu, fields, i, k1, k2, k3, k4)
        
        istat = cudaDeviceSynchronize()
        
        ! Копирование результатов обратно на хост для сортировки
        euclidean_distance_h = euclidean_distance_d
        manhattan_distance_h = manhattan_distance_d
        id_euclidean_h = id_euclidean_d
        id_manhattan_h = id_manhattan_d

        ! пузырьковая сортировка по расстоянию (на хосте)
        do j = 1, elements_edu - 1
            do k = j + 1, elements_edu
                if (euclidean_distance_h(j) > euclidean_distance_h(k)) then
                    dist_temp = euclidean_distance_h(j)
                    euclidean_distance_h(j) = euclidean_distance_h(k)
                    euclidean_distance_h(k) = dist_temp

                    id_temp = id_euclidean_h(j)
                    id_euclidean_h(j) = id_euclidean_h(k)
                    id_euclidean_h(k) = id_temp
                end if

                if (manhattan_distance_h(j) > manhattan_distance_h(k)) then
                    dist_temp = manhattan_distance_h(j)
                    manhattan_distance_h(j) = manhattan_distance_h(k)
                    manhattan_distance_h(k) = dist_temp

                    id_temp = id_manhattan_h(j)
                    id_manhattan_h(j) = id_manhattan_h(k)
                    id_manhattan_h(k) = id_temp
                end if
            end do
        end do

        ! Вычисление предсказанной цены
        euclidean_predict_price = 0
        manhattan_predict_price = 0
        
        do j = 1, KNN
            euclidean_predict_price = euclidean_predict_price + apartments_h(id_euclidean_h(j), 5)
            manhattan_predict_price = manhattan_predict_price + apartments_h(id_manhattan_h(j), 5)
        end do
        euclidean_predict_price = euclidean_predict_price / KNN
        manhattan_predict_price = manhattan_predict_price / KNN

        euclidean_accuracy(i) = real(euclidean_predict_price) / real(test_apartments_h(i, 5))
        manhattan_accuracy(i) = real(manhattan_predict_price) / real(test_apartments_h(i, 5))

        print *, 'Тестовая выборка ', i, ':'
        print *, '       Этаж', '    Кол-во комнат', '   Площадь', '   Район ID',&
         '  Реал. цена', '    Евклид', '     Точность', '        Манхеттен', '   Точность'
        print *, test_apartments_h(i, 1:4), test_apartments_h(i, 5), &
                euclidean_predict_price, euclidean_accuracy(i), &
                manhattan_predict_price, manhattan_accuracy(i)
    end do
    
    call cpu_time(end_time)
    total_time = end_time - start_time
    print *, 'Общее время выполнения: ', total_time, ' секунд'
    print *, 'CUDA version'

contains

    attributes(global) subroutine compute_distances_kernel( &
        apartments, test_apartments, &
        euclidean_distance, manhattan_distance, &
        id_euclidean, id_manhattan, &
        n_apartments, n_fields, test_idx, k1, k2, k3, k4)
        
        integer, intent(in) :: apartments(n_apartments, n_fields)
        integer, intent(in) :: test_apartments(n_apartments, n_fields)
        integer, intent(out) :: euclidean_distance(n_apartments)
        integer, intent(out) :: manhattan_distance(n_apartments)
        integer, intent(out) :: id_euclidean(n_apartments)
        integer, intent(out) :: id_manhattan(n_apartments)
        integer, value :: n_apartments, n_fields, test_idx, k1, k2, k3, k4
        
        integer :: idx
        real :: diff1, diff2, diff3, diff4
        
        idx = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        
        if (idx <= n_apartments) then
            ! Вычисление Евклидова расстояния
            diff1 = real(test_apartments(test_idx, 1) - apartments(idx, 1))
            diff2 = real(test_apartments(test_idx, 2) - apartments(idx, 2))
            diff3 = real(test_apartments(test_idx, 3) - apartments(idx, 3))
            diff4 = real(test_apartments(test_idx, 4) - apartments(idx, 4))
            
            euclidean_distance(idx) = int(sqrt( &
                k1 * diff1**2 + &
                k2 * diff2**2 + &
                k3 * diff3**2 + &
                k4 * diff4**2))
            
            ! Вычисление Манхеттенского расстояния
            manhattan_distance(idx) = int( &
                k1 * abs(diff1) + &
                k2 * abs(diff2) + &
                k3 * abs(diff3) + &
                k4 * abs(diff4))
            
            ! Сохранение индексов
            id_euclidean(idx) = idx
            id_manhattan(idx) = idx
        end if
        
    end subroutine compute_distances_kernel

end program lab1
