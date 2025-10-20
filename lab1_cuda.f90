program lab1_cuda
    use cudafor
    use iso_fortran_env, only: real64
    implicit none

    real(real64) :: start_time, end_time, total_time
    integer, parameter :: elements_edu = 10000, elements_test = 100, fields = 5
    integer, parameter :: max_floors = 50, max_rooms = 5, max_square = 300, distincts = 3

    integer :: i, j
    real :: f, r, s, d

    integer :: apartments(elements_edu, fields)
    integer :: test_apartments(elements_test, fields)
    
    ! GPU arrays
    integer, device :: apartments_d(elements_edu, fields)
    integer, device :: test_apartments_d(elements_test, fields)
    integer, device :: euclidean_distance_d(elements_edu)
    integer, device :: manhattan_distance_d(elements_edu)
    integer, device :: id_euclidean_d(elements_edu)
    integer, device :: id_manhattan_d(elements_edu)

    real, parameter :: base_price_per_sqm = 100.00
    real, parameter :: floorK = 0.2
    real, parameter :: roomK = 0.8
    real, parameter :: distinctK(3) = [0.6, 1.0, 1.8]

    integer :: k1 = max_square / max_floors, k2 = max_square / max_rooms
    integer :: k3 = 1, k4 = max_square / distincts 

    integer :: KNN = 35
    integer :: euclidean_predict_price, manhattan_predict_price
    real :: euclidean_accuracy(elements_test), manhattan_accuracy(elements_test)
    
    ! Temporary arrays for CPU
    integer :: euclidean_distance(elements_edu)
    integer :: manhattan_distance(elements_edu)
    integer :: id_euclidean(elements_edu)
    integer :: id_manhattan(elements_edu)
    
    type(dim3) :: grid, block
    integer :: istat

    call cpu_time(start_time)

    ! генерация обучающей выборки
    call random_seed()
    
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

    print *, 'Обучающая выборка:'
    print *, '      Этаж', '    Кол-во комнат', '   Площадь', '   Район ID', '  Стоимость'

    do i = 1, 10
        print *, apartments(i, :)
    end do

    ! генерация тестовой выборки
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

    ! Копируем данные на GPU
    apartments_d = apartments
    test_apartments_d = test_apartments

    ! Настройка grid и block
    block = dim3(256, 1, 1)
    grid = dim3(ceiling(real(elements_edu)/block%x), 1, 1)

    print *, 'Тестовая выборка:'
    print *, '       Этаж', '    Кол-во комнат', '   Площадь', '   Район ID',&
     '  Реал. цена', '    Евклид', '     Точность', '        Манхеттен', '   Точность'

    do i = 1, elements_test
        ! Инициализация индексов на GPU
        call init_indices<<<grid, block>>>(id_euclidean_d, id_manhattan_d, elements_edu)
        istat = cudaDeviceSynchronize()
        
        ! Вычисление расстояний на GPU
        call compute_distances<<<grid, block>>>(test_apartments_d, apartments_d, &
                                              euclidean_distance_d, manhattan_distance_d, &
                                              id_euclidean_d, id_manhattan_d, &
                                              k1, k2, k3, k4, i, elements_edu)
        istat = cudaDeviceSynchronize()

        ! Копируем результаты обратно на CPU для сортировки
        euclidean_distance = euclidean_distance_d
        manhattan_distance = manhattan_distance_d
        id_euclidean = id_euclidean_d
        id_manhattan = id_manhattan_d

        ! Сортировка на CPU (можно заменить на GPU сортировку для большей производительности)
        call bubble_sort(euclidean_distance, id_euclidean, elements_edu)
        call bubble_sort(manhattan_distance, id_manhattan, elements_edu)

        ! Предсказание цены
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

        print *, test_apartments(i, :), euclidean_predict_price, euclidean_accuracy(i), &
                                        manhattan_predict_price, manhattan_accuracy(i)
    end do
    
    call cpu_time(end_time)
    total_time = end_time - start_time
    print *, 'Общее время выполнения: ', total_time, ' секунд'

contains

    attributes(global) subroutine init_indices(id_euclidean, id_manhattan, n)
        integer, device :: id_euclidean(*), id_manhattan(*)
        integer, value :: n
        integer :: idx
        
        idx = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        if (idx <= n) then
            id_euclidean(idx) = idx
            id_manhattan(idx) = idx
        end if
    end subroutine init_indices

    attributes(global) subroutine compute_distances(test_apt, apt, &
                                                   euclidean_dist, manhattan_dist, &
                                                   id_euclidean, id_manhattan, &
                                                   k1, k2, k3, k4, test_idx, n)
        integer, device :: test_apt(elements_test, fields), apt(elements_edu, fields)
        integer, device :: euclidean_dist(*), manhattan_dist(*)
        integer, device :: id_euclidean(*), id_manhattan(*)
        integer, value :: k1, k2, k3, k4, test_idx, n
        integer :: idx
        real :: diff1, diff2, diff3, diff4
        
        idx = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        if (idx <= n) then
            ! Евклидово расстояние
            diff1 = real(test_apt(test_idx, 1) - apt(idx, 1))
            diff2 = real(test_apt(test_idx, 2) - apt(idx, 2))
            diff3 = real(test_apt(test_idx, 3) - apt(idx, 3))
            diff4 = real(test_apt(test_idx, 4) - apt(idx, 4))
            
            euclidean_dist(idx) = int(sqrt(&
                k1 * diff1 * diff1 + &
                k2 * diff2 * diff2 + &
                k3 * diff3 * diff3 + &
                k4 * diff4 * diff4))
            
            ! Манхэттенское расстояние
            manhattan_dist(idx) = int(&
                k1 * abs(diff1) + &
                k2 * abs(diff2) + &
                k3 * abs(diff3) + &
                k4 * abs(diff4))
        end if
    end subroutine compute_distances

    subroutine bubble_sort(distances, ids, n)
        integer :: distances(*), ids(*)
        integer, value :: n
        integer :: j, k, dist_temp, id_temp
        
        do j = 1, n - 1
            do k = j + 1, n
                if (distances(j) > distances(k)) then
                    dist_temp = distances(j)
                    distances(j) = distances(k)
                    distances(k) = dist_temp

                    id_temp = ids(j)
                    ids(j) = ids(k)
                    ids(k) = id_temp
                end if
            end do
        end do
    end subroutine bubble_sort

end program lab1_cuda
