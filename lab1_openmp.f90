program lab1
    use iso_fortran_env, only: real64
    use OMP_LIB ! подключение библиотеки OpenMP
    implicit none

    real(real64) :: start_time, end_time, total_time
    ! инициализация переменных, объявление констант
    integer, parameter :: elements_edu = 10000, elements_test = 100, fields = 5
    integer, parameter :: max_floors = 50, max_rooms = 5, max_square = 300, distincts = 3

    integer :: i, j, k ! итераторы для циклов
    real :: f, r, s, d

    integer :: apartments(elements_edu, fields)
    integer :: test_apartments(elements_test, fields)

    real, parameter :: base_price_per_sqm = 100.00
    real, parameter :: floorK = 0.2
    real, parameter :: roomK = 0.8
    real, parameter :: distinctK(3) = [0.6, 1.0, 1.8]

    integer :: k1 = max_square / max_floors, k2 = max_square / max_rooms
    integer::  k3 = 1, k4 = max_square / distincts 

    integer :: KNN = 35
    integer :: dist_temp, id_temp
    integer :: euclidean_distance(elements_edu)
    integer :: manhattan_distance(elements_edu)
    integer :: id_euclidean(elements_edu)
    integer :: id_manhattan(elements_edu)

    integer :: euclidean_predict_price = 0, manhattan_predict_price = 0
    real :: euclidean_accuracy(elements_test), manhattan_accuracy(elements_test)

    call cpu_time(start_time)

    ! генерация обучающей выборки
    call random_seed()
    
    !$OMP PARALLEL PRIVATE(f, r, s, d) SHARED(apartments)
    !$OMP DO
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
    !$OMP END DO
    !$OMP END PARALLEL

    print *, 'Обучающая выборка:'
    print *, '      Этаж', '    Кол-во комнат', '   Площадь', '   Район ID', '  Стоимость'

    do i = 1, 10 
        print *, apartments(i, :)
    end do

    ! генерация тестовой выборки
    !$OMP PARALLEL PRIVATE(f, r, s, d) SHARED(test_apartments)
    !$OMP DO
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
    !$OMP END DO
    !$OMP END PARALLEL
    
    ! === МНОГОПОТОЧНАЯ РЕАЛИЗАЦИЯ АЛГОРИТМА KNN ===
    
    !$OMP PARALLEL PRIVATE(j, k, dist_temp, id_temp, euclidean_predict_price, &
    !$OMP& manhattan_predict_price, euclidean_distance, manhattan_distance, &
    !$OMP& id_euclidean, id_manhattan) SHARED(apartments, test_apartments, &
    !$OMP& euclidean_accuracy, manhattan_accuracy)
    !$OMP DO
    do i = 1, elements_test
        do j = 1, elements_edu      
            ! вычисление Евклидова расстояния
            euclidean_distance(j) = int(sqrt(&
            k1 * (real(test_apartments(i, 1) - apartments(j, 1)))**2 + &
            k2 * (real(test_apartments(i, 2) - apartments(j, 2)))**2 + &
            k3 * (real(test_apartments(i, 3) - apartments(j, 3)))**2 + &
            k4 * (real(test_apartments(i, 4) - apartments(j, 4)))**2))

            id_euclidean(j) = j

            ! вычисление Манхеттенского расстояния
            manhattan_distance(j) = int(&
            k1 * abs(test_apartments(i, 1) - apartments(j, 1)) + &
            k2 * abs(test_apartments(i, 2) - apartments(j, 2)) + &
            k3 * abs(test_apartments(i, 3) - apartments(j, 3)) + &
            k4 * abs(test_apartments(i, 4) - apartments(j, 4)))

            id_manhattan(j) = j
        end do

        ! пузырьковая сортировка по расстоянию
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

        do j = 1, KNN
            euclidean_predict_price = euclidean_predict_price + apartments(id_euclidean(j), 5)
            manhattan_predict_price = manhattan_predict_price + apartments(id_manhattan(j), 5)
        end do
        euclidean_predict_price = euclidean_predict_price / KNN
        manhattan_predict_price = manhattan_predict_price / KNN

        euclidean_accuracy(j) = real(euclidean_predict_price)/test_apartments(i, 5)
        manhattan_accuracy(j) = real(manhattan_predict_price)/test_apartments(i, 5)

        !вот эту часть вынести в отдельный цикл
        !$OMP CRITICAL
        print *, 'Тестовая выборка:'
        print *, '       Этаж', '    Кол-во комнат', '   Площадь', '   Район ID',&
         '  Реал. цена', '    Евклид', '     Точность', '        Манхеттен', '   Точность'
        print *, test_apartments(i, :), euclidean_predict_price, euclidean_accuracy(j), &
                                        manhattan_predict_price, manhattan_accuracy(j)
        !$OMP END CRITICAL
    end do
    !$OMP END DO
    !$OMP END PARALLEL

    call cpu_time(end_time)
    total_time = end_time - start_time
    print *, 'Общее время выполнения: ', total_time, ' секунд'
end program lab1
