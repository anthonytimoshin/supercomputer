program lab_mpi
    use iso_fortran_env, only: real64
    use mpi
    implicit none

    ! MPI переменные
    integer :: ierr, rank, num_procs
    integer :: status(MPI_STATUS_SIZE)
    
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
    
    ! Переменные для распределения работы
    integer :: chunk_size, start_idx, end_idx, extra_work
    integer, allocatable :: local_euclidean_predict(:), local_manhattan_predict(:)
    real, allocatable :: local_euclidean_accuracy(:), local_manhattan_accuracy(:)

    ! Инициализация MPI
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, num_procs, ierr)

    if (rank == 0) then
        call cpu_time(start_time)
        print *, 'Количество процессов:', num_procs
    end if

    ! Генерация обучающей выборки (только на процессе 0)
    if (rank == 0) then
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
    end if

    ! Рассылка данных всем процессам
    call MPI_Bcast(apartments, elements_edu * fields, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
    call MPI_Bcast(test_apartments, elements_test * fields, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

    ! Распределение работы между процессами
    chunk_size = elements_test / num_procs
    extra_work = mod(elements_test, num_procs)
    
    if (rank < extra_work) then
        start_idx = rank * (chunk_size + 1) + 1
        end_idx = start_idx + chunk_size
    else
        start_idx = rank * chunk_size + extra_work + 1
        end_idx = start_idx + chunk_size - 1
    end if

    ! Выделение памяти для локальных результатов
    allocate(local_euclidean_predict(start_idx:end_idx))
    allocate(local_manhattan_predict(start_idx:end_idx))
    allocate(local_euclidean_accuracy(start_idx:end_idx))
    allocate(local_manhattan_accuracy(start_idx:end_idx))

    ! === ПАРАЛЛЕЛЬНАЯ РЕАЛИЗАЦИЯ АЛГОРИТМА KNN ===
    
    do i = start_idx, end_idx
        euclidean_predict_price = 0
        manhattan_predict_price = 0
        
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

        local_euclidean_predict(i) = euclidean_predict_price
        local_manhattan_predict(i) = manhattan_predict_price
        local_euclidean_accuracy(i) = real(euclidean_predict_price)/test_apartments(i, 5)
        local_manhattan_accuracy(i) = real(manhattan_predict_price)/test_apartments(i, 5)
    end do

    ! Сбор результатов на процессе 0
    if (rank == 0) then
        ! Инициализация массивов для сбора результатов
        euclidean_accuracy = 0.0
        manhattan_accuracy = 0.0
        
        ! Сохранение результатов от процесса 0
        do i = start_idx, end_idx
            euclidean_accuracy(i) = local_euclidean_accuracy(i)
            manhattan_accuracy(i) = local_manhattan_accuracy(i)
        end do
        
        ! Получение результатов от других процессов
        do i = 1, num_procs - 1
            ! Определение диапазона для процесса i
            if (i < extra_work) then
                start_idx = i * (chunk_size + 1) + 1
                end_idx = start_idx + chunk_size
            else
                start_idx = i * chunk_size + extra_work + 1
                end_idx = start_idx + chunk_size - 1
            end if
            
            ! Получение данных
            call MPI_Recv(euclidean_accuracy(start_idx:end_idx), end_idx - start_idx + 1, &
                         MPI_REAL, i, 1, MPI_COMM_WORLD, status, ierr)
            call MPI_Recv(manhattan_accuracy(start_idx:end_idx), end_idx - start_idx + 1, &
                         MPI_REAL, i, 2, MPI_COMM_WORLD, status, ierr)
        end do
        
        ! Вывод результатов
        print *, 'Тестовая выборка:'
        print *, '       Этаж', '    Кол-во комнат', '   Площадь', '   Район ID',&
         '  Реал. цена', '    Евклид', '     Точность', '        Манхеттен', '   Точность'
        
        do i = 1, min(10, elements_test)  ! Выводим только первые 10 результатов
            euclidean_predict_price = int(test_apartments(i, 5) * euclidean_accuracy(i))
            manhattan_predict_price = int(test_apartments(i, 5) * manhattan_accuracy(i))
            
            print *, test_apartments(i, :), euclidean_predict_price, euclidean_accuracy(i), &
                                            manhattan_predict_price, manhattan_accuracy(i)
        end do
        
        call cpu_time(end_time)
        total_time = end_time - start_time
        print *, 'Общее время выполнения: ', total_time, ' секунд'
        
    else
        ! Отправка результатов на процесс 0
        call MPI_Send(local_euclidean_accuracy(start_idx:end_idx), end_idx - start_idx + 1, &
                     MPI_REAL, 0, 1, MPI_COMM_WORLD, ierr)
        call MPI_Send(local_manhattan_accuracy(start_idx:end_idx), end_idx - start_idx + 1, &
                     MPI_REAL, 0, 2, MPI_COMM_WORLD, ierr)
    end if

    ! Освобождение памяти
    deallocate(local_euclidean_predict, local_manhattan_predict)
    deallocate(local_euclidean_accuracy, local_manhattan_accuracy)

    call MPI_Finalize(ierr)

end program lab_mpi
